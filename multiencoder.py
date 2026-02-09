"""
Re-implementation of
K. Kim et al., "Show Me the Instruments: Musical Instrument Retrieval From Mixture Audio,"
ICASSP 2023, Rhodes Island, Greece, doi: 10.1109/ICASSP49357.2023.10097162

for a proper comparison with the contrastive embeddings obtained with the AST models.

TODO compared to the original paper:
- use AST instead of (much less powerfull) CNNs
- use a smaller amount of multi-encoder outputs, to make the task easier (e.g. 3 outputs for mixes made of 3 tracks)

TODO for this implementation:
- reload a pre-trained AST model for classification on ALL DIFFERENT instruments (weights will remain fixed)
- allow training
- timbre mix eval (in another file): optionally allow a different model to generate single-track embeddings
    and the multi-track mixture's embedding (so we don't need to rewrite the whole evaluation procedure)

"""

import argparse
import pathlib

import wandb
import yaml
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import mixdataset
from AST import ASTFineTuned
from trainer import TrainerBase


class MultiEncoderModel(torch.nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()

        # - - - Build 2 ASTs here, with their own config - - -
        # The single has its config already in its checkpoint
        self.stem_encoder = ASTFineTuned.from_checkpoint(config['model']['pretrained_single'])

        # The self.config will be considered as the multi encoder's config
        self.config = config
        self.config['model']['pretrained'] = self.config['model']['pretrained_multi']
        # Labels: a pre-defined, fixed set of embeddings will be retrieved at output
        self.config['dataset']['id2label'] = {}
        label_id = 0
        for embed_index in range(self.n_output_embeds):
            for embed_j in range(self.config['model']['embeddings_size']):
                self.config['dataset']['id2label'][label_id] = f'e{embed_index}_{embed_j}'
                label_id += 1
        # TODO use a custom AST w/ a custom compute_audio_embeddings method?
        self.mixture_encoder = ASTFineTuned.from_custom_config(self.config)

    @property
    def n_output_embeds(self):
        """ Number of output embeddings for a given mixture-of-instruments audio. """
        return len(self.config['model']['instrument_groups'])

    @property
    def device(self):
        return self.mixture_encoder.device

    def estimate_unmixed_embeddings(self, mixture_audios):
        n_mixtures = mixture_audios.shape[0]
        unmixed_outputs = self.mixture_encoder(mixture_audios)
        unmixed_embeds = unmixed_outputs[0].logits.view(n_mixtures, self.n_output_embeds, -1)
        return unmixed_embeds

    def forward(self, mixture_audios, mixture_ids, stems_audios, stems_ids):
        n_mixtures = mixture_audios.shape[0]
        assert stems_audios.shape[0] // n_mixtures == self.n_output_embeds

        # Retrieve the hidden embeddings (we don't care about the pre-training classification outputs)
        #     and Reshape to add a channels dimension (may be useful later for permutations-based loss)
        stems_outputs = self.stem_encoder(stems_audios)
        stems_embeds = stems_outputs[1]
        assert len(stems_embeds.shape) == 2, "2D embeddings (no multi-channel audio yet) expected at this point"
        embeds_size = stems_embeds.shape[1]
        stems_embeds = stems_embeds.view(n_mixtures, self.n_output_embeds, embeds_size)
        #  Retrieve the "logits" considered as concatenated embeddings
        unmixed_embeds = self.estimate_unmixed_embeddings(mixture_audios)
        assert unmixed_embeds.shape == stems_embeds.shape

        return stems_embeds, unmixed_embeds

    def save_checkpoint(self, path: pathlib.Path):
        """ Save the 2 sub-models in a single file """
        checkpoint = {
            'config': self.config,
            'stem_encoder': {
                'model_state_dict': self.stem_encoder.state_dict(), 'custom_config': self.stem_encoder.custom_config},
            'mixture_encoder': {
                'model_state_dict': self.mixture_encoder.state_dict(), 'custom_config': self.mixture_encoder.custom_config},
        }
        torch.save(checkpoint, path)

    @staticmethod
    def from_checkpoint(checkpoint_path: Union[str, pathlib.Path]):
        raise NotImplementedError("Not yet implemented for this model")  # TODO implement this
        checkpoint = torch.load(checkpoint_path)  # FIXME Code below copied from AST.py
        # Check that git config is OK...
        current_git_state, checkpoint_git_state = utils.get_git_info(), checkpoint['custom_config']['git']
        for k in current_git_state.keys():
            if current_git_state[k] != checkpoint_git_state[k]:
                warnings.warn(f"\nGit {k} is different in the checkpoint and in the current code\n\tcurrent: {current_git_state[k]}, checkpoint: {checkpoint_git_state[k]}")
        model = ASTFineTuned.from_custom_config(checkpoint['custom_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


class MultiEncoderTrainer(TrainerBase):
    def __init__(self, config: Dict[str, Any], secrets_path="secrets.yaml"):
        super().__init__(config, secrets_path)
        assert self.config['model']['base'] == 'AST' and self.config['model']['type'] == 'MultiEncoder'

        for split in ["train", "valid", "test"]:
            dataset_kwargs = {
                'use_small_dataset': self.config['dataset']['use_small_dataset'][split],
                'split': split,
                'pad': 'right',
                'target_sr': self.config['model']['sr'],
                'exclude_augmented_train': (not self.config['dataset']['training_data_augmentation']),
            }
            self.datasets[split] = mixdataset.MixDataset(
                instrument_group_source=self.config['model']['instrument_group_source'],
                families=self.config['model']['instrument_groups'],
                use_single_notes_as_references=True,
                contrastive_dataloader=True,
                **dataset_kwargs,
            )
            self.dataloaders[split] = self.datasets[split].get_dataloader(
                batch_size=self.config['training']['batch_size'],
                num_workers=self.config['training']['num_workers'],
            )

        # TODO Allow the MultiEnc to load its corresponding single-enc itself? (and build the labels itself also)
        self.model = MultiEncoderModel(self.config)

    def init_optimizer(self):
        # The single-instrument (stem) encoder won't be trained here - only optimize the "demixing" model
        self.optimizer = torch.optim.Adam(self.model.mixture_encoder.parameters(), lr=self.config['training']['start_lr'])

    def split_minibatch(self, minibatch: tuple[torch.Tensor]):
        """ Splits and checks the minibatch and how audios (mixed or not) are grouped.
        TODO Also returns the single-instrument stems and mixtures separately.
        """
        audios, instruments_UIDs, groups_indices = minibatch
        n_mixtures = groups_indices.max().item()
        n_stems_per_mix = (audios.shape[0] // n_mixtures) - 1
        assert n_stems_per_mix == self.model.n_output_embeds
        # In the audios, single-instrument stems are expected first, then the corresponding mix
        # (same index abs value, but negative). E.g.: 1, 1, 1, -1, 2, 2, 2, -2, 3, 3, ... w/ 3 stems per mix
        expected_groups_indices = sum([[mix_id, ]*n_stems_per_mix + [-mix_id] for mix_id in range(1, n_mixtures+1)], [])
        assert torch.all(groups_indices == torch.tensor(expected_groups_indices, device=groups_indices.device))
        # Now separate the audios (for the single-instrument and mixture models). These operations don't need
        # to be differentiable, and are performed on the CPU (so using many small ops is OK)
        stems_mask = (groups_indices > 0)
        stems_ids = groups_indices[stems_mask]
        stems_audios = audios[stems_mask, :]
        stems_instr_UIDs = np.asarray(instruments_UIDs)[stems_mask].tolist()
        mixtures_mask = (groups_indices < 0)
        mixture_ids = groups_indices[mixtures_mask]
        mixture_audios = audios[mixtures_mask, :]
        mixture_instr_UIDs = np.asarray(instruments_UIDs)[mixtures_mask].tolist()
        # Move to GPU before returning (not the UIDs... these are unpadded strings)
        return (mixture_audios.to(self.device), mixture_ids.to(self.device), mixture_instr_UIDs,
                stems_audios.to(self.device), stems_ids.to(self.device), stems_instr_UIDs)

    def train_step(self, step_i: int, minibatch):
        mixture_audios, mixture_ids, mixture_instr_UIDs, stems_audios, stems_ids, stems_instr_UIDs \
            = self.split_minibatch(minibatch)

        # TODO Build a forward method that handles sub-models' outputs
        stems_embeds, unmixed_embeds = self.model(mixture_audios, mixture_instr_UIDs, stems_audios, stems_instr_UIDs)
        loss = self.distance_loss(stems_embeds, unmixed_embeds)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Quick eval (not the full valid/test eval with extensive metrics and plots)
        with torch.no_grad():
            current_lr = self.optimizer.param_groups[0]['lr']
            log_data = {"loss/train": loss.item(), 'lr': current_lr}
        self.wandb_run.log(data=log_data, step=step_i)

        self.scheduler.step()

    def distance_loss(self, stems_embeds, unmixed_embeds):
        # Cosine distance loss (don't use the one in maths.py: it computes distances for all possible pairs)
        stems_embeds_norm = F.normalize(stems_embeds, p=2, dim=-1)
        unmixed_embeds_norm = F.normalize(unmixed_embeds, p=2, dim=-1)
        cosine_similarities = (stems_embeds_norm * unmixed_embeds_norm).sum(dim=-1)
        loss = torch.mean(1.0 - cosine_similarities)

        # TODO Try the PIT loss? Issue later: we would need to write a different search method... because the position
        # of outputs would not be quaranteed anymore

        return loss

    def eval_performance(self, training_step_i: int, dataset_split: str = 'valid'):
        """ Performs the evaluation on the whole validation or test dataset (quite long!) """
        assert dataset_split in ['test', 'valid']
        dataset, dataloader = self.datasets[dataset_split], self.dataloaders[dataset_split]

        # Compute valid|test loss
        losses = []
        with torch.no_grad():
            losses = []
            for minibatch in tqdm(
                    dataloader, total=len(dataloader), desc=f"{dataset_split.title()} step", position=1, leave=False):
                mixture_audios, mixture_ids, mixture_instr_UIDs, stems_audios, stems_ids, stems_instr_UIDs \
                    = self.split_minibatch(minibatch)
                stems_embeds, unmixed_embeds = self.model(mixture_audios, mixture_instr_UIDs, stems_audios,
                                                          stems_instr_UIDs)
                loss = self.distance_loss(stems_embeds, unmixed_embeds)
                losses.append(loss.item())
                break  # FIXME TEMP, REMOVE
        log_data = {f'loss/{dataset_split}': np.mean(losses),}
        # TODO Save if validation loss is the best so far

        # Timbre Mix Search - using the custom class for this model
        import multienc_mixsearch  # Import here to avoid circular imports

        reference_datasets = ['train', 'valid'] + (['test'] if dataset_split == 'test' else [])
        timbre_mix_evaluator = multienc_mixsearch.TimbreMixSearchEvaluator(
            multi_encoder_model=self.model,
            sr=self.config['model']['sr'],
            batch_size=self.config['training']['batch_size'],
            use_cuda=('cuda' in self.config['training']['device']),
            num_workers=(self.config['training']['num_workers'] * 4),
            verbose=True,
            reference_datasets=reference_datasets,
            eval_split=dataset_split,
            use_cache=False,
        )
        estimated_stems_embeds_df, metrics = timbre_mix_evaluator.perform_eval()
        main_metrics = timbre_mix_evaluator.select_main_metrics(metrics)  # Cosine distance only for this model
        figs = {}  #  timbre_mix_evaluator.plot_eval_sounds_df(eval_sounds_df)  # TODO Get some figs... maybe don't care?
        #rng = np.random.default_rng(seed=training_step_i)  # TODO get some audio demo !
        #audios = timbre_mix_evaluator.get_audios_for_mix(
        #    eval_sounds_df, dataset_index=rng.integers(0, len(timbre_mix_evaluator.mix_dataset)))
        # Log metrics and plots
        log_data.update({f'timbremix/{k}/{dataset_split}': v for k, v in main_metrics.items()})
        for fig_name, fig in figs.items():
            log_data[f'timbremixfig/{fig_name}/{dataset_split}'] = wandb.Image(fig)
        # For the non-main metrics: don't log all of them (take up too much screen space), log them as a table
        mix_metrics_table = wandb.Table(columns=["metric", "value"])
        for key, value in metrics.items():
            mix_metrics_table.add_data(key, value)
        log_data[f"timbremixdetails/{dataset_split}"] = mix_metrics_table

        self.wandb_run.log(data=log_data, step=training_step_i)  # Call only once per step



def main():
    """Entry point for the multiencoder pipeline.

    Accepts a positional YAML config path and runs a single configuration.
    """
    parser = argparse.ArgumentParser(description="Run multiencoder with a specified YAML config file.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        base_config: Dict[str, Any] = yaml.safe_load(f)

    # Single-run config path
    print("Loaded multiencoder configuration from:", args.config)
    # Placeholder: instantiate and run multiencoder logic with base_config here.
    
    # TODO we should allow other option than training... But that's it for now.
    trainer = MultiEncoderTrainer(base_config)
    trainer.run()
    

if __name__ == "__main__":
    main()
