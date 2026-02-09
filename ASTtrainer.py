import argparse
import copy
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
import transformers
import wandb
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import distance_metrics
from torch.nn import functional as F
from tqdm import tqdm

import maths
import mergeddataset
import mixdataset
import utils
from AST import ASTFineTuned
import timbresearch
import timbremixsearch
from trainer import TrainerBase



class ASTTrainer(TrainerBase):
    def __init__(self, config: Dict[str, Any], secrets_path="secrets.yaml"):
        super().__init__(config, secrets_path)
        assert self.config['model']['base'] == 'AST'

        # Load the dataset(s) - required to retrieve labels to configure the model's new output layer
        # Set some default values in the config (for recently added fields)
        self._set_default_config_values()

        for split in ["train", "valid", "test"]:
            dataset_kwargs = {
                'use_small_dataset': self.config['dataset']['use_small_dataset'][split],
                'split': split,
                'pad': 'right',
                'target_sr': self.config['model']['sr'],
                'exclude_augmented_train': (not self.config['dataset']['training_data_augmentation']),
            }
            if self.config['model']['type'] == 'InstrumentGroupClassification':
                self.datasets[split] = mergeddataset.MIDIInstrumentGroupClassificationDataset(**dataset_kwargs)
            elif self.config['model']['type'] == 'InstrumentDirectClassification':
                self.datasets[split] = mergeddataset.InstrumentDirectClassificationDataset(**dataset_kwargs)
            elif self.config['model']['type'] == 'MultiLabelClassification':
                self.datasets[split] = mergeddataset.MultiLabelDataset(**dataset_kwargs)
            elif self.config['model']['type'] == 'ContrastiveEmbeddings':
                hard_negatives_ratio = self.config['training']['hard_negatives_ratio'] if split == 'train' else 0.0
                self.datasets[split] = mergeddataset.ContrastiveInstrumentDataset(
                    hard_negatives_ratio=hard_negatives_ratio,
                    **dataset_kwargs
                )
            elif self.config['model']['type'] == 'ContrastiveMixEmbeddings':
                self.datasets[split] = mixdataset.MixDataset(
                    instrument_group_source=self.config['model']['instrument_group_source'],
                    families=self.config['model']['instrument_groups'],
                    use_single_notes_as_references=True,
                    contrastive_dataloader=True,
                    **dataset_kwargs,
                )
            else:
                raise NotImplementedError(f"{self.config['model']['type']} has no associated dataset")
            self.dataloaders[split] = self.datasets[split].get_dataloader(
                batch_size=self.config['training']['batch_size'],
                num_workers=self.config['training']['num_workers'],
            )

        # Save those labels in the config (labels are mandatory here; will be used by the model)
        if self.is_classifier:
            if ('id2label' not in self.config['dataset'].keys()) or (self.config['dataset']['id2label'] is None):
                print("config['dataset']['id2label'] is not provided and will be automatically set using the training "
                      "dataset's id2label")
                self.config['dataset']['id2label'] = self.datasets['train'].id2label
        else:
            self.config['dataset']['id2label'] = {}

        # Build and load the model
        self.model = ASTFineTuned.from_custom_config(self.config)

        self._best_valid_loss = np.inf  # For tracking the best model

    def _set_default_config_values(self):
        # Enable training data augmentation unless explicitly disabled in config
        self.config['dataset']['training_data_augmentation'] = bool(
            self.config['dataset'].get('training_data_augmentation', True)
        )
        # hard negatives ratio: only useful/meaningful for contrastive training; set to -1.0 if not provided
        self.config['training']['hard_negatives_ratio'] = float(self.config['training'].get('hard_negatives_ratio', -1.0))

    @property
    def id2datasource(self):
        return self.datasets['train'].id2datasource

    @property
    def datasource2id(self):
        return self.datasets['train'].datasource2id


    def split_minibatch(self, minibatch: tuple[torch.Tensor]):
        # Don't send everything to the GPU...
        if self.is_classifier:
            dataset_indices, data_source_ids, audios, targets = minibatch
            targets = targets.to(self.device)
            groups_indices, instruments_UIDs = None, None
        else:
            if self.config['model']['type'] == 'ContrastiveEmbeddings':
                dataset_indices, audios, groups_indices = minibatch
                instruments_UIDs = None
            elif self.config['model']['type'] == 'ContrastiveMixEmbeddings':
                audios, instruments_UIDs, groups_indices = minibatch
                dataset_indices = None
            else:
                raise NotImplementedError(f"Minibatch splitting not implemented for {self.config['model']['type']=}")
            targets, data_source_ids = None, None
        return dataset_indices, data_source_ids, instruments_UIDs, audios, targets, groups_indices

    def train_step(self, step_i: int, minibatch):
        """ Performs a single training step (a single minibatch) """
        dataset_indices, data_source_ids, instruments_UIDs, audios, targets, groups_indices = self.split_minibatch(minibatch)

        # TODO Here, some audio data augmentation, maybe some extra FX, .... (but targets would need to change
        #    accordingly)

        # The model will move itself the audios (converted to specs) to the GPU
        outputs, last_hidden_embeddings = self.model(audios)

        loss = self.loss(
            output_logits=outputs.logits, targets=targets,
            last_hidden_embeddings=last_hidden_embeddings, groups_indices=groups_indices,
        )
        # Backward pass and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Quick eval (not the full valid/test eval with extensive metrics and plots)
        with torch.no_grad():
            current_lr = self.optimizer.param_groups[0]['lr']
            log_data = {"loss/train": loss.item(), 'lr': current_lr}

            if self.is_classifier:
                output_logits, targets = outputs.logits.cpu(), targets.cpu()  # Avoid GPU calls for such small tensors
                assert len(output_logits.shape) == len(targets.shape) == 2

            if self.config['model']['type'] in ['InstrumentGroupClassification', 'InstrumentDirectClassification', ]:
                # compute accuracy.... even if batch size is small
                predicted_indices, target_indices = output_logits.argmax(dim=1), targets.argmax(dim=1)
                accuracy = (predicted_indices == target_indices).sum().item() / target_indices.shape[0]
                log_data["acc/train"] = accuracy

            elif self.config['model']['type'] == 'MultiLabelClassification':
                log_data.update(self.multi_label_metrics(output_logits, targets, 'train'))

            elif self.config['model']['type'] in ['ContrastiveEmbeddings', 'ContrastiveMixEmbeddings']:
                pass  # Nothing but the loss to log here...

            else:
                raise ValueError(f"Unsupported {self.config['model']['type']=}")
        self.wandb_run.log(data=log_data, step=step_i)

        self.scheduler.step()

    def loss(self, output_logits: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None,
             last_hidden_embeddings: Optional[torch.Tensor] = None, groups_indices: Optional[torch.Tensor] = None):
        if self.is_classifier:
            assert output_logits is not None and targets is not None
            if self.config['training']['criterion'] == 'CE':
                return F.cross_entropy(output_logits, targets)
            elif self.config['training']['criterion'] == 'BCE':
                return F.binary_cross_entropy_with_logits(output_logits, targets)
            else:
                raise NotImplementedError(f"{self.config['training']['criterion']=} ({self.is_classifier=})")

        else:  # NOT a classifier; no target should be provided
            assert targets is None and last_hidden_embeddings is not None and groups_indices is not None

            if self.config['model']['type'] == 'ContrastiveEmbeddings':
                # The following losses assume that all batched items are provided in pairs, i.e. groups_indices
                #  (representing what 'group', i.e. instrument, an item belongs to) should be 0, 0, 1, 1, 2, 2, ...
                N = groups_indices.shape[0]  # Minibatch size
                assert len(groups_indices.shape) == 1 and ((N % 2) == 0)
                even_pos_mask = torch.zeros((N, ), dtype=torch.bool)
                even_pos_mask[torch.arange(0, N, 2)] = True
                assert torch.all(groups_indices[even_pos_mask] == groups_indices[~even_pos_mask])

                embeds_0, embeds_1 = last_hidden_embeddings[even_pos_mask], last_hidden_embeddings[~even_pos_mask]
                if self.config['training']['criterion'].lower() == 'triplet':
                    return maths.triplet_loss_from_pairs(embeds_0, embeds_1)  # TODO margin arg... from config
                elif self.config['training']['criterion'].lower() == 'InfoNCE'.lower():
                    # TODO Temperature from config OR temperature as a learned model parameter (see CLIP)
                    return maths.infonce_loss(embeds_0, embeds_1)
                else:
                    raise NotImplementedError(f"{self.config['training']['criterion']=} ({self.is_classifier=})")

            elif self.config['model']['type'] == 'ContrastiveMixEmbeddings':
                # TODO InfoNCE hparam from config and/or learned
                if self.config['training']['criterion'].lower() == 'InfoNCE'.lower():
                    return maths.infonce_mix_loss(last_hidden_embeddings, groups_indices)
                elif self.config['training']['criterion'].lower() == 'InfoNCE-full'.lower():
                    return maths.infonce_mix_full_loss(last_hidden_embeddings, groups_indices, symmetric=True)
                elif self.config['training']['criterion'].lower() == 'InfoNCE-full-not-symmetrical'.lower():
                    return maths.infonce_mix_full_loss(last_hidden_embeddings, groups_indices, symmetric=False)
                elif self.config['training']['criterion'].lower() == 'triplet':  # TODO margin from cfg
                    return maths.triplet_loss_from_mix(last_hidden_embeddings, groups_indices, full=False)
                elif self.config['training']['criterion'].lower() == 'triplet-full':  # TODO margin from cfg
                    return maths.triplet_loss_from_mix(last_hidden_embeddings, groups_indices, full=True)
                else:
                    raise NotImplementedError(f"{self.config['training']['criterion']=}")
            else:
                raise ValueError(f"Unsupported loss for {self.config['model']['type']=}")

    def multi_label_metrics(
            self, output_logits: torch.Tensor, targets: torch.Tensor, dataset_split: str,
            prob_threshold=0.5,   # FIXME Should adapt that for separating pos/neg labels...
    ):
        dataset = self.datasets[dataset_split]
        metrics = {}

        assert (output_logits.shape == targets.shape) and (len(targets.shape) == 2)
        predicted_probs = F.sigmoid(output_logits)
        predicted_bool = (predicted_probs >= prob_threshold)
        target_bool = (targets >= prob_threshold)
        # For instr family, only 0 or 1 label is allowed at the moment.... so normal accuracy also computed later
        for k in ['all', 'fx', 'instr_family']:
            # Save that for 'all', but also for 'instrument_family' and 'FX' separately...
            #    Some instruments have no clear label
            metrics[f'exact_match/{k}'] = torch.all((predicted_bool == target_bool), dim=1).float().mean().item()
            metrics[f'hamming_acc/{k}'] = (predicted_bool == target_bool).float().mean().item()

            if dataset_split != 'train':  # only a single minibatch is provided during training
                # Metrics that need more data (e.g. with all labels being present, ...)
                if k == 'all':
                    _target_bool, _pred_bool, _pred_probs = target_bool, predicted_bool, predicted_probs
                elif k in ['instr_family', 'fx']:
                    if k == 'instr_family':
                        subset_indices = dataset.instrument_family_ids
                    else:
                        subset_indices = dataset.fx_ids
                    _target_bool = target_bool[:, subset_indices]
                    _pred_bool = predicted_bool[:, subset_indices]
                    _pred_probs = predicted_probs[:, subset_indices]
                else:
                    raise KeyError(k)
                metrics[f'ROC_AUC/{k}'] = sklearn.metrics.roc_auc_score(_target_bool, _pred_probs, average='macro')
                metrics[f'mAP/{k}'] = sklearn.metrics.average_precision_score(_target_bool, _pred_probs, average='macro')
                # precision, recall and F1 should probably not be computed when 0 labels are in the target
                if k in ['instr_family', 'fx']:
                    has_label_mask = _target_bool.sum(dim=1) >= 1
                    _target_bool = target_bool[has_label_mask, :]
                    _pred_bool = predicted_bool[has_label_mask, :]
                metrics[f'precision/{k}'] = sklearn.metrics.precision_score(_target_bool, _pred_bool, average='samples')
                metrics[f'recall/{k}'] = sklearn.metrics.recall_score(_target_bool, _pred_bool, average='samples')
                metrics[f'F1_score/{k}'] = sklearn.metrics.f1_score(_target_bool, _pred_bool, average='samples')

        #  Compute some usual (basic, simple) accuracies for instrument source;
        #  (NOT family, some patches have no clear family!)
        for metric_name, indices_subset in [
            ('acc/instr_source', dataset.instrument_source_ids), ('acc/instr_family', dataset.instrument_family_ids)]:
            sub_pred_ids = output_logits[:, indices_subset].argmax(dim=1)
            sub_targets_bool = target_bool[:, indices_subset]
            # But for family, allow empty families (exclude them from the accuracy)
            if metric_name == 'acc/instr_family':
                num_target_classes = sub_targets_bool.sum(dim=1)  # vector result, For each item
                if torch.any(num_target_classes > 1):
                    warnings.warn("This accuracy computation requires 0 or 1 instrument family. Skipping...")
                    continue
                # Keep only items with a single instrument family
                sub_pred_ids = sub_pred_ids[num_target_classes == 1]
                sub_targets_bool = sub_targets_bool[num_target_classes == 1, :]
            # raise a warning if multiple (or no) positives and found in any category...
            if not torch.all(sub_targets_bool.sum(dim=1) == 1):
                warnings.warn(f"For {metric_name=}, some items have either no label or more than one label. "
                              f"The basic (single-label) classification accuracy can't be computed.")
                continue
            sub_target_ids = sub_targets_bool.float().argmax(dim=1)
            metrics[metric_name] = (sub_target_ids == sub_pred_ids).float().mean().item()

        # top-K accuracy: average proportion of true labels present in the top-K predicted labels,
        # normalized by the minimum of the number of true labels and K.
        ranked_pred_indices = torch.argsort(output_logits, dim=1, descending=True)
        for k in [2, 3, 5]:
            top_k_pred_indices = ranked_pred_indices[:, :k]
            proportions = []
            for i in range(target_bool.size(0)):  # Suboptimal for loop, using operations on Python sets...
                true_labels = set(torch.nonzero(target_bool[i], as_tuple=True)[0].tolist())
                if len(true_labels) == 0:
                    continue  # Just ignore
                top_k_labels = set(top_k_pred_indices[i].tolist())
                proportions.append(len(true_labels & top_k_labels) / min(len(true_labels), k))
            metrics[f"acc/top{k}"] = sum(proportions) / len(proportions)

        # Add the dataset split as a suffix for all metrics
        return {f'{k}/{dataset_split}': m for k, m in metrics.items()}

    def eval_performance(self, training_step_i: int, dataset_split='valid'):
        """ Performs the evaluation on the whole validation or test dataset (quite long!) """
        assert dataset_split in ['test', 'valid']
        dataset, dataloader = self.datasets[dataset_split], self.dataloaders[dataset_split]

        # Direct classifier for training instruments cannot be used for valid|test classification (only use embeddings)
        evaluate_classification = (
                self.is_classifier and not self.config['model']['type'] == 'InstrumentDirectClassification')

        # Retrieve all results in a DataFrame, used later for precise display of values
        if evaluate_classification:
            results_df: pd.DataFrame = copy.deepcopy(dataset.df)
            results_df['target_probs'] = [torch.empty((0, ))] * len(results_df)
            results_df['output_logits'] = [torch.empty((0, ))] * len(results_df)

        # Compute all losses (cannot be done for some models...)
        if self.config['model']['type'] not in ['InstrumentDirectClassification', ]:
            with torch.no_grad():
                losses = []
                for minibatch in tqdm(
                        dataloader, total=len(dataloader), desc=f"{dataset_split.title()} step", position=1, leave=False):
                    # minibatch is handled differently for each model type
                    dataset_indices, data_source_ids, instruments_UIDS, audios, targets, groups_indices \
                        = self.split_minibatch(minibatch)
                    outputs, last_hidden_embeddings = self.model(audios)

                    losses.append(self.loss(outputs.logits, targets, last_hidden_embeddings, groups_indices).item())
                    if evaluate_classification:
                        #  Don't store outputs logits (nor targets) is there's no classifier
                        # Store PyTorch Tensor data in the dataframe... For concat later (to retrieve 2D tensors)
                        output_logits, target_probs = outputs.logits.cpu(), targets.cpu()
                        dataset_indices = dataset_indices.cpu().numpy()
                        for batch_i, dataset_i in enumerate(dataset_indices):
                            # TODO Also store embeddings? Maybe not necessary...
                            # Don't do sequential indexing here (row then col) otherwise the assignation is done on a copy
                            results_df.iat[dataset_i, results_df.columns.get_loc('output_logits')] = output_logits[batch_i, :]
                            results_df.iat[dataset_i, results_df.columns.get_loc('target_probs')] = target_probs[batch_i, :]
                    #warnings.warn(f"FIXME DISABLED {dataset_split.title()}")
                    #break  # FIXME EEEEEEE

        else:  # Models with no validation/test loss... default dummy 0.0
            losses = [0.0] * len(dataloader)

        #       = = = = = = = = = = =           (MIDI) Instrument Group Classification           = = = = = = = = = = =
        if self.config['model']['type'] == 'InstrumentGroupClassification':
            log_data = {}
            # Columns only used for single-label classification
            results_df['target_id'] = results_df['target_probs'].apply(lambda x: x.argmax().item())
            results_df['predicted_id'] = results_df['output_logits'].apply(lambda x: x.argmax().item())
            for k in ['target', 'predicted']:
                results_df[f"{k}_label"] = results_df[f"{k}_id"].apply(lambda x: dataset.id2label[x])

            # Check for mistakes in indexing: targets (as str) should be the MIDI instrument group
            assert np.all(results_df['midi_instrument_group'] == results_df['target_label'])
            results_df['correct_prediction'] = (results_df['target_id'] == results_df['predicted_id'])

            # Compute accuracies separately for each data source...
            sources = list(set(results_df['data_source'].values))
            for s in ([''] + sources):
                if s == '':
                    sub_df, suffix = results_df, ''
                else:
                    sub_df, suffix = results_df[results_df['data_source'] == s], f'_{s}'
                log_data[f"acc{suffix}/{dataset_split}"] = sub_df['correct_prediction'].sum() / len(sub_df)

            # Also log some Seaborn plots (e.g. detailed accuracies for each category)
            g = sns.catplot(
                kind='count', data=results_df, y='midi_instrument_group', col='data_source',
                hue='correct_prediction', hue_order=[True, False], sharex=False, sharey=True,
            )
            log_data[f'pred_by_instr/{dataset_split}'] = wandb.Image(g.figure)
            plt.close()
            # Also computes plot the accuracies
            accuracies_df = []
            for i, s in enumerate(sources):
                sub_df = results_df[results_df['data_source'] == s]
                for label in set(sub_df['midi_instrument_group']):
                    sub_sub_df = sub_df[sub_df['midi_instrument_group'] == label]
                    accuracies_df.append({
                        'data_source': s, 'midi_instrument_group': label,
                        'accuracy': sub_sub_df['correct_prediction'].sum() / len(sub_sub_df),
                    })
            accuracies_df = pd.DataFrame(accuracies_df)
            g = sns.catplot(
                data=accuracies_df, kind='bar', col='data_source', y='midi_instrument_group', x='accuracy', sharex=False
            )
            log_data[f'acc_by_instr/{dataset_split}'] = wandb.Image(g.figure)
            plt.close()

            self.wandb_run.log(data=log_data, step=training_step_i)

        #       = = = = = = = = = = =         (Training) Instrument Direct Classification         = = = = = = = = = = =
        elif self.config['model']['type'] == 'InstrumentDirectClassification':
            pass  # Can't be applied to valid or test instruments

        #       = = = = = = = = = = = = = =           Multi Label Classification           = = = = = = = = = = = = = =
        elif self.config['model']['type'] == 'MultiLabelClassification':
            # Compute an extended set of multi-label metrics (more than during training)
            output_logits = torch.vstack(tuple(results_df['output_logits'].values))
            output_probs = F.sigmoid(output_logits)
            target_probs = torch.vstack(tuple(results_df['target_probs'].values))
            log_data = self.multi_label_metrics(output_logits, target_probs, dataset_split)

            # Detailed plots: accuracies of instruments groups (excluding samples without an instr.)
            # Log as: 'pred_by_instr_family' and 'pred_by_instr_source'
            for subset_name, indices_subset, labels_subset in [
                ('instr_source', dataset.instrument_source_ids, dataset.instrument_source_labels),
                ('instr_family', dataset.instrument_family_ids, dataset.instrument_family_labels),
                #('fx', dataset.fx_ids),
            ]:
                sub_target_probs = target_probs[:, indices_subset]
                has_label_mask = sub_target_probs.sum(dim=1) > 0
                sub_target_probs = sub_target_probs[has_label_mask, :]
                sub_target_ids = sub_target_probs.argmax(dim=1)
                sub_pred_ids = output_logits[:, indices_subset][has_label_mask, :].argmax(dim=1)
                results_df[f'{subset_name}_correct_prediction'] = pd.array([pd.NA] * len(results_df), dtype="boolean")  # Nullable bool type
                results_df.loc[has_label_mask.numpy(), f'{subset_name}_correct_prediction'] = (sub_target_ids == sub_pred_ids).numpy()

                # Immediately plot the ROC curves. Not possible for FX... WandB ROC requires a single target label:
                #   the error from WandB was: "multilabel-indicator format is not supported"
                sub_output_probs = output_probs[:, indices_subset][has_label_mask, :]
                log_data[f'ROC/{subset_name}/{dataset_split}'] = wandb.plot.roc_curve(
                    sub_target_ids, sub_output_probs, labels=labels_subset)
                # Also plot the confusion matrices - very useful! Easy to read even with 10--15 classes
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                matrix = sklearn.metrics.confusion_matrix(sub_target_ids, sub_pred_ids)
                sns.heatmap(matrix, annot=True, fmt='d', xticklabels=labels_subset, yticklabels=labels_subset, ax=ax)
                ax.set(xlabel='Predicted labels', ylabel='True labels', title='Confusion Matrix')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                fig.tight_layout()
                log_data[f'prediction_plot/{subset_name}_confusion/{dataset_split}'] = wandb.Image(fig)
                plt.close()
            # 2 sns plots
            g = sns.catplot(
                kind='count', data=results_df[~results_df['instrument_family'].isna()],
                y='instrument_family', col='data_source',
                hue='instr_family_correct_prediction', hue_order=[True, False], sharex=False, sharey=True,
            )
            log_data[f'prediction_plot/instr_family/{dataset_split}'] = wandb.Image(g.figure)
            plt.close()
            g = sns.catplot(
                kind='count', data=results_df[~results_df['instrument_source'].isna()],
                y='instrument_source', col='data_source',
                hue='instr_source_correct_prediction', hue_order=[True, False], sharex=False, sharey=True,
            )
            log_data[f'prediction_plot/instr_source/{dataset_split}'] = wandb.Image(g.figure)
            plt.close()

            # FX ROC curve, multi-label, not handled by WandB
            sub_target_probs = target_probs[:, dataset.fx_ids]
            sub_pred_probs = output_probs[:, dataset.fx_ids]
            # Initialize the plot
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            for i in range(sub_target_probs.shape[1]):
                fpr, tpr, _ = sklearn.metrics.roc_curve(sub_target_probs[:, i], sub_pred_probs[:, i])
                roc_auc = sklearn.metrics.auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{dataset.fx_labels[i]} (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.0], xlabel='False Positive Rate (FPR)', ylabel='True Positive Rate (TPR)', title='Receiver Operating Characteristic (ROC) Curves')
            ax.legend(loc="lower right")
            log_data[f'ROC/fx/{dataset_split}'] = wandb.Image(fig)
            plt.close()

            self.wandb_run.log(data=log_data, step=training_step_i)

        #       = = = = = = = = = = = = = =           Embeddings (pre-)training           = = = = = = = = = = = = = =
        elif self.config['model']['type'] in ['ContrastiveEmbeddings', 'ContrastiveMixEmbeddings']:
            pass  # Nothing else to eval...

        #       = = = = = = = = = =           Other model type           = = = = = = = = = =
        else:
            raise ValueError(f"Unsupported {self.config['model']['type']=}")

        #       = = = = = = = = = = =           Evaluation for any kind of model           = = = = = = = = = = =
        # Average loss
        avg_loss = np.mean(losses)
        log_data = {f"loss/{dataset_split}": avg_loss}
        self.wandb_run.log(data=log_data, step=training_step_i)
        # Save this model as 'best' if the loss is lower
        if dataset_split == 'valid':
            if avg_loss < self._best_valid_loss:
                self._best_valid_loss = avg_loss
                self.save_checkpoint(suffix='_best')

        # Timbre search
        reference_datasets = ['train', 'valid'] + (['test'] if dataset_split == 'test' else [])
        timbre_distance_metrics, main_distance_metric = ('l2', 'cosine'), 'l2'


        if self.config['dataset']['use_small_dataset'][dataset_split]:
            warnings.warn("Skipping timbresearch for that small dataset")

        elif 'MixEmbeddings' in self.config['model']['type']:

            timbre_mix_evaluator = timbremixsearch.TimbreMixSearchEvaluator(
                model=self.model,
                batch_size=self.config['training']['batch_size'],
                num_workers=(self.config['training']['num_workers'] * 4),  # Data loading is a main bottleneck
                use_cuda=True,
                use_cache=False,
                eval_split=dataset_split,
                reference_datasets=reference_datasets,
                verbose=True,
            )
            eval_sounds_df, metrics = timbre_mix_evaluator.perform_eval(distance_metrics=timbre_distance_metrics)
            main_metrics = timbre_mix_evaluator.select_main_metrics(metrics, main_distance_metric=main_distance_metric)
            figs = timbre_mix_evaluator.plot_eval_sounds_df(eval_sounds_df)
            rng = np.random.default_rng(seed=training_step_i)
            audios = timbre_mix_evaluator.get_audios_for_mix(
                eval_sounds_df, dataset_index=rng.integers(0, len(timbre_mix_evaluator.mix_dataset)))
            # Log metrics and plots
            log_data = {f'timbremix/{k}/{dataset_split}': v for k, v in main_metrics.items()}
            for fig_name, fig in figs.items():
                log_data[f'timbremixfig/{fig_name}/{dataset_split}'] = wandb.Image(fig)
            self.wandb_run.log(data=log_data, step=training_step_i)
            plt.close()
            # Full metrics: in a table (only for the final test)
            if dataset_split == 'test':
                metrics_table = wandb.Table(columns=["metric", "value"])
                for name, value in metrics.items():
                    metrics_table.add_data(name, value)
                self.wandb_run.log({f"timbremixdetails/{dataset_split}": metrics_table}, step=training_step_i)
            # Log audios
            log_data = {f'timbremixaudio/{dataset_split}/mix': wandb.Audio(
                audios['mix']['audio'], caption=audios['mix']['info'], sample_rate=timbre_mix_evaluator.sr
            )}
            for family in timbre_mix_evaluator.mix_dataset.mixed_families:
                for short_k, long_k in {'GT': 'GT_single_notes', 'matched': 'matched_single_notes'}.items():
                    log_data[f'timbremixaudio/{dataset_split}/{family}/{short_k}'] = wandb.Audio(
                        audios[long_k][family]['audio'], caption=audios[long_k][family]['info'], sample_rate=timbre_mix_evaluator.sr,
                    )
            self.wandb_run.log(data=log_data, step=training_step_i)
            # Save results' DF to W&B and local SSD (but only some cols for the CSV and W&B)
            timbre_pkl_path = self.run_storage_dir / f'timbremix_{dataset_split}.df.pkl'
            eval_sounds_df.to_pickle(timbre_pkl_path)
            cols_to_keep = ['split', 'dataset_index', ]
            for family in timbre_mix_evaluator.mix_dataset.mixed_families:
                cols_to_keep.append(f'{family}_GT_instrument_UID')
                cols_to_keep.append(f'{main_distance_metric}__{family}__GT_rank')
                cols_to_keep.append(f'{main_distance_metric}__{family}__top_10_UIDs')
            simpler_sounds_df = eval_sounds_df[cols_to_keep]
            timbre_csv_path = self.run_storage_dir / f'timbremix_{dataset_split}.csv'
            simpler_sounds_df.to_csv(timbre_csv_path)
            self.wandb_run.save(glob_str=timbre_csv_path, base_path=self.run_storage_dir)

        else: # Basic mono-instrument timbre search
            timbre_evaluator = timbresearch.TimbreSearchEvaluator(
                self.model,
                eval_split=dataset_split,
                reference_datasets=reference_datasets,
                use_small_dataset=self.config['dataset']['use_small_dataset'],
                batch_size=self.config['training']['batch_size'],
                verbose=True,
            )
            # L2 and Cosine only, Mahalanobis seems worse... But it should be tried with other models
            timbre_sounds_df, timbre_metrics = timbre_evaluator.perform_eval(distance_metrics=timbre_distance_metrics)
            log_data = {}
            for k, v in timbre_metrics.items():
                log_data[f'timbresearch/{k}/{dataset_split}'] = v
            timbre_figs = timbre_evaluator.plot_eval_sounds_df(timbre_sounds_df)
            for fig_name, fig in timbre_figs.items():
                log_data[f'timbresearchfig/{fig_name}/{dataset_split}'] = wandb.Image(fig)
            plt.close()
            self.wandb_run.log(data=log_data, step=training_step_i)
            # Save the TimbreDataFrame (erase any previously saved...) as .pkl and .csv, locally and into W&B
            timbre_pkl_path = self.run_storage_dir / f'timbresearch_{dataset_split}.df.pkl'
            timbre_sounds_df.to_pickle(timbre_pkl_path)
            # self.wandb_run.save(glob_str=timbre_pkl_path, base_path=self.run_storage_dir)  # Too big with the embeddings...
            # Aa very limited number of columns, for readability (W&B table) and disk size (.csv)
            cols_to_keep = [
                'instrument_UID', 'is_reference', 'instrument_family', 'fx',
                'midi_pitch', 'midi_velocity', 'midi_artist', 'midi_track',
            ]
            for m in timbre_distance_metrics:
                cols_to_keep += [f'{m}_best_match', f'{m}_top_5']
            simpler_timbre_sounds_df = timbre_sounds_df[cols_to_keep]
            timbre_csv_path = self.run_storage_dir / f'timbresearch_{dataset_split}.csv'
            simpler_timbre_sounds_df.to_csv(timbre_csv_path, index=False)
            self.wandb_run.save(glob_str=timbre_csv_path, base_path=self.run_storage_dir)
            # This table is disabled... Absolutely unreadable in W&B
            # timbre_sounds_table = wandb.Table(dataframe=simpler_timbre_sounds_df)
            # self.wandb_run.log(data={f'timbresearch/{dataset_split}': timbre_sounds_table}, step=training_step_i)

            # Log t-SNE or UMAP? (for reference notes only? otherwise: too many points...)
            # Not now... That's still too much info for W&B:
            #    "Note: we currently downsample to a random subset of 1000 rows and 50 dimensions for all three algorithms."

        pass


# =====================================================  MAIN  ========================================================
def main():
    _parser = argparse.ArgumentParser(description="Run with a specified YAML config file.")
    _parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    _args = _parser.parse_args()

    with open(_args.config, "r") as f:
        _base_config = yaml.safe_load(f)

    if 'sweep' not in _base_config:  # No sweep, single training run
        _trainer = ASTTrainer(_base_config)
        _trainer.run()

    #  if the sweep field exists: handle it
    else:
        print("Sweep configuration detected. Running sweep...")
        sweep_config = _base_config.pop('sweep')

        # Generate all configurations based on sweep parameters
        def generate_sweep_configs(base_config, sweep_config):
            # Start with a single base configuration
            all_configs = [copy.deepcopy(base_config)]

            # Function to update a specific nested key in a config
            def update_nested_key(config, path, key, value):
                target = config
                if path:
                    for part in path.split('.'):
                        if part not in target:
                            target[part] = {}
                        target = target[part]
                target[key] = value
                return config

            # Process each sweep parameter and generate all combinations
            def process_sweep_params(sweep_values, path=""):
                nonlocal all_configs
                for key, value in sweep_values.items():
                    current_path = f"{path}.{key}" if path else key

                    if isinstance(value, dict):
                        # If the value is a dictionary, recurse deeper
                        process_sweep_params(value, current_path)
                    elif isinstance(value, list) and len(value) > 0:
                        # If the value is a list, create a new config for each value
                        print(f"Sweeping over {current_path}: {value}")

                        new_configs = []
                        for config in all_configs:
                            for val in value:
                                # Create a new config with this value
                                new_config = copy.deepcopy(config)
                                new_config = update_nested_key(new_config, path, key, val)

                                # Update run name to reflect the sweep parameter
                                if 'run' in new_config and 'name' in new_config['run']:
                                    # Avoid adding the same parameter multiple times to the name
                                    param_str = f"_{key}_{val}"
                                    if param_str not in new_config['run']['name']:
                                        new_config['run']['name'] = f"{new_config['run']['name']}{param_str}"

                                new_configs.append(new_config)

                        # Replace the old configs with the new ones
                        all_configs = new_configs

            # Process all sweep parameters
            process_sweep_params(sweep_config)
            return all_configs

        # Generate all configurations based on sweep parameters
        sweep_configs = generate_sweep_configs(_base_config, sweep_config)

        # Run each configuration
        for i, config in enumerate(sweep_configs):
            print(f"\n========== Running sweep configuration {i+1}/{len(sweep_configs)} ==========")
            trainer = ASTTrainer(config)
            trainer.run()

if __name__ == "__main__":
    main()
