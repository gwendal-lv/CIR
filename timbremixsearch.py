import re
from pathlib import Path
from typing import Union, Optional, Sequence, List, Tuple
import argparse
import pickle
import os

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import wandb

import mixdataset
from timbresearch import TimbreSearchBase
import utils


class TimbreMixSearchEvaluator(TimbreSearchBase):
    def __init__(self, model, sr: Optional[int] = None, batch_size=32, use_cuda=True,
                 eval_split='valid', reference_datasets=('train', 'valid'),
                 exclude_augmentations=True, reference_note=None,
                 verbose=False, num_workers=0, use_cache=False):
        """
        Class for searching mixed timbres using a unified model that can be used to compute embeddings for all sounds
        (mixtures or individual samples).
        Currently relies on a dataset with only mixtures of the same families (e.g., always drums + bass + guitar).

        :param model: Must have .compute_audio_embeddings(...) and .to(device) methods, and a .sr integer field.
                        By default, this model will be used to compute the embeddings for ALL sounds (mixtures
                        and single-instrument sounds).
        :param sr: Sampling rate to use, if None will use model.sr
        :param batch_size: Batch size for processing
        :param eval_split: The sub-dataset to be evaluated
        :param reference_datasets: What splits should be used to compute reference embeddings
        :param use_small_dataset: Whether to use small dataset
        :param exclude_augmentations: Don't use data-augmented presets as references (anchors) for search
        :param reference_note: If given as (midi_pitch, midi_velocity), the same reference note (or a similar one)
                               will be used for all instruments. If None, the median pitch for each instrument,
                               and a default velocity value 100 will be used.
        :param verbose: Whether to print verbose output
        :param num_workers: Number of worker processes for data loading
        """
        # Call the base class constructor
        super().__init__(model=model, sr=sr, batch_size=batch_size, use_cuda=use_cuda,
                         eval_split=eval_split, reference_datasets=reference_datasets,
                         use_small_dataset=False,  # small dataset is not allowed for the mix dataset
                         exclude_augmentations=exclude_augmentations,
                         reference_note=reference_note, verbose=verbose, use_cache=use_cache)

        self.num_workers = num_workers

        # For this Mix evaluation, we use dedicated datasets.
        # However, we keep the self.datasets dict from the base class (used for reference single-note sounds)
        self.mix_dataset = mixdataset.MixDataset(
            split=self.eval_split,
            use_single_notes_as_references=True,  # Won't be used in practice (we'll use the mixes only, not the single-instru tracks)
            target_sr=self.sr,
            contrastive_dataloader=False,  # Just need to load samples for evaluation, not to mine positives/negatives
            use_cache=use_cache,
        )

    @property
    def embeddings_count(self):
        """ The number of embeddings that must be computed to perform the timbre search
            (reference sounds + eval sounds) """
        return self.sounds_df['is_reference'].sum() + len(self.mix_dataset)

    def compute_mix_embeddings(self) -> pd.DataFrame:
        """

        :return: DataFrame with columns
            ['split', 'embedding', '{family0}_GT_instrument_UID', '{family1}_GT_instrument_UID', ... ]
            where the number of family//instru_UID columns is defined by self.mix_dataset
        """
        # The dataloader will provide more sounds than necessary, we'll keep only the mixes (not the mono-instruments
        #    tracks or single notes)
        ref_sounds_df = self.sounds_df[self.sounds_df['is_reference']]

        batch_size_with_extra_tracks = self.batch_size * self.mix_dataset.n_tracks_per_item
        dataloader = self.mix_dataset.get_dataloader(batch_size_with_extra_tracks, num_workers=self.num_workers)
        # We'll build the result dataframe column by column instead of the usual row by row
        all_embeddings = []
        all_instruments_UIDs_by_family = {family: [] for family in self.mix_dataset.mixed_families}
        for minibatch in dataloader:
            audios, track_names, mix_labels = minibatch
            mix_tracks_mask = (mix_labels < 0)
            assert mix_tracks_mask.sum() == (mix_tracks_mask.shape[0] // self.mix_dataset.n_tracks_per_item)

            embeddings = self.model.compute_audio_embeddings(audios[mix_tracks_mask, :], self.sr, paths=None)
            all_embeddings.append(embeddings.cpu())
            #  Store the instruments UIDs
            for family_idx, family in enumerate(self.mix_dataset.mixed_families):
                family_tracks_mask = mix_tracks_mask.roll(family_idx+1)
                for i in range(len(track_names)):
                    if family_tracks_mask[i]:
                        track_name = track_names[i]
                        instrument_UID = re.sub(r'^Mix_\d+__', '', track_name)
                        # Sanity check: this instrument UID must have a reference sound in this evaluator
                        instrument_index = self._instrument_UID_to_index[instrument_UID]
                        ref_sound = ref_sounds_df[ref_sounds_df['_instrument_index'] == instrument_index]
                        assert len(ref_sound) == 1, f"Exactly one reference should correspond to {instrument_UID=} {instrument_index=} from the mix dataset"
                        assert ref_sound['instrument_UID'].values[0] == instrument_UID, f"{instrument_UID=} from the mix dataset not found in the references"

                        all_instruments_UIDs_by_family[family].append(instrument_UID)

        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()  # np required for checks just after this
        df_data_by_col = {'embedding': [all_embeddings[i, :] for i in range(all_embeddings.shape[0])]}
        for family, instruments_UIDs in all_instruments_UIDs_by_family.items():
            df_data_by_col[f'{family}_GT_instrument_UID'] = instruments_UIDs
        embeddings_df = pd.DataFrame(df_data_by_col)
        embeddings_df['split'] = self.eval_split
        # Also store the dataset index, so we can retrieve the original sample later
        embeddings_df['dataset_index'] = range(len(embeddings_df))  # At this point, actually equal to .index...
        return embeddings_df

    def perform_eval(self, distance_metrics=('l2', 'cosine', 'mahalanobis'), top_k=(1, 5, 10)):
        """
        TODO doc
        """
        device = 'cuda:0' if self.use_cuda else self.model.device
        self.model = self.model.to(device)
        with self._measure_time(f'[TimbreMixSearch-{self.eval_split}] Performing evaluation'):
            # 1.   Compute embeddings for all reference sounds, then for the WHOLE split to be evaluated
            #         and Store those in the main DF for future visualizations, analysis, ...
            with self._measure_time("[TimbreMixSearch] Computing embeddings for all evaluated sounds"):
                # can't use the embeddings computation from DF: sounds are mixed
                # on the fly and not stored on a SSD
                eval_mixes_embeds_df = self.compute_mix_embeddings()
                eval_mixes_embeds_df = self._handle_partial_failure(eval_mixes_embeds_df)
            with self._measure_time("[TimbreMixSearch] Computing embeddings for reference sounds"):
                ref_embeddings_df = self.compute_embeddings_from_df(self.sounds_df[self.sounds_df['is_reference']])
                ref_embeddings_df = self._handle_partial_failure(ref_embeddings_df)
                # Add the family and instrument UIDs to the reference DF (needed for the metrics computation)
                #    w/ a sanity check, ensures DFs remain consistent at this point
                _ref_sounds_sub_df = self.sounds_df.loc[ref_embeddings_df.index]
                assert _ref_sounds_sub_df['instrument_UID'].equals(ref_embeddings_df['instrument_UID'])
                ref_embeddings_df['family'] = _ref_sounds_sub_df[self.mix_dataset.instrument_group_source]

            # 2a. Prepare the columns of eval_mixes_embeddings_df to store results
            nones = [None] * len(eval_mixes_embeds_df)
            for dist_name in distance_metrics:
                for family in self.mix_dataset.mixed_families:
                    eval_mixes_embeds_df[f'{dist_name}__{family}__GT_rank'] = pd.Series(nones, dtype='Int64')
                    for k in top_k:
                        eval_mixes_embeds_df[f'{dist_name}__{family}__top_{k}_UIDs'] = pd.Series(nones, dtype='object')
                        eval_mixes_embeds_df[f'{dist_name}__{family}__GT_in_top_{k}'] = pd.Series(nones, dtype='boolean')

            # 2b.    For each embedding distance measurement type:
            #           evaluate for all sounds; then extract metrics: absolute acc, top-K, average distance with GT, ....
            # We keep the reference sounds in this eval subset, could be useful for sanity checks
            metrics = {}

            ref_embeds = torch.from_numpy(np.vstack(ref_embeddings_df['embedding'].values)).to(device)  # Matrix
            eval_mixes_embeds = torch.from_numpy(np.vstack(eval_mixes_embeds_df['embedding'])).to(device)

            for dist_name in distance_metrics:
                with self._measure_time(f"[TimbreMixSearch] computing and sorting {dist_name} distances"):
                    distance_function = self.available_distances[dist_name]
                    # with self._measure_time(f"[TimbreMixSearch] {dist_name} distance using {device=}"):
                    with torch.no_grad():  # Compute once using CUDA
                        distances_matrix = distance_function(ref_embeds, eval_mixes_embeds).cpu().numpy()

                    # Store results directly in eval_mixes_embeddings_df - Numerous columns required for each family
                    # FIXME something is very slow here... maybe the multiple sorts?
                    for row_i, (df_idx, row) in enumerate(eval_mixes_embeds_df.iterrows()):
                        # Establish correspondences between DF indexes and the pre-computed distances
                        distances_df = pd.Series(distances_matrix[:, row_i], index=ref_embeddings_df.index)
                        distances_df = distances_df.rename('distance').to_frame()
                        cols_to_copy = ['instrument_UID', '_instrument_index', 'family']
                        distances_df[cols_to_copy] = ref_embeddings_df[cols_to_copy]
                        # Don't sort yet; perform the sort on a per-family basis
                        for family in self.mix_dataset.mixed_families:
                            family_distances_df = distances_df.loc[distances_df['family'] == family]
                            family_distances_df = self._sort_distance_with_nans(family_distances_df, seed=row_i)
                            family_distances_df['sort_rank'] = range(len(family_distances_df))
                            GT_instrument_UID = row[f'{family}_GT_instrument_UID']
                            _GT_instrument_index = self._instrument_UID_to_index[GT_instrument_UID]
                            _rank_idx = (family_distances_df['_instrument_index'] == _GT_instrument_index).idxmax()
                            GT_rank = family_distances_df.at[_rank_idx, 'sort_rank']
                            eval_mixes_embeds_df.at[row.name, f'{dist_name}__{family}__GT_rank'] = GT_rank

                            matching_instruments = family_distances_df.iloc[0:max(top_k)]['instrument_UID']
                            for k in top_k:
                                top_k_instruments = matching_instruments[0:k].values.tolist()
                                eval_mixes_embeds_df.at[row.name, f'{dist_name}__{family}__top_{k}_UIDs'] = top_k_instruments
                                is_in_top_k = (GT_instrument_UID in top_k_instruments)
                                eval_mixes_embeds_df.at[row.name, f'{dist_name}__{family}__GT_in_top_{k}'] = is_in_top_k
                        # TODO Also retrieve the global top-10? (for all sounds, not restricted to a family)

                    # Store best matches, and Finally compute average top-K accuracies
                    for k in top_k:
                        cross_family_avg = []
                        for family in self.mix_dataset.mixed_families:
                            avg = eval_mixes_embeds_df[f'{dist_name}__{family}__GT_in_top_{k}'].mean()
                            metrics[f'{dist_name}__{family}__top_{k}_acc'] = avg
                            cross_family_avg.append(avg)
                        metrics[f'{dist_name}__OVERALL__top_{k}_acc'] = np.mean(cross_family_avg)
                    # Also compute the average rank of the GT instrument (i.e. in which top-k the GT can be found)
                    cross_family_avg = []
                    for family in self.mix_dataset.mixed_families:
                        avg = eval_mixes_embeds_df[f'{dist_name}__{family}__GT_rank'].mean()
                        metrics[f'{dist_name}__{family}__GT_rank_avg'] = avg
                        metrics[f'{dist_name}__{family}__GT_rank_med'] \
                            = eval_mixes_embeds_df[f'{dist_name}__{family}__GT_rank'].median()
                        cross_family_avg.append(avg)
                    metrics[f'{dist_name}__OVERALL__GT_rank_avg'] = np.mean(cross_family_avg)

        return eval_mixes_embeds_df, metrics

    @staticmethod
    def select_main_metrics(metrics, main_distance_metric='l2'):
        """ Select the main metrics for logged curves, etc.... Others can be logged as a table maybe? """
        top_k_values = sorted({
            key.split('top_')[1].split('_acc')[0]
            for key in metrics.keys() if ('top_' in key and '_acc' in key)
        }, key=int)
        main_metrics = {
            key.replace('__OVERALL__', '_'): metrics[key]
            for key in [f'{main_distance_metric}__OVERALL__top_{k}_acc' for k in top_k_values]
        }
        # GT rank is already on a plot - and 3 values / split in W&B is better (e.g. only top-1, top-5, top-10)
        #main_metrics[f'{main_distance_metric}_GT_rank'] = metrics[f'{main_distance_metric}__OVERALL__GT_rank_avg']
        return main_metrics

    def _sort_distance_with_nans(self, distances_df: pd.DataFrame, seed: Optional[int] = None):
        if not np.all(distances_df.isna()):
            distances_df = distances_df.sort_values(by='distance')
        # Else if ONLY NaNs: no sort at all... so we shuffle (not to return always the same best matches...)
        #    TODO We should have a 'None' InstrumentUID here!
        else:
            rng = np.random.default_rng(seed=seed)
            distances_index = distances_df.index.tolist()
            rng.shuffle(distances_index)
            distances_df = distances_df[distances_index]
        return distances_df

    def plot_eval_sounds_df(self, eval_sounds_df: pd.DataFrame, distance_metric='l2'):
        """ Plots detailed results for a single distance metric, returns the dict of mpl figures. """
        df = eval_sounds_df.copy()  # Will be modified (new cols, ...)
        m = distance_metric  # Shorter notation
        figs = {}  # to store the figures

        # 1. Accuracies by family for top-k
        available_top_ks = []
        for k in range(100):  # Max k=100 for top-ks
            # Check if this top-k is available in the dataframe
            if any(f'{m}__{family}__GT_in_top_{k}' in df.columns for family in self.mix_dataset.mixed_families):
                available_top_ks.append(k)
        if len(available_top_ks) == 0:
            print(f"Warning: No top-k accuracy columns found for distance metric {m}")
            return figs

        # Create a figure for accuracies by family - try to keep an almost-square fig (better for logging into W&B)
        data_sources = sorted(self.mix_dataset.notes_df['data_source'].unique())
        fig, axes = plt.subplots(len(available_top_ks), len(data_sources), figsize=(8, 8))  #  , sharex='row') -> does not work
        for i, k in enumerate(available_top_ks):
            for j, data_source in enumerate(data_sources):
                accuracies = {}
                for family in self.mix_dataset.mixed_families:  # Different (random) use of data sources for each family
                    data_source_mask = df[f'{family}_GT_instrument_UID'].apply(lambda x: x.startswith(data_source))
                    if data_source_mask.sum() > 0:  # Some families are missing from some sub-datasets
                        accuracies[family] = df.loc[data_source_mask, f'{m}__{family}__GT_in_top_{k}'].mean()
                    else:
                        accuracies[family] = -0.001
                # add percentages on top of bars (font size 8)
                bars = pd.Series(accuracies).plot(kind='bar', ax=axes[i, j], color=f'C{j}')
                for bar in bars.patches:
                    x, y = bar.get_x() + bar.get_width() / 2, bar.get_height()
                    if y >= 0.0:
                        axes[i, j].text(x, y, f'{y:.1%}', ha='center', va='bottom', fontsize=8)

                axes[i, j].set(ylabel=f'Top-{k} accuracy', ylim=[0.0, 1.0], title=f'Top-{k} acc - {data_source}')
                if i == len(available_top_ks) - 1:
                    axes[i, j].tick_params(axis='x', rotation=45)
                else:
                    axes[i, j].set_xticklabels([])
        fig.suptitle(f'Top-K accuracies, by instrument family and sub-dataset')
        fig.tight_layout()
        figs['accuracies_by_family'] = fig

        # 2. boxplots of ranks by family
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        # Build a long-form dataframe, easier to plot
        ranks_data = []
        for family in self.mix_dataset.mixed_families:
            ranks_sub_df = df[f'{m}__{family}__GT_rank'].rename('GT_rank').to_frame()
            ranks_sub_df['family'] = family
            ranks_sub_df['data_source'] = df[f'{family}_GT_instrument_UID'].apply(lambda x: x.split('__')[0])
            ranks_data.append(ranks_sub_df)
        ranks_long_df = pd.concat(ranks_data).reset_index(drop=True)
        sns.boxplot(
            data=ranks_long_df, x='family', y='GT_rank',
            hue='data_source', hue_order=data_sources, showfliers=False, ax=ax
        )
        ax.set(ylabel='Unmixing output rank', title='Output rank of ground truth instruments, by family and sub-dataset')
        ax.tick_params(axis='x', rotation=45)
        # Lower rank is better, so invert y-axis
        ax.invert_yaxis()

        fig.tight_layout()
        figs['avg_rank_by_family'] = fig

        return figs

    def get_audios_for_mix(self, eval_sounds_df: pd.DataFrame, dataset_index: int, distance_metric='l2'):
        """
         A method to retrieve some audio examples?
           i.e. return the mix, (the GT mono-instrument tracks... but requires another dataset instance),
                the GT single notes (and their rank), the retrieved instrument's single note
            TODO w/ instrument's NAMES (not UIDs...) each time (requires a unified column in the datasets' dataframes,
                currently there's a different column for each data_source: nsynth, surge, ...)
        """
        mix_results = eval_sounds_df.loc[(eval_sounds_df['dataset_index'] == dataset_index).idxmax()]

        dataset_audios, dataset_track_names = self.mix_dataset[dataset_index]
        assert dataset_track_names[-1].lower() == 'mix'
        audios = {'mix': {'audio': dataset_audios[-1, :], 'info': f'mix ({len(self.mix_dataset.mixed_families)} instruments)'}, }
        # Sanity check
        for i, family in enumerate(self.mix_dataset.mixed_families):
            assert mix_results[f'{family}_GT_instrument_UID'] == dataset_track_names[i]

        assert self.eval_split in self.reference_datasets
        ref_sounds_df = self.sounds_df[self.sounds_df['is_reference']]
        audios['GT_single_notes'], audios['matched_single_notes'] = {}, {}
        for family in self.mix_dataset.mixed_families:
            matched_note = ref_sounds_df.loc[ref_sounds_df['instrument_UID'] ==
                                             mix_results[f'{distance_metric}__{family}__top_1_UIDs'][0]]
            matched_note = matched_note.squeeze()
            audios['matched_single_notes'][family] = {
                'audio': self.datasets[matched_note['split']].get_audio(matched_note['path']),
                'info': f'{matched_note["instrument_UID"]}'
            }

            # Retrieve a GT note with the same pitch as the matched note (if available) - otherwise use the default ref
            gt_ref_note = ref_sounds_df.loc[ref_sounds_df['instrument_UID'] == mix_results[f'{family}_GT_instrument_UID']].squeeze()
            assert gt_ref_note['split'] == self.eval_split
            gt_instrument_notes = self.sounds_df[self.sounds_df['_instrument_index'] == gt_ref_note['_instrument_index']]
            gt_notes_matched_pitch = gt_instrument_notes[gt_instrument_notes['midi_pitch'] == matched_note['midi_pitch']].copy()
            if len(gt_notes_matched_pitch) > 0:
                gt_notes_matched_pitch['vel_distance'] = np.abs(
                    gt_notes_matched_pitch['midi_velocity'] - matched_note['midi_velocity'])
                idx_min = gt_notes_matched_pitch['vel_distance'].idxmin()
                gt_audio = self.datasets[self.eval_split].get_audio(gt_notes_matched_pitch.at[idx_min, 'path'])
            else:  # No corresponding pitch... just load the default reference audio for that instrument
                gt_audio = self.datasets[self.eval_split].get_audio(gt_ref_note['path'])
            audios['GT_single_notes'][family] = {
                'audio': gt_audio, 'info': f'{gt_ref_note["instrument_UID"]}'
            }

        return audios


def main():
    parser = argparse.ArgumentParser(description='TimbreMixSearch - Search for timbres/presets in a mix')
    parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint', required=True)
    parser.add_argument('--eval-split', type=str, default='test', help='Dataset split to evaluate on')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--n-workers', type=int, default=0, help='Number of worker processes for data loading')
    parser.add_argument('--force-use-cache', action='store_true', help='Only generate logs and plots from cached results')
    parser.add_argument('--distance-metric', type=str, default='cosine', help='Distance metric to use (l2, cosine, mahalanobis, ...)')
    parser.add_argument('--wandb-log', action='store_true', help='Log results to W&B')
    parser.add_argument('--wandb-project', type=str, default=None, help='Override W&B project for logging/resuming (useful if the run was moved)')

    args = parser.parse_args()

    # Import here to avoid circular imports
    import AST

    print(f"Loading model from {args.checkpoint}...")
    # FIXME not always AST
    model = AST.ASTFineTuned.from_checkpoint(args.checkpoint)
    config = model.custom_config

    # Prepare W&B logging
    if args.wandb_log:
        # Retrieve a logger for this W&B run id (other details are in secrets.yaml)
        #    in order to log new values into that same run
        with open("secrets.yaml", "r") as f:
            secrets = yaml.safe_load(f)
        wandb.login(key=secrets['wandb']['api_key'])
        project_name = args.wandb_project or config['run']['project']
        wandb_run = wandb.init(
            entity=secrets['wandb']['team'],
            project=project_name,
            # name=config['run'].get('name', None),
            id=config['run']['wandb_id'],  # is it enough? Are the project and name really necessary?
            resume='must',
        )
    else:
        wandb_run = None

    # Use the exact same args as in the training script
    print(f"Computing timbre mix search results for {args.eval_split=}, {args.distance_metric=}")
    reference_datasets = ['train', 'valid'] + (['test'] if args.eval_split == 'test' else [])
    timbre_mix_evaluator = TimbreMixSearchEvaluator(
        model=model,
        num_workers=args.n_workers,
        batch_size=config['training']['batch_size'],
        eval_split=args.eval_split,
        reference_datasets=reference_datasets,
        verbose=True,
        use_cuda=args.cuda,
        use_cache=False,
    )

    # Create cache directory if it doesn't exist
    os.makedirs("cache", exist_ok=True)
    cache_file = "cache/TEMP_timbremixsearch.pkl"
    if not args.force_use_cache:
        eval_sounds_df, metrics = timbre_mix_evaluator.perform_eval(
            distance_metrics=(args.distance_metric, ),
        )
        with open(cache_file, 'wb') as f:
            pickle.dump((eval_sounds_df, metrics), f)
        print(f"Results saved successfully to {cache_file=}.")
    else:
        try:
            with open(cache_file, 'rb') as f:
                eval_sounds_df, metrics = pickle.load(f)
            print(f"Saved results loaded successfully from {cache_file=}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Cache file {cache_file} not found. Run without --force-use-cache first.")

    main_metrics = timbre_mix_evaluator.select_main_metrics(metrics, main_distance_metric=args.distance_metric)
    for k, m in main_metrics.items():
        print((f"{k}: {m:.1%}" if '_acc' in k else f"{k}: {m}"))
    # Save metrics for W&B
    log_data = {f'timbremix/{k}/{args.eval_split}': v for k, v in main_metrics.items()}

    # Always do plots
    # TODO Upload to W&B?? and allow to store locally (high-def PDFs)
    figs = timbre_mix_evaluator.plot_eval_sounds_df(
        eval_sounds_df, distance_metric=args.distance_metric,
    )
    plt.show()

    # Try retrieve demonstration audios
    # TODO Allow to store the results (local SSD), and allow to choose the dataset index from an arg....
    audios = timbre_mix_evaluator.get_audios_for_mix(
        eval_sounds_df, 1, distance_metric=args.distance_metric,
    )

    # Should automatically use the next 'step' number
    if wandb_run is not None:
        wandb_run.log(data=log_data)


if __name__ == '__main__':
    main()
