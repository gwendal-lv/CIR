from pathlib import Path
import os
import pickle
from typing import Union, Optional, Sequence

import numpy as np
import pandas
import pandas as pd
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchaudio

from tqdm import tqdm

import maths
import mergeddataset
import utils


# TODO Implement a "dataset_sources" arg to keep only some Sub-dataset (e.g. not NSynth but keep Surge and Arturia)
# TODO Implement no-padding arg (will disable batching)
class TimbreSearchBase:
    def __init__(self, model, sr:Optional[int] = None, batch_size=32,
                 eval_split='valid', reference_datasets=('train', 'valid'),
                 use_small_dataset: Union[bool, dict[str, bool]]=False,
                 exclude_augmentations=True, reference_note=None,
                 verbose=False, use_cuda=False, use_cache=False):
        """
        Base class for timbre search functionality.

        :param model:  Must have .compute_audio_embeddings(...) and .to(device) methods, and a .sr integer field
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
        :param use_cuda: If True, the model will be moved to cuda:0 during evaluation
        """
        self.verbose = verbose
        self.use_cuda, self.use_cache = use_cuda, use_cache
        # TQDM is too annoying when it's an inner loop - disabled for now
        # self._tqdm_call = ((lambda x, y: tqdm(x, desc=y, mininterval=0.5)) if self.verbose else (lambda x, y: x))
        self._measure_time = utils.measure_time if verbose else utils.dummy_measure_time

        # Check that the model has the proper functionalities to allow for timbre search
        assert hasattr(model, 'sr')
        for method_name in ['compute_audio_embeddings', 'to']:
            assert hasattr(model, method_name) and callable(getattr(model, method_name))
        self.model = model
        try:
            model_sr = self.model.sr
        except AttributeError:
            model_sr = None
        if sr is not None:
            self.sr = sr
            if model_sr is not None:
                assert self.sr == model_sr, f"{self.sr}, {model_sr}"
        else:  # No sr given as arg
            assert model_sr is not None, f"self.model.sr could not be found but sr has not been given as arg"
            self.sr = model_sr
        self.batch_size = batch_size

        assert eval_split in ['train', 'valid', 'test'] and eval_split in reference_datasets
        self.eval_split = eval_split
        assert all([split in ['train', 'valid', 'test'] for split in reference_datasets])
        # Use small dataset: can use different values for each sub-dataset
        if isinstance(use_small_dataset, bool):  # If already a dict, just use it as it is
            use_small_dataset = {k: use_small_dataset for k in ['train', 'valid', 'test']}
        self.use_small_dataset = use_small_dataset
        self.reference_datasets = reference_datasets

        #  If we only use reference notes from that unique dataset, it makes the task easier...
        #     having many anchors from all datasets seems more appropriate
        with self._measure_time("[TimbreSearch] Loading the datasets"):
            self.datasets = {}
            for split in self.reference_datasets:
                # We use the most generic type possible! Don't restrict to labeled samples
                # FIXME maybe DON'T use augmented instrument for the training set...?
                self.datasets[split] = mergeddataset.MultimodalSoundsDataset(
                    use_small_dataset=self.use_small_dataset[split], split=split, pad='right', target_sr=self.sr,
                )
            # This DF will represent our "database" of sounds - don't forget that concat would allow duplicates indices
            self.sounds_df = pd.concat([ds.df for k, ds in self.datasets.items()], ignore_index=True)
            # indices used for faster search - for internal class use only
            self._instrument_index_to_UID = list(self.sounds_df['instrument_UID'].unique())
            self._instrument_UID_to_index = {uid: i for i, uid in enumerate(self._instrument_index_to_UID)}
            self.sounds_df['_instrument_index'] = self.sounds_df['instrument_UID'].apply(lambda x: self._instrument_UID_to_index[x])

            # If required, only use non-augmented presets
            if exclude_augmentations:
                self.sounds_df = self.sounds_df[~self.sounds_df['is_augmented']]
            # Identify some reference notes in the datasetS
            self.sounds_df.insert(3, 'is_reference', False)
            ref_instruments = list(sorted(set(self.sounds_df.instrument_UID)))

        self._select_reference_sounds(ref_instruments, reference_note)
        print(f"[TimbreSearch]    - - - - - {self.sounds_df['is_reference'].sum()} reference sounds - - - - -")

        # Column ready to be filled
        self.sounds_df.insert(3, 'embedding', None)

    def _select_reference_sounds(self, ref_instruments, reference_note):
        """
        Selects reference sounds for each instrument in the dataset.
        Will try to load from cache if possible, otherwise triggers a (rather long) computation.

        For each instrument, tries to find a reference sound with a specific pitch and velocity.
        If an exact match isn't found, selects the closest available note.

        Args:
            ref_instruments: List of instrument UIDs
            reference_note: Tuple of (pitch, velocity) or None

        """
        if not self.use_cache:
            self._recompute_reference_sounds(ref_instruments, reference_note)  # updates self.sounds_df
            return

        # Implement a cache for this computation
        cache_dir = Path("cache")
        cache_name = f"TimbreSearchBase-refsounds_{'-'.join(sorted(self.reference_datasets))}"
        if reference_note is not None:
            cache_name += f"_refnote-{reference_note[0]}-{reference_note[1]}"
        cache_file = cache_dir / f"{cache_name}.pkl"
        if not cache_dir.exists():
            os.makedirs(cache_dir, exist_ok=True)

        # Try to load from cache if it exists
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_sounds_df = pickle.load(f)
                # Check if the paths in the cached dataframe match exactly with the current dataframe
                if (cached_sounds_df['_instrument_index'].equals(self.sounds_df['_instrument_index'])
                        and cached_sounds_df['midi_pitch'].equals(self.sounds_df['midi_pitch'])):
                    print(f"[TimbreSearch] Using cached reference sounds from '{cache_file}'")
                    self.sounds_df = cached_sounds_df
                    return
                else:
                    print("[TimbreSearch] Cache exists but paths don't match, recomputing reference sounds")
            except Exception as e:
                print(f"[TimbreSearch] Error loading cache: {e}, recomputing reference sounds")

        self._recompute_reference_sounds(ref_instruments, reference_note)  # updates self.sounds_df

        # Save to cache for future use
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.sounds_df, f)
            print(f"[TimbreSearch] Saved reference sounds to cache '{cache_file}'")
        except Exception as e:
            print(f"[TimbreSearch] Error saving cache: {e}")

    def _recompute_reference_sounds(self, ref_instruments, reference_note):
        """ Actual computation (if could not be loaded from cache).
         If ref note is not available for a given instrument, then try other velocities and nearby pitches
         only search in audios that correspond to a single MIDI note (will have non-NaN pitch and vel cols)
         """
        # Some UIDs can be already excluded from this list of UIDs, e.g. data augmentations
        instrument_UIDs_backup = list(self.sounds_df['instrument_UID'].unique())

        notes_df = self.sounds_df[self.sounds_df['midi_pitch'].notna()]

        for instrument_UID in ref_instruments:
            instrument_index = self._instrument_UID_to_index[instrument_UID]
            available_audios_df = notes_df[notes_df['_instrument_index'] == instrument_index].copy()
            # Compute a different ref pitch for each instrument (Depending on what's available in our dataset)
            if reference_note is None:
                ref_pitch = int(available_audios_df['midi_pitch'].median())
                ref_vel = 100
            else:
                ref_pitch, ref_vel = reference_note
            # First (to speed up...) try to get an exact match if exists
            exact_ref_match = available_audios_df[
                (available_audios_df['midi_pitch'] == ref_pitch) & (available_audios_df['midi_velocity'] == ref_vel)
                ]
            assert len(exact_ref_match) <= 1
            if len(exact_ref_match) == 1:
                ref_index = exact_ref_match.iloc[0].name  # Dataframe index
            else:
                # Otherwise, find the best available note to be used as reference
                available_audios_df['pitch_distance'] = np.abs(available_audios_df['midi_pitch'] - ref_pitch)
                available_audios_df['velocity_distance'] = np.abs(available_audios_df['midi_velocity'] - ref_vel)
                available_audios_df = available_audios_df.sort_values(by=['pitch_distance', 'velocity_distance'])
                assert len(available_audios_df) > 0  # No Reference file available
                ref_index = available_audios_df.iloc[0].name  # Dataframe index
            self.sounds_df.loc[ref_index, 'is_reference'] = True

        # Sanity check: number of reference notes and number of instruments
        # FIXME Check that all UIDs from the datasetS are in the reference sounds list
        ref_sounds_df = self.sounds_df[self.sounds_df['is_reference']]
        ref_instruments_UIDs = ref_sounds_df['instrument_UID'].unique()
        for instrument_UID, instrument_index in self._instrument_UID_to_index.items():
            if instrument_UID in instrument_UIDs_backup:
                assert instrument_UID in ref_instruments_UIDs, f"{instrument_UID=} not in the selected reference sounds"
        assert len(ref_instruments_UIDs) == len(instrument_UIDs_backup)

    def compute_embeddings_from_df(self, df: pandas.DataFrame):
        """ Computes the embeddings for a Dataframe of sounds (will use the 'path' column) """
        n_batches = len(df) // self.batch_size + int((len(df) % self.batch_size) != 0)
        indices_batches = np.array_split(df.index, n_batches)
        df_batches = [df.loc[indices_batches[i], :] for i in range(n_batches)]

        res_df = df.loc[:, ['split', 'instrument_UID', '_instrument_index', 'path']].copy()
        res_df['embedding'] = None
        for df_batch in df_batches:
            audios = list()
            paths = list()
            for p in df_batch['path']:
                x, sr = torchaudio.load(p)
                if sr != self.sr:
                    x = torchaudio.transforms.Resample(sr, self.sr)(x)
                audios.append(x)
                paths.append(Path(p))
            embeddings = self.model.compute_audio_embeddings(audios, self.sr, paths=paths)
            # Split the embeddings now for storing them in the dataframe... Probably not optimal, but easier to
            #       handle, and harder to make mistakes later
            embeddings = [embeddings[i, :].cpu().numpy() for i in range(embeddings.shape[0])]
            for df_idx, embedding in zip(df_batch.index, embeddings):
                res_df.at[df_idx, 'embedding'] = embedding
        return res_df

    @staticmethod
    def _handle_partial_failure(df: pd.DataFrame):
        """ Processes the 'embeddings' column of a dataframe of results. Handle NaNs obtained with some baselines,
            e.g. TTB. For full failures, we keep NaNs. For partial failures (e.g. 1 feature coord is missing...)
            we use 0.0 """
        #full_failure_mask = df.embedding.apply(lambda x: np.any(np.isnan(x)))
        #partial_failure_mask = df.embedding.apply(lambda x: np.any(np.isnan(x)))
        df['embedding'] = df['embedding'].apply(
            lambda x: (np.nan_to_num(x, nan=0.0, copy=False) if not np.all(np.isnan(x)) else x)
        )
        return df

    @property
    def available_distances(self):
        return {'l2': maths.distance_l2, 'cosine': maths.distance_cosine, 'mahalanobis': maths.distance_mahalanobis}


class TimbreSearchEvaluator(TimbreSearchBase):
    def __init__(self, model, sr:Optional[int] = None, batch_size=32,
                 eval_split='valid', reference_datasets=('train', 'valid'),
                 use_small_dataset: Union[bool, dict[str, bool]]=False,
                 exclude_augmentations=True, reference_note=None,
                 verbose=False, use_cuda=False,
                 ):
        """
        :param model:  Must have .compute_audio_embeddings(...) and .to(device) methods, and a .sr integer field
        :param eval_split: The sub-dataset to be evaluated
        :param reference_datasets: What splits should be used to compute reference embeddings
        :param use_small_dataset:
        :param exclude_augmentations: Don't use data-augmented presets as references (anchors) for search
        :param reference_note: If given as (midi_pitch, midi_velocity), the same reference note (or a similar one)
                                will be used for all instruments. If None, the median pitch for each instrument,
                                and a default velocity value 100 will be used.
        :param use_cuda: If True, the model will be moved to cuda:0 during evaluation
        """
        # Call the base class constructor
        super().__init__(model=model, sr=sr, batch_size=batch_size,
                         eval_split=eval_split, reference_datasets=reference_datasets,
                         use_small_dataset=use_small_dataset, exclude_augmentations=exclude_augmentations,
                         reference_note=reference_note, verbose=verbose, use_cuda=use_cuda)

        print(f"[TimbreSearch]    - - - - - {len(self.datasets[self.eval_split])} sounds in the eval dataset ({self.eval_split=}) - - - - -")

    @property
    def embeddings_count(self):
        """ The number of embeddings that must be computed to perform the timbre search
            (reference sounds + eval sounds) """
        return len(self.sounds_df[self.sounds_df['is_reference'] | (self.sounds_df['split'] == self.eval_split)])

    def perform_eval(self, distance_metrics=('l2', 'cosine', 'mahalanobis')):
        """
        Evaluates the model's performance by computing embeddings for reference sounds and evaluation sounds,
        then calculating distances between them using specified metrics to find the best matches.

        The method computes embeddings for all reference sounds and all sounds in the evaluation split,
        then for each distance metric, it calculates distances between reference embeddings and evaluation
        embeddings. It identifies the top-k matches for each evaluation sound and computes accuracy metrics.

        :param distance_metrics: Tuple of distance metrics to use for evaluation. Options include 'l2', 'cosine', and 'mahalanobis'
        :return: A tuple containing:
                 - eval_sounds_df: DataFrame with evaluation sounds and their best matches for each distance metric
                 - metrics: Dictionary with top-k accuracy metrics for each distance metric
        """
        device = 'cuda:0' if self.use_cuda else self.model.device
        self.model = self.model.to(device)
        with self._measure_time(f'[TimbreSearch-{self.eval_split}] Performing evaluation'):
            # 1.   Compute embeddings for all reference sounds, then for the WHOLE split to be evaluated
            #         and Store those in the main DF for future visualizations, analysis, ...
            with self._measure_time("[TimbreSearch] Computing embeddings for reference sounds"):
                ref_embeddings_df = self.compute_embeddings_from_df(self.sounds_df[self.sounds_df['is_reference']])
                ref_embeddings_df = self._handle_partial_failure(ref_embeddings_df)
                self.sounds_df.loc[ref_embeddings_df.index, 'embedding'] = ref_embeddings_df['embedding']
            with self._measure_time("[TimbreSearch] Computing embeddings for all evaluated sounds"):
                # Compute only for non-ref ; don't compute things twice...
                non_ref_eval_sounds_mask = ((self.sounds_df['split'] == self.eval_split) & (~self.sounds_df['is_reference']))
                eval_embeddings_df = self.compute_embeddings_from_df(self.sounds_df[non_ref_eval_sounds_mask])
                eval_embeddings_df = self._handle_partial_failure(eval_embeddings_df)
                self.sounds_df.loc[eval_embeddings_df.index, 'embedding'] = eval_embeddings_df['embedding']

            # 2.    For each embedding distance measurement type:
            #           evaluate for all sounds; then extract metrics: absolute acc, top-K, average distance with GT, ....
            # We keep the reference sounds in this eval subset, could be useful for sanity checks
            eval_sounds_df = self.sounds_df[self.sounds_df['split'] == self.eval_split].copy()  # Including reference sounds
            metrics = {}

            ref_embeds = torch.from_numpy(np.vstack(ref_embeddings_df['embedding'].values)).to(device)  # Matrix
            eval_sounds_embeds = torch.from_numpy(np.vstack(eval_sounds_df['embedding'])).to(device)

            for distance_name in distance_metrics:
                with self._measure_time(f"[TimbreSearch] {distance_name} distance computations"):
                    distance_function = self.available_distances[distance_name]
                    with self._measure_time(f"[TimbreSearch] {distance_name} distance using {device=}"):
                        with torch.no_grad():  # Compute once using CUDA
                            # Distances between NaNs are Nan - and they behave nicely with the sorting later on
                            distances_matrix = distance_function(ref_embeds, eval_sounds_embeds).cpu().numpy()

                    n_correct = {1: 0, 3: 0, 5: 0}  # n_correct in the Top 1, top 3 and top 5
                    best_match, best_matches = [], []  # For each row ; will be eventually a new column of the dataframe
                    for row_i, (df_idx, row) in enumerate(eval_sounds_df.iterrows()):
                        # Establish correspondences between DF indexes and the pre-computed distances
                        distances = pd.Series(distances_matrix[:, row_i], index=ref_embeddings_df.index)
                        distances = distances.rename('distance')
                        if not np.all(distances.isna()):
                            distances = distances.sort_values()
                        # Else if ONLY NaNs: no sort at all... so we shuffle (not to return always the same best matches...)
                        #    TODO We should have a 'None' InstrumentUID here!
                        else:
                            rng = np.random.default_rng(seed=row_i)
                            distances_index = distances.index.tolist()
                            rng.shuffle(distances_index)
                            distances = distances[distances_index]
                        GT_instrument_UID = row['instrument_UID']
                        for k in n_correct.keys():
                            matching_instruments = ref_embeddings_df.loc[distances.iloc[0:k].index]['instrument_UID'].values
                            if GT_instrument_UID in matching_instruments:
                                n_correct[k] += 1
                        matching_instruments = ref_embeddings_df.loc[distances.iloc[0:5].index]['instrument_UID'].values
                        best_match.append(matching_instruments[0])
                        best_matches.append(','.join(matching_instruments.tolist()))
                    # Store best matches, and Finally compute average top-K accuracies
                    eval_sounds_df[f'{distance_name}_best_match'] = best_match
                    eval_sounds_df[f'{distance_name}_top_5'] = best_matches
                    for k, n in n_correct.items():
                        metrics[f'{distance_name}_top_{k}_acc'] = n / len(eval_sounds_df)
        return eval_sounds_df, metrics

    def plot_eval_sounds_df(self, eval_sounds_df, distance_metric='l2'):
        """ Plots detailed results for a single distance metric, returns the dict of mpl figures. """
        df = eval_sounds_df.copy()  # Will be modified (new cols, ...)
        m = distance_metric  # Shorter notation
        df['in_top_1'] = (df[f'{m}_best_match'] == df['instrument_UID'])
        df['in_top_5'] = df.apply((lambda row: row['instrument_UID'] in (row[f'{m}_top_5'].split(','))), axis=1)
        df['instrument_family'] = df['instrument_family'].fillna('None')
        families_order = sorted(df['instrument_family'].unique())
        data_sources_order = list(sorted(df['data_source'].unique()))

        figs = {}
        # 1. Detailed accuracies (for all available top-Ks), by instrument category (including "other")
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        plot_kwargs = {'order': families_order, 'hue_order': (True, False), 'palette': 'Set2'}
        sns.countplot(data=df, x='instrument_family', hue='in_top_1', ax=axes[0, 0], **plot_kwargs)
        sns.countplot(data=df, x='instrument_family', hue='in_top_5', ax=axes[0, 1], **plot_kwargs)
        # Compute accuracies by hand, then plot
        for j, top_K in enumerate([1, 5]):
            accuracies = {}
            for family in families_order:
                sub_df = df[df['instrument_family'] == family]
                accuracies[family] = sub_df[f'in_top_{top_K}'].sum() / len(sub_df)
            accuracies = pd.Series(accuracies)
            accuracies.plot(kind='bar', ax=axes[1, j])
            axes[1, j].set(ylabel=f'Top-{top_K} accuracy', ylim=[0.0, 1.0])

        for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            axes[i, j].tick_params(axis='x', rotation=90)
        fig.tight_layout()
        figs['accuracies_by_family'] = fig

        # 1bis. accuracies by data source (surge, nsynth, etc...)
        fig, axes = plt.subplots(1, 2, figsize=(8, 6))
        plot_kwargs = {'hue_order': (True, False), 'palette': 'Set2'}
        sns.countplot(data=df, x='data_source', hue='in_top_1', ax=axes[0], **plot_kwargs)
        sns.countplot(data=df, x='data_source', hue='in_top_5', ax=axes[1], **plot_kwargs)
        fig.tight_layout()
        figs['accuracies_by_dataset'] = fig

        # 1ter. Detailed accuracies for single MIDI notes vs. more complex - also split by MIDI pitch proximity
        #        (the closest half vs the furthest half), and maybe the same for velocity
        #      And data sources in the same plot
        df['is_pitch_close_to_ref'] = False  # Default value for all (including sounds played from MIDI track)
        df['has_pitch'] = False
        for instrument_UID in set(df['instrument_UID'].unique()):
            # Ref pitch: NSynth shares instruments across test and valid sets - so the reference note may be
            #  missing from either the valid or test df. Then: use the full DF to retrieve the unique ref.
            ref_sounds = self.sounds_df[self.sounds_df['is_reference']]  # First reduce the DF for UID string search
            ref_sounds = ref_sounds[ref_sounds['instrument_UID'] == instrument_UID]
            assert len(ref_sounds) == 1, f"Should have only 1 reference sound for {instrument_UID=} ({len(ref_sounds)=})"
            ref_pitch = ref_sounds['midi_pitch'].values[0]
            # And extract the sub-dataframe to plot
            sub_df = df[df['instrument_UID'] == instrument_UID]
            sub_df = sub_df[~sub_df['midi_pitch'].isna()]
            df.loc[sub_df.index, 'has_pitch'] = True
            pitch_distance_to_ref = np.abs(sub_df['midi_pitch'] - ref_pitch)
            median_pitch_distance = pitch_distance_to_ref.median()
            sub_df = sub_df[pitch_distance_to_ref <= median_pitch_distance]  # Only keep the close ones
            df.loc[sub_df.index, 'is_pitch_close_to_ref'] = True
        # Build another 'all' DF that corresponds to all data sources (easier to plot...)
        extended_df = df.copy()
        extended_df['data_source'] = 'all'
        extended_df = pd.concat([df, extended_df], axis=0)
        extended_data_sources = ['all'] + data_sources_order
        fig, axes = plt.subplots(2, 3, figsize=(8, 8), sharey='row')
        plot_kwargs = {'order': extended_data_sources, 'hue_order': (True, False), 'palette': 'Set2'}
        for i, top_K in enumerate([1, 5]):
            for j, pitch_condition in enumerate(['close', 'far', 'no_pitch']):
                ax = axes[i, j]
                if pitch_condition == 'close':
                    _df = extended_df[extended_df['has_pitch'] & extended_df['is_pitch_close_to_ref']]
                    ax.set(title=f'Top-{top_K}\npitch CLOSE to ref pitch')
                elif pitch_condition == 'far':
                    _df = extended_df[extended_df['has_pitch'] & (~extended_df['is_pitch_close_to_ref'])]
                    ax.set(title=f'Top-{top_K}\npitch FAR from ref pitch')
                else:
                    _df = extended_df[~extended_df['has_pitch']]
                    ax.set(title=f'Top-{top_K}\nno unique pitch')
                # countplot w/ percentages on top of the bars
                sns.countplot(data=_df, x='data_source', hue=f'in_top_{top_K}', ax=ax, **plot_kwargs)
                max_y = ax.get_ylim()[1]
                for p in ax.patches:
                    height = p.get_height()
                    percentage = 100.0 * height / np.sum(_df['data_source'] == 'all')
                    if height > 0:  # Only annotate if there's a bar
                        big_height = (height > (0.75 * max_y))
                        y_offset = (0.02 * max_y)
                        ax.text(
                            p.get_x() + p.get_width() / 2.0, y_offset if big_height else height + y_offset,
                            f'{percentage:.2f}%', ha="center", va="bottom", color='black', fontsize=9, rotation=90
                        )
        fig.tight_layout()
        figs['accuracies_by_MIDI'] = fig

        # 2. Confusion matrices - between instrument categories
        #       (if that makes sense??? we can expect VERY diagonal results, a wrong retrieval should be the same cat.)
        # find the family of the best match
        best_match_metric, ref_sounds = 'cosine', self.sounds_df[self.sounds_df['is_reference']]
        def find_best_match_family(row: pd.Series):
            best_match = ref_sounds[ref_sounds['instrument_UID'] == row[f'{best_match_metric}_best_match']]
            return best_match.iloc[0]['instrument_family']
        df['best_match_instrument_family'] = df.apply(find_best_match_family, axis=1)
        df.loc[df['best_match_instrument_family'].isna(), 'best_match_instrument_family'] = 'None'
        fig, ax = plt.subplots(1, 1, figsize=(8, 7.5))
        conf_matrix = sklearn.metrics.confusion_matrix(df['instrument_family'], df['best_match_instrument_family'])
        # DataFrame for the confusion matrix for better visualization
        conf_matrix_df = pd.DataFrame(conf_matrix, index=families_order, columns=families_order)
        sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='viridis', ax=ax)
        ax.set(title='Confusion matrix for instrument families',
               ylabel='GT family', xlabel=f'Family of the best match ({best_match_metric} distance)')
        fig.tight_layout()
        figs['confusion_matrix'] = fig

        return figs


if __name__ == '__main__':
    # Debugging / development code...

    import pickle
    import tempfile

    import AST

    _recompute = True  # If False, does not recompute but only reloads pickled results

    if _recompute:  # Recompute the debug data
        _model = AST.ASTFineTuned.from_checkpoint("checkpoints/AST-multi-label/2025-05-29__2000.pt")
        _evaluator = TimbreSearchEvaluator(
            _model, use_small_dataset=True, eval_split='valid',
            reference_datasets=('train', 'valid', ),
            verbose=True, use_cuda=True,
        )
        _eval_sounds_df, _metrics = _evaluator.perform_eval()

        print(_metrics)
        _temp_df_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
        with open(_temp_df_file.name, 'wb') as f:
            pickle.dump((_eval_sounds_df, _metrics), f)
        print("Fichier temporaire pour les résultats:", _temp_df_file.name)

    # Reload saved files, to debug/dev plots, etc...
    if not _recompute:
        with open("/tmp/tmpnu3lrweo.pkl", 'rb') as f:
            _eval_sounds_df, _metrics = pickle.load(f)
        print(_metrics)
        # Also do plots now... They will be used for logging into WandB
        _figs = TimbreSearchEvaluator.plot_eval_sounds_df(_eval_sounds_df)
        plt.show()

    pass
