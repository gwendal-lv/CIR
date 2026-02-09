from typing import Optional
import argparse
import pickle

import torch
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

import mixdataset
from timbresearch import TimbreSearchBase
from multiencoder import MultiEncoderModel


class TimbreMixSearchEvaluator(TimbreSearchBase):
    def __init__(self, multi_encoder_model: MultiEncoderModel,
                 sr: Optional[int] = None, batch_size=32, use_cuda=True,
                 eval_split='valid', reference_datasets=('train', 'valid'),
                 exclude_augmentations=True, reference_note=None,
                 verbose=False, num_workers=0, use_cache=False):
        """
        Class for instrument retrieval in mixtures using a MultiEncoderModel, which provide a fixed amount of
        output embeddings which are supposed to correspond to single-instrument embeddings.

        :param model: MultiEncoderModel
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
        # The ".model" in this class will be the single-instrument encoder
        self.multi_encoder_model = multi_encoder_model
        super().__init__(model=self.multi_encoder_model.stem_encoder,
                         sr=sr, batch_size=batch_size, use_cuda=use_cuda,
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

    def estimate_stems_embeddings(self):
        """  Use the multi-encoder to estimate embeddings corresponding to individual stems """
        ref_sounds_df = self.sounds_df[self.sounds_df['is_reference']]  # TODO maybe delete if unused

        # The dataloader will provide more sounds than necessary, we'll keep only the mixes (not the mono-instrument
        #    tracks or single notes)
        n_stems_per_mix = (self.mix_dataset.n_tracks_per_item - 1)
        batch_size_with_extra_tracks = self.batch_size * self.mix_dataset.n_tracks_per_item
        dataloader = self.mix_dataset.get_dataloader(batch_size_with_extra_tracks, num_workers=self.num_workers)

        estimated_stems_df = {'embedding': [], 'embedding_family': [], 'GT_instrument_UID': []}
        with torch.no_grad():
            for minibatch in dataloader:
                audios, track_names, mix_labels = minibatch
                mix_tracks_mask = (mix_labels < 0)
                n_mixes = mix_tracks_mask.sum()
                assert n_mixes == (mix_tracks_mask.shape[0] // self.mix_dataset.n_tracks_per_item)
                # Multi-channel audio
                audios = audios.to(self.multi_encoder_model.device)
                unmixed_embeds = self.multi_encoder_model.estimate_unmixed_embeddings(audios[mix_tracks_mask, :]).cpu()
                # Retrieve corresponding instrument UIDs, while performing checks when possible
                GT_tracks_names = np.asarray(track_names)[~mix_tracks_mask].tolist()
                for mix_i in range(n_mixes):
                    expected_prefix = f'Mix_{mix_i}__'
                    for stem_j in range(n_stems_per_mix):
                        track_name = GT_tracks_names[mix_i * n_stems_per_mix + stem_j]
                        assert track_name.startswith(expected_prefix)
                        GT_instrument_UID = track_name.replace(expected_prefix, '')
                        estimated_stems_df['GT_instrument_UID'].append(GT_instrument_UID)
                        estimated_stems_df['embedding'].append(unmixed_embeds[mix_i, stem_j, :])  # Remove 1 dim
                        # Store the family to which this stem corresponds (fixed position)
                        estimated_stems_df['embedding_family'].append(self.mix_dataset.mixed_families[stem_j])

        estimated_stems_df = pd.DataFrame(estimated_stems_df)
        return estimated_stems_df

    def perform_eval(self, distance_metrics=('cosine', ), top_k=(1, 5, 10)):
        device = 'cuda:0' if self.use_cuda else self.multi_encoder_model.device
        self.multi_encoder_model = self.multi_encoder_model.to(device)  # Includes self.model, part of self.multi_encoder_model
        self.multi_encoder_model.eval()
        with self._measure_time(f'[TimbreMixSearch-MultiEncoder-{self.eval_split}] Performing evaluation'):
            # 1.   Compute embeddings
            if False:  # FIXME, REMOVE
                with open('/tmp/tmpzgu0z6zg_dataframes.pkl', 'rb') as f:
                    est_stems_embeds_df, ref_embeddings_df = pickle.load(f)
            else:
                with self._measure_time("[TimbreMixSearch-MultiEncoder] Computing 'unmixed' embeddings from mixtures sounds"):
                    est_stems_embeds_df = self.estimate_stems_embeddings()
                with self._measure_time("[TimbreMixSearch-MultiEncoder] Computing embeddings for reference sounds"):
                    ref_embeddings_df = self.compute_embeddings_from_df(self.sounds_df[self.sounds_df['is_reference']])

            # 2a.   Prepare the columns of est_stems_embeds_df to read or store results
            est_stems_embeds_df['GT_instrument_index'] = est_stems_embeds_df['GT_instrument_UID'].apply(
                lambda x: self._instrument_UID_to_index[x])
            # TODO Sanity check: all GT instruments indexes must be among the references...
            nones = [None] * len(est_stems_embeds_df)
            for dist_name in distance_metrics:
                est_stems_embeds_df[f'{dist_name}__GT_rank'] = pd.Series(nones, dtype='Int64')
                for k in top_k:
                    est_stems_embeds_df[f'{dist_name}__top_{k}_UIDs'] = pd.Series(nones, dtype='object')
                    est_stems_embeds_df[f'{dist_name}__GT_in_top_{k}'] = pd.Series(nones, dtype='boolean')
            # 2b.   Also retrieve the families now for the ref dataframe
            _ref_df = self.sounds_df[self.sounds_df['is_reference']]
            ref_embeddings_df['family'] = ref_embeddings_df['_instrument_index'].apply(
                lambda x: _ref_df[_ref_df['_instrument_index'] == x][self.mix_dataset.instrument_group_source].values[0]
            )
            #  and prepare masks to quickly retrieve reference sounds for each type of embedding family
            masks_by_family = {family: (ref_embeddings_df['family'] == family) for family in self.mix_dataset.mixed_families}

            # 3.    a. Distance and metrics computations (for each distance)
            metrics = {}
            ref_embeds = torch.from_numpy(np.vstack(ref_embeddings_df['embedding'].values)).to(device)  # Matrix
            est_stems_embeds = torch.from_numpy(np.vstack(est_stems_embeds_df['embedding'])).to(device)
            for dist_name in distance_metrics:
                with self._measure_time(f"[TimbreMixSearch-MultiEncoder] computing and sorting {dist_name} distances"):
                    distance_function = self.available_distances[dist_name]
                    # Distance computation on CUDA is so fast we don't need to measure it. The sorting, later, is slow...
                    #with self._measure_time(f"[TimbreMixSearch-MultiEncoder] {dist_name} distance using {device=}"):
                    with torch.no_grad():
                        distances_matrix = distance_function(ref_embeds, est_stems_embeds).cpu().numpy()

                    for row_i, (df_idx, row) in enumerate(est_stems_embeds_df.iterrows()):
                        family_mask = masks_by_family[row['embedding_family']]
                        # Establish correspondences between DF indexes and the pre-computed distances
                        refs_distances_df = pd.Series(distances_matrix[:, row_i], index=ref_embeddings_df.index)
                        # For a fair comparison with the contrastive model: only the same family of
                        #    references should be provided for matching (family corresponding
                        #    to the currently considered multi-encoder output embedding)
                        refs_distances_df = refs_distances_df[family_mask]
                        refs_distances_df = refs_distances_df.rename('distance').to_frame()
                        cols_to_copy = ['split', 'instrument_UID', '_instrument_index']
                        refs_distances_df[cols_to_copy] = ref_embeddings_df.loc[family_mask, cols_to_copy]

                        # Perform the sort now on a per-family basis
                        refs_distances_df = refs_distances_df.sort_values(by='distance')
                        refs_distances_df['sort_rank'] = range(len(refs_distances_df))
                        _rank_idx = (refs_distances_df['_instrument_index'] == row['GT_instrument_index']).idxmax()
                        GT_rank = refs_distances_df.at[_rank_idx, 'sort_rank']
                        est_stems_embeds_df.at[row.name, f'{dist_name}__GT_rank'] = GT_rank

                        matching_instruments = refs_distances_df.iloc[0:max(top_k)]['instrument_UID']
                        for k in top_k:
                            top_k_instruments = matching_instruments[0:k].values.tolist()
                            est_stems_embeds_df.at[row.name, f'{dist_name}__top_{k}_UIDs'] = top_k_instruments
                            is_in_top_k = (row['GT_instrument_UID'] in top_k_instruments)
                            est_stems_embeds_df.at[row.name, f'{dist_name}__GT_in_top_{k}'] = is_in_top_k

                # 3b. Compute metrics
                metrics[f'{dist_name}__OVERALL__GT_rank'] = est_stems_embeds_df[f'{dist_name}__GT_rank'].mean()
                for k in top_k:
                    metrics[f'{dist_name}__OVERALL__top_{k}_acc'] = est_stems_embeds_df[f'{dist_name}__GT_in_top_{k}'].mean()
                # Retrieve the per-family metrics easily now
                #    Where the families here are those of the FIXED-POSITION output embeddings of the multi-encoder
                for family in est_stems_embeds_df['embedding_family'].unique():
                    sub_df = est_stems_embeds_df[est_stems_embeds_df['embedding_family'] == family]
                    metrics[f'{dist_name}__{family}__GT_rank'] = sub_df[f'{dist_name}__GT_rank'].mean()
                    for k in top_k:
                        metrics[f'{dist_name}__{family}__top_{k}_acc'] = sub_df[f'{dist_name}__GT_in_top_{k}'].mean()

        return est_stems_embeds_df, metrics

    @staticmethod
    def select_main_metrics(metrics, main_distance_metric='cosine'):
        import timbremixsearch  # Here to prevent circular imports
        return timbremixsearch.TimbreMixSearchEvaluator.select_main_metrics(metrics, main_distance_metric)


def main():
    print('WARNING: currently, this script should be run as main only for debugging purposes (cannot reload a trained model / checkpoint yet...)')
    parser = argparse.ArgumentParser(description='MultiEnc MixSearch')
    parser.add_argument('config', type=str, help='Path to the YAML configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    multi_encoder = MultiEncoderModel(config)

    device_str = config['training']['device']
    use_cuda = isinstance(device_str, str) and device_str.startswith('cuda')

    evaluator = TimbreMixSearchEvaluator(
        multi_encoder_model=multi_encoder,
        sr=config['model']['sr'],
        batch_size=config['training']['batch_size'],
        use_cuda=use_cuda,
        num_workers=config['training']['num_workers'],
        verbose=True,
    )

    est_stems_embeds_df, metrics = evaluator.perform_eval()
    for k, v in metrics.items():
        print(f"{k}: {v:.1%}" if k.endswith('_acc') else f"{k}: {v}")
    pass


if __name__ == '__main__':
    main()
