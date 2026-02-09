import warnings
import argparse
from functools import cached_property
from typing import Optional, List
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm

from mergeddataset import MultiLabelDataset


class MixDataset(torch.utils.data.Dataset):
    def __init__(self, split: str,
                 instrument_group_source='midi_instrument_group',
                 families=('Chromatic Percussion', 'Bass', 'Synth Lead'),  # Warning: check upper/lower case...
                 use_single_notes_as_references=False,
                 contrastive_dataloader=True,
                 target_sr=48000, max_audio_len_s=10.0, pad='right',
                 use_small_dataset=False, exclude_augmented_train=False,
                 use_cache=True,
                 ):
        """
        A dataset that mixes samples from a MultiLabelDataset.

        :param use_single_notes_as_references: If False, the dataset will return individual tracks (with multiple
            notes for each instrument) as well as the mix. if True, returns individual notes (e.g. only the median
            pitch for a given instrument_UID) instead of multi-note tracks/stems.
            The "Mix" track, however, is always multi-note.
        """
        assert split in ['test', 'valid', 'train'], "'all' split not allowed here"
        if isinstance(use_small_dataset, bool):
            warnings.warn("Shouldn't set use_small_dataset=True for mixing sounds")
        else:
            assert isinstance(use_small_dataset, dict)
            assert all([not use_small_dataset[split] for split in use_small_dataset.keys()])

        # The size of this dataset will be different from the MultiLabelDataset - arbitrary...
        if not use_small_dataset:
            self.size = 5000 if split in ['test', 'valid'] else 40000
        else:
            self.size = 500 if split in ['test', 'valid'] else 40000
        self.use_single_notes_as_references = use_single_notes_as_references
        self.contrastive_dataloader = contrastive_dataloader

        self.pad = pad
        self.multi_label_dataset = MultiLabelDataset(
            split=split,
            use_small_dataset=use_small_dataset,
            pad=None,  # Don't pad notes for this nested dataset used mostly for single notes
            target_sr=target_sr,
            max_audio_len_s=max_audio_len_s,
            exclude_augmented_train=exclude_augmented_train,
        )

        # keep the dataframe of single-note sounds - we'll build tracks from these pre-rendered notes
        self.notes_df = self.multi_label_dataset.df[self.multi_label_dataset.df['midi_pitch'].notna()].copy()
        # Build some instrument INDICES (integers, not string UID) for a much faster retrieval of notes later
        # (search in a DF). And the same for families (integer instead of string)
        self.instrument_index_to_UID = sorted(self.notes_df['instrument_UID'].unique())
        self.instrument_UID_to_index = {UID: i for i, UID in enumerate(self.instrument_index_to_UID)}
        self.notes_df['instrument_index'] = self.notes_df['instrument_UID'].apply(lambda x: self.instrument_UID_to_index[x])

        assert instrument_group_source in ['midi_instrument_group', 'instrument_family']
        assert instrument_group_source == 'midi_instrument_group', "Only MIDI instrument groups are supported for now (need to sample from MIDI pitches' distributions associated to MIDI groups)"
        self.instrument_group_source = instrument_group_source
        self.notes_df[self.instrument_group_source] = self.notes_df[self.instrument_group_source].fillna('none')
        instrument_families = sorted(self.notes_df[self.instrument_group_source].unique())
        self.family_str_to_index = {s: i for i, s in enumerate(instrument_families)}
        self.notes_df['family_index'] = self.notes_df[self.instrument_group_source].apply(lambda x: self.family_str_to_index[x])

        # Restrict the notes to the currently selected families
        self.mixed_families = families
        for mixed_family in self.mixed_families:
            assert mixed_family in instrument_families, f"{mixed_family=} not available in the dataset. Available families: {', '.join(instrument_families)}"
        mixed_families_i = [self.family_str_to_index[s] for s in self.mixed_families]
        self.notes_df = self.notes_df[self.notes_df['family_index'].isin(mixed_families_i)]

        # Pre-compute all instruments UIDs for each item now... should help for constrastive training later
        self.precomputed_instruments = self._get_cached_precomputed_instruments(use_cache=use_cache)

        # Attributes for generating notes and mixing tracks
        self.silence_probability = 0.25
        self.percussions_gain = 4.0  # Gain applied to percussion tracks in the mix

        # The seed may be changed by the *training* dataset (new seed each time an item is retrieved)
        self._current_items_seeds = [i for i in range(len(self))]

        # TODO Maybe don't use augmented sounds for training? Or maybe do?

    def __str__(self):
        return (f"{self.__class__.__name__} (split={self.split}) with {len(self)} items,\nmixing "
                f"{len(self.mixed_families)} instrument families: {', '.join(self.mixed_families)}.\n"
                f"|_ Underlying MultiLabelDataset: {len(self.multi_label_dataset)} audio files, "
                f"{len(self.multi_label_dataset.df['instrument_UID'].unique())} instruments.\n"
                f"|_ Audio: {self.sr/1000:.1f} kHz, max length {self.max_audio_len_s:.1f}s, {len(self.mixed_families) + 1} tracks / item.\n"
                f"|_ Notes and mixing parameters: silence probability {self.silence_probability:.2f}, "
                f"percussion gain {self.percussions_gain:.1f}x.")

    @property
    def sr(self):
        return self.multi_label_dataset.target_sr

    @property
    def max_audio_len_s(self):
        return self.multi_label_dataset.max_audio_len_s

    @property
    def midi_dataset(self):
        return self.multi_label_dataset.midi_dataset

    @property
    def split(self):
        return self.multi_label_dataset.split

    def _get_cached_precomputed_instruments(self, invalidate_cache=False, use_cache=True):
        """
        Get cached precomputed instruments or compute them if not cached.

        Args:
            invalidate_cache: If True, ignore the cache and recompute the instruments

        Returns:
            DataFrame with precomputed instruments
        """
        if not use_cache:
            return self._precompute_instruments()

        cache_dir = Path("cache")
        cache_file = cache_dir / f"MixDataset_{self.split}.pkl"

        if cache_file.exists() and not invalidate_cache:
            try:
                with open(cache_file, "rb") as f:
                    cache_data = pickle.load(f)
                # Verify cache data matches current data
                notes_df_match = (self.notes_df.index.equals(cache_data["notes_df"].index) and 
                                 self.notes_df["instrument_UID"].equals(cache_data["notes_df"]["instrument_UID"]))
                instrument_match = (self.instrument_index_to_UID == cache_data["instrument_index_to_UID"])
                families_match = (self.mixed_families == cache_data["mixed_families"])
                if notes_df_match and instrument_match and families_match:
                    return cache_data["precomputed_instruments"]
                else:
                    print("[MixDataset] Cached data does not match current configuration. Recomputing precomputed instruments.")
            except (FileNotFoundError, EOFError, pickle.UnpicklingError, AttributeError, KeyError) as e:
                warnings.warn(f"[MixDataset] Error while reading precomputed instruments cache:\n{e}")

        # If we get here, either the cache doesn't exist, is invalid, or we're forcing recomputation
        precomputed_instruments = self._precompute_instruments()
        # Save to cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "precomputed_instruments": precomputed_instruments,
            "notes_df": self.notes_df,
            "instrument_index_to_UID": self.instrument_index_to_UID,
            "mixed_families": self.mixed_families
        }
        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)

        return precomputed_instruments

    def _precompute_instruments(self, seed_offset=0):
        """
        Pre-compute instruments UIDs for each item.
        Sample the instrument by randomly sampling a note (even if not in the scale) and select the corresponding
            instrument. We should do this because the original datasets of notes is properly balanced, but some
            sub-datasets (e.g., nsynth) have fewer instruments but a lot of notes; otherwise them would almost never
            be sampled.

        Returns a DataFrame with columns: 'item_index', 'instrument_UID', 'instrument_index', 'family_index', 'family_name'
        Uses a distinct np.random.default_rng with seed as the item index.
        """
        records = []
        # Absolutely needs NOT to be re-computed for each item (large DF, slow search, even on integers)
        notes_df_by_family = {
            family: self.notes_df[self.notes_df['family_index'] == self.family_str_to_index[family]]
            for family in self.mixed_families
        }
        # Local dataframe to keep track of advancement
        instruments_usage_df = list()
        for family, family_df in notes_df_by_family.items():
            for instrument_index in family_df['instrument_index'].unique():
                instruments_usage_df.append({
                    'in_dataset': False,
                    'instrument_index': instrument_index,
                    'instrument_UID': self.instrument_index_to_UID[instrument_index],
                    'family_index': self.family_str_to_index[family],
                    'family_name': family,
                })
        instruments_usage_df = pd.DataFrame(instruments_usage_df)
        # We ensure NOW that all instrument are in the dataset... maybe do that
        # 1) Only use instruments which are not yet in the dataset
        # 2) when all instruments are present (there are much less instruments than sounds), just randomly
        #       sample some instruments
        all_instruments_in_dataset = False
        for i in range(len(self)):
            rng = np.random.default_rng(seed=(i + seed_offset))
            for family, family_df in notes_df_by_family.items():
                family_index = self.family_str_to_index[family]
                chosen_row = None
                if not all_instruments_in_dataset:  # initial mode: force sample instrument among those not yet used
                    # Check that some instruments are still available in this family (otherwise, random choice below...)
                    available_instruments = instruments_usage_df[
                        (instruments_usage_df['family_index'] == family_index)
                        & (~instruments_usage_df['in_dataset'])
                    ]
                    if len(available_instruments) > 0:
                        chosen_row = available_instruments.iloc[rng.integers(0, len(available_instruments))]
                        instruments_usage_df.loc[instruments_usage_df['instrument_index'] == chosen_row['instrument_index'], 'in_dataset'] = True
                        # check is all have been used once, now (faster processing after that init phase)
                        all_instruments_in_dataset = np.all(instruments_usage_df['in_dataset'])
                    else:
                        chosen_row = None

                if chosen_row is None:  # "normal" mode (if chosen row is not selected yet): just randomly get an instrument
                    chosen_row = family_df.iloc[rng.integers(0, len(family_df))]

                records.append({
                    'item_index': i,
                    'instrument_UID': chosen_row['instrument_UID'],
                    'instrument_index': chosen_row['instrument_index'],
                    'family_index': family_index,
                    'family_name': family
                })
        # Sanity check: all instruments must be present
        df = pd.DataFrame(records)
        assert len(df['instrument_UID'].unique()) == len(self.notes_df['instrument_UID'].unique())
        return df

    @cached_property
    def C_midi_scales(self):
        return {
        "Major": [60, 62, 64, 65, 67, 69, 71],
        "Natural Minor": [60, 62, 63, 65, 67, 68, 70],
        "Harmonic Minor": [60, 62, 63, 65, 67, 68, 71],
        "Melodic Minor": [60, 62, 63, 65, 67, 69, 71],
        "Dorian": [60, 62, 63, 65, 67, 69, 70],
        "Phrygian": [60, 61, 63, 65, 67, 68, 70],
        "Lydian": [60, 62, 64, 66, 67, 69, 71],
        "Mixolydian": [60, 62, 64, 65, 67, 69, 70],
        #"Locrian": [60, 61, 63, 65, 66, 68, 70],
        #"Whole Tone": [60, 62, 64, 66, 68, 70],
        "Blues": [60, 63, 65, 66, 67, 70],
        "Pentatonic Major": [60, 62, 64, 67, 69],
        "Pentatonic Minor": [60, 63, 65, 67, 70],
        #"Octatonic": [60, 61, 63, 64, 66, 67, 69, 71],
        #"Chromatic": [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
    }

    @staticmethod
    def get_full_range_scale(scale: np.ndarray):
        single_octave_scale = scale % 12
        scale = single_octave_scale.copy()
        while scale.max() < 127:
            single_octave_scale += 12
            scale = np.concatenate([scale, single_octave_scale])
        return scale[scale <= 127]

    def __len__(self):
        return self.size

    @property
    def n_tracks_per_item(self):
        return len(self.mixed_families) + 1

    def __getitem__(self, i):
        """
        For each item, we do as follows:
        - sample 1 instrument UID for each group
        - sample a key and scale
           - then get the full range of pitches (from 0 to 127) corresponding to the scale

        - sample a BPM tempo and start time (random small initial pause)
        - for each instrument UID:
            - sample a rhythm (when are notes being played)
            - get notes in the scale (if available... otherwise random notes)
            - concatenate: append notes (mono synths: cut the prev note (w/ very small fadeout) before playing the new one
            - try to equalize levels somehow... (waveform peak set to 0dB for each instru track, after merging notes?)
        - return N+1 audios if there's N instruments (e.g., N=4 isolated instruments' tracks + 1 mix)
        """
        # Instruments were pre-computed at construction
        item_instruments = self.precomputed_instruments[self.precomputed_instruments['item_index'] == i]
        assert item_instruments['family_name'].to_list() == list(self.mixed_families)

        # If training dataset, always use a different seed! But in a deterministic way. So: store the last seed
        #     used for a given item, add the len(self)
        if self.split == 'train':
            seed = self._current_items_seeds[i]
            self._current_items_seeds[i] += len(self)
        else:
            seed = i
        rng = np.random.default_rng(seed=seed)

        # Compute the full-range scale first - we'll samples pitches from it later
        scale_name = rng.choice(list(self.C_midi_scales.keys()))
        scale = np.asarray(self.C_midi_scales[scale_name]) + rng.integers(0, 12)
        scale = self.get_full_range_scale(scale)

        # Generate the rhythm (not the pitches or velocities) for each family: generate a list of integers that
        #    indicates the length (in samples of each note). E.g., @16kHz, BPM=60 (1 quarter note/second) :
        #    [8000, 8000, 16000] represents two eighth notes and one quarter note
        bpm = rng.uniform(60.0, 140.0)
        n_quarter = self.sr / (bpm / 60.0)  # (float...) Number of samples for each quarter note
        n_samples_total = int(self.sr * self.max_audio_len_s)
        # Fewer notes for percussion and bass (?)... otherwise, same distribution for all
        audio_notes_len = [list() for _ in self.mixed_families]
        perc_tracks_indices = list()
        for track_i, family in enumerate(self.mixed_families):
            prev_n_rand = 0  # to "Humanize" (small errors on notes' onsets)
            # Probability of the half, quarter, eighth, sixteenth notes
            if family.lower() in ['percussion', 'chromatic percussion']:
                note_len_ratio_probs = {2.0: 0.0, 1.0: 0.8, 0.5: 0.2, 0.25: 0.0}
                perc_tracks_indices.append(track_i)
            elif family in ['bass', ]:
                note_len_ratio_probs = {2.0: 0.45, 1.0: 0.4, 0.5: 0.1, 0.25: 0.05}
            else:
                note_len_ratio_probs = {2.0: 0.1, 1.0: 0.3, 0.5: 0.3, 0.25: 0.3}
            ratios, probs = list(note_len_ratio_probs.keys()), list(note_len_ratio_probs.values())
            #  Keep adding some notes until the sum of lengths more than the total
            while sum(audio_notes_len[track_i], 0) < n_samples_total:
                note_len = int(rng.choice(ratios, p=probs) * n_quarter) - prev_n_rand  # Not to accumulate errors
                prev_n_rand = int(rng.uniform(-0.02*n_quarter, 0.02*n_quarter))  # 3%-5% is "audible", but depends...
                note_len += prev_n_rand  # Not to accumulate errors
                audio_notes_len[track_i].append(note_len)
            # Reduce the last note's length such that the total is exactly what we need
            n_extra = sum(audio_notes_len[track_i], 0) - n_samples_total
            audio_notes_len[track_i][-1] -= n_extra

        # Generate audio tracks for each instrument family
        audio_tracks = torch.zeros((self.n_tracks_per_item, n_samples_total))
        for track_i, family in enumerate(self.mixed_families):
            # Get the pre-computed instrument UID and index for this family
            family_row = item_instruments[item_instruments['family_name'] == family].iloc[0]
            instrument_index = family_row['instrument_index']
            family_index = family_row['family_index']

            family_df = self.notes_df[self.notes_df['family_index'] == family_index]
            instrument_df = family_df[family_df['instrument_index'] == instrument_index]
            notes_in_scale = instrument_df[instrument_df['midi_pitch'].isin(scale)]
            if len(notes_in_scale) == 0:  # If no note not could be found in the scale, just get a random note
                notes_in_scale = instrument_df.loc[rng.choice(instrument_df.index)]  # pd.Series
                notes_in_scale = notes_in_scale.to_frame().T  # Here, we retrieve a one-line DF
            # Generate the track from the "rhythm" (the delays between notes - no actual pause...)
            cur_sample = 0
            for note_len in audio_notes_len[track_i]:
                if track_i not in perc_tracks_indices:
                    silence = rng.choice([True, False], p=[self.silence_probability, 1.0-self.silence_probability])
                else:
                    silence = False

                if not silence:
                    # Don't sample uniform pitches... Instead, use the distributions from Slakh
                    # Otherwise we get high-pitched unrealistic basses, distorted percs and pianos, ...
                    pitches_in_scale = [int(p) for p in notes_in_scale['midi_pitch'].values]
                    # this requires the MIDI families... not the other merged (but better) groups!
                    #   Is ok for basic families, though (was not OK for classification tasks)
                    pitches_probs = []
                    for pitch in pitches_in_scale:  # FIXME sample by pitch: put that in a dedicated method (will be reused)
                        try:
                            entry_index = self.midi_dataset.notes_distributions['pitch'][family]['values'].index(pitch)
                            pitches_probs.append(self.midi_dataset.notes_distributions['pitch'][family]['probabilities'][entry_index])
                        except ValueError:
                            pitches_probs.append(0.0)
                    if sum(pitches_probs) > 0:
                        pitches_probs = np.asarray(pitches_probs) / sum(pitches_probs)
                        note_row = notes_in_scale.loc[rng.choice(notes_in_scale.index, p=pitches_probs)]
                    else:
                        note_row = notes_in_scale.loc[rng.choice(notes_in_scale.index)]
                    audio = self.multi_label_dataset.get_audio(note_row['path'])
                    # Maybe cut (with fadeout) or pad with zeros
                    if len(audio) < note_len:
                        audio = F.pad(audio, (0, note_len - len(audio)))
                    elif len(audio) > note_len:
                        audio = audio[:note_len]
                        n_fadeout = min(int(self.sr * 0.010), note_len)  # 10ms fadeout
                        audio[-n_fadeout:] *= torch.linspace(1.0, 0.0, n_fadeout)

                else:  # Silent "note"
                    audio = torch.zeros((note_len, ))  # Actually useless (audios init to 0.0) but cleaner code

                audio_tracks[track_i, cur_sample:cur_sample+note_len] = audio
                cur_sample += note_len

        # Handle volumes, then Mix tracks
        max_amplitudes = torch.abs(audio_tracks[:-1, :]).max(dim=1)[0] / 0.99  # Keep a small amplitude margin vs. 0dB
        audio_tracks[:-1, :] *= (1.0 / max_amplitudes.unsqueeze(1))  # Unsqueeze adds a dim for proper broadcast
        for track_i in perc_tracks_indices:  # If there's a percussion track, mix it twice as loud
            audio_tracks[track_i, :] *= self.percussions_gain
        audio_tracks[-1, :] = torch.sum(audio_tracks[:-1, :], dim=0)
        audio_tracks[-1, :] *= (0.99 / torch.max(torch.abs(audio_tracks[-1, :])))
        for track_i in perc_tracks_indices:  # Re-normalize the volume of the perc tracks after mixing
            audio_tracks[track_i, :] *= (1.0 / self.percussions_gain)

        # Small shift (smaller than the shortest note), not to always have a precise multi-note onset at t=0
        n_init_shift = rng.integers(0, int(n_quarter * 0.1))
        audio_tracks = audio_tracks.roll(n_init_shift, dims=1)

        # For the first channels: allow to return individual notes instead of multi-note tracks
        if self.use_single_notes_as_references:
            for track_i, family in enumerate(self.mixed_families):
                instrument_UID = item_instruments[item_instruments['family_name'] == family].iloc[0]['instrument_UID']
                # Random pitches for training, fixed median for valid/test
                note_row = self.multi_label_dataset.retrieve_single_note_from_instrument(
                    instrument_UID, median_pitch=(self.split == 'train'), rng=rng,
                )
                audio = self.multi_label_dataset.get_audio(note_row['path'])
                if self.pad is not None and self.pad.lower() == 'right':
                    audio = F.pad(audio, (0, n_samples_total - audio.shape[0]))
                else:
                    raise ValueError(f"Unsupported pad argument {self.pad=}")  # None not allowed
                audio_tracks[track_i, :] = audio * (0.99 / audio.max())

        #  Return audios AND some instrument labels, including a label for the mix
        return audio_tracks, (item_instruments['instrument_UID'].to_list() + ["Mix", ])

    @staticmethod
    def _dataloader_collate_fn(batch):
        """
        Collate function for the dataloader.

        Args:
            batch: A list of tuples (audio_tracks, instrument_uids) from __getitem__

        Returns:
            Processed batch data ready for the model
        """
        audio_tracks, instrument_UIDs = zip(*batch)
        # Change the "mix" UID...
        signed_labels = []
        for i in range(len(instrument_UIDs)):
            assert instrument_UIDs[i][-1] == "Mix"
            instrument_UIDs[i][-1] += f"_{i}"
            for j in range(len(instrument_UIDs[i])-1):
                instrument_UIDs[i][j] = f"Mix_{i}__{instrument_UIDs[i][j]}"
            signed_labels += ([i+1] * len(instrument_UIDs[i]))
            signed_labels[-1] *= -1  # Use negatives labels, which will indicate the mix of the corresponding positive labels
        # Return a matrix of audio (not a batch of multi-track audio) - so everything can easily processed by a
        #    model without reshaping first
        audio_tracks = torch.stack(audio_tracks)
        audio_tracks = audio_tracks.view(-1, audio_tracks.shape[-1])
        return audio_tracks, sum(instrument_UIDs, []), torch.tensor(signed_labels)

    def get_dataloader(self, batch_size: int, seed=20250711, num_workers=0) -> torch.utils.data.DataLoader:
        """
        Get a DataLoader for this dataset (specific for contrastive learning, or not).

        Args:
            batch_size: Number of audio tracks per batch
            seed: Random seed for the sampler
            num_workers: Number of worker processes for data loading

        Returns:
            A DataLoader instance
        """
        if self.contrastive_dataloader:

            sampler = MixContrastiveSampler(self, batch_size, shuffle=(self.split == 'train'), seed=seed)
            dataloader = torch.utils.data.DataLoader(
                dataset=self,
                batch_size=sampler.n_mixes_per_minibatch,
                num_workers=num_workers,
                sampler=sampler,
                drop_last=True,
                collate_fn=self._dataloader_collate_fn,
            )
        else:
            sampler = torch.utils.data.RandomSampler(self, generator=torch.Generator().manual_seed(seed))
            assert batch_size % self.n_tracks_per_item == 0, f"{batch_size=} must be divisible by {self.n_tracks_per_item=}"
            n_mixes_per_minibatch = batch_size // self.n_tracks_per_item
            dataloader = torch.utils.data.DataLoader(
                dataset=self,
                batch_size=n_mixes_per_minibatch,
                num_workers=num_workers,
                sampler=(sampler if self.split == 'train' else None),
                shuffle=(False if self.split != 'train' else None),
                drop_last=(self.split == 'train'),
                collate_fn=self._dataloader_collate_fn,
            )
        return dataloader


class MixContrastiveSampler(torch.utils.data.Sampler):
    def __init__(self, data_source: MixDataset, batch_size: int, seed=20250711, shuffle=False):
        """
        Random sampler for MixDataset, that is suitable for contrastive training and validation.
        Ensures that different instruments UIDs are not found twice in a minibatch (except in the
        mix in which that instrument can be found)

        Args:
            data_source: The MixDataset to sample from
            batch_size: Number of samples per batch (the same must be provided to the dataloader)
            seed: Random seed
            shuffle: If False, the sampler will keep the same batches for each epoch (still considers that instruments
                     in the provided MixDataset are properly shuffled). If True, a random batch, *with replacement*,
                     will be sampled at each step.
        """
        super().__init__()
        self.data_source, self.batch_size, self.seed, self.shuffle = data_source, batch_size, seed, shuffle
        assert batch_size % data_source.n_tracks_per_item == 0, f"{batch_size=} must be divisible by {data_source.n_tracks_per_item=}"
        self.n_mixes_per_minibatch = batch_size // data_source.n_tracks_per_item

        if not shuffle:
            self.epoch_df, self.mix_indices_per_minibatch = self._get_deterministic_epoch_df()
            self.n_minibatches = len(self.mix_indices_per_minibatch)
        else:
            pass  # The __iter__ method will do everything. TODO setup an arbitrary length, though...
            # data_source returns some track audio
            self.epoch_df, self.mix_indices_per_minibatch = None, None
            self.n_minibatches = (len(self.data_source) * self.data_source.n_tracks_per_item) // self.batch_size
        self._current_epoch_seed = seed

    def _get_deterministic_epoch_df(self, invalidate_cache=False):
        """
        Get cached deterministic epoch dataframe or build it if not cached.

        Args:
            invalidate_cache: If True, ignore the cache and rebuild the epoch dataframe

        Returns:
            Tuple of (epoch_df, mix_indices_per_minibatch)
        """
        cache_dir = Path("cache")
        cache_file = cache_dir / f"MixRandomSampler_{self.data_source.split}.pkl"

        if cache_file.exists() and not invalidate_cache:
            try:
                with open(cache_file, "rb") as f:
                    cache_data = pickle.load(f)

                # Verify cache data matches current configuration
                seed_match = self.seed == cache_data["seed"]
                shuffle_match = self.shuffle == cache_data["shuffle"]
                batch_size_match = self.batch_size == cache_data["batch_size"]
                # Check for perfect item-by-item equality of instrument data (stored in the dataset)
                instrument_match = self.data_source.instrument_index_to_UID == cache_data.get("instrument_index_to_UID", [])
                families_match = self.data_source.mixed_families == cache_data.get("mixed_families", [])
                precomputed_match = self.data_source.precomputed_instruments.equals(cache_data["precomputed_instruments"])

                if seed_match and shuffle_match and batch_size_match and instrument_match and families_match and precomputed_match:
                    return cache_data["epoch_df"], cache_data["mix_indices_per_minibatch"]
                else:
                    print("[MixRandomSampler] Cached data does not match current configuration. Rebuilding epoch dataframe.")
            except (FileNotFoundError, EOFError, pickle.UnpicklingError, AttributeError, KeyError) as e:
                warnings.warn(f"[MixRandomSampler] Error while reading cached epoch dataframe:\n{e}")

        # If we get here, either the cache doesn't exist, is invalid, or we're forcing rebuilding
        epoch_df, mix_indices_per_minibatch = self._build_deterministic_epoch_df()

        # Save to cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "epoch_df": epoch_df,
            "mix_indices_per_minibatch": mix_indices_per_minibatch,
            # Additional data for more thorough cache validation
            "seed": self.seed,
            "shuffle": self.shuffle,
            "batch_size": self.batch_size,
            "instrument_index_to_UID": self.data_source.instrument_index_to_UID,
            "mixed_families": self.data_source.mixed_families,
            "precomputed_instruments": self.data_source.precomputed_instruments
        }

        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)

        return epoch_df, mix_indices_per_minibatch

    def _build_deterministic_epoch_df(self):
        tracks_df = self.data_source.precomputed_instruments.copy()
        tracks_df.rename(columns={'item_index': 'mix_index'}, inplace=True)
        tracks_df['minibatch_index'] = -1

        # Build the minibatches one by one, check that there's no overlap.
        #    For the last batches, overlap will happen if we only take some unused instrument_UIDs
        #    So we'll allow to reuse some non-overlaping instruments, which were already used in the previous batches.
        tracks_df['extra_minibatch_indices'] = [[] for _ in range(len(tracks_df))]
        mix_indices_per_minibatch = []
        rng = np.random.default_rng(seed=self.seed)

        # Use tqdm to show progress of assigning instruments to minibatches
        pbar = tqdm(total=len(tracks_df), desc="[MixRandomSampler] Building deterministic epoch dataframe")
        remaining = len(tracks_df)

        while tracks_df['minibatch_index'].min() < 0:  # While all instruments don't have an associated minibatch
            current_batch_mix_indices = []
            current_batch_instrument_indices = []

            # The last minibatch will be completed with re-used instruments (so all instruments will be in the dataset)
            while len(current_batch_mix_indices) < self.n_mixes_per_minibatch:
                # We'll first try to use some unused tracks
                available_tracks = tracks_df[tracks_df['minibatch_index'] < 0].copy()
                chosen_tracks_df = self._get_non_overlaping_tracks(available_tracks, current_batch_instrument_indices, rng)
                if chosen_tracks_df is not None:
                    tracks_df.loc[chosen_tracks_df.index, 'minibatch_index'] = len(mix_indices_per_minibatch)
                else:  # No available tracks/instrumentsUIDs without overlap
                    # At this point we just use all tracks (don't care if some tracks are used three times or more...
                    # which is sub-optimal). Otherwise this fails even for small batch sizes (24...).
                    max_n_tries = (100 if self.batch_size > 24 else 10)
                    chosen_tracks_df = self._get_non_overlaping_tracks(
                        tracks_df, current_batch_instrument_indices, rng, max_n_tries=max_n_tries,
                        authorize_overlapping=True,  # Only for validation and test anyways
                    )
                    #if chosen_tracks_df is None:
                    #    # FIXME Use random non-overlapping (ignore the current usage counts)
                    #    raise RuntimeError("The instrument_UID non-overlap constraint is too strong (failed twice) for"
                    #                       "the underlying dataset, maybe it contains too few sounds")

                    extra_minibatch_indices = tracks_df.loc[chosen_tracks_df.index, 'extra_minibatch_indices'].iloc[0]
                    extra_minibatch_indices.append(len(mix_indices_per_minibatch))
                    for df_idx in chosen_tracks_df.index:
                        tracks_df.at[df_idx, 'extra_minibatch_indices'] = extra_minibatch_indices

                current_batch_instrument_indices += chosen_tracks_df['instrument_index'].to_list()
                current_batch_mix_indices.append(chosen_tracks_df['mix_index'].values[0])
            # Sanity checks: size checks, and double-check that no overlap can be found
            assert len(current_batch_mix_indices) == self.n_mixes_per_minibatch
            assert len(current_batch_instrument_indices) == self.n_mixes_per_minibatch * len(self.data_source.mixed_families)
            #assert len(current_batch_instrument_indices) == len(set(current_batch_instrument_indices))

            mix_indices_per_minibatch.append(current_batch_mix_indices)

            # Update progress bar
            new_remaining = np.sum(tracks_df['minibatch_index'] < 0)
            pbar.update(remaining - new_remaining)
            remaining = new_remaining

        pbar.close()
        # final sanity check: double-check that all instruments can be found (by reconstructing the epoch_df...)
        epoch_df = list()
        for minibatch_idx, mix_indices in enumerate(mix_indices_per_minibatch):
            minibatch_df = pd.concat([tracks_df[tracks_df['mix_index'] == mix_idx] for mix_idx in mix_indices], axis=0)
            # extra_minibatch_indices is now a list
            in_extra_minibatch = minibatch_df['extra_minibatch_indices'].apply(lambda x: minibatch_idx in x)
            assert np.all((minibatch_df['minibatch_index'] == minibatch_idx) | in_extra_minibatch)
            epoch_df.append(minibatch_df)
        epoch_df = pd.concat(epoch_df, axis=0)
        assert len(set(epoch_df['instrument_index'])) == len(set(self.data_source.precomputed_instruments['instrument_index']))

        return epoch_df, mix_indices_per_minibatch

    def _get_non_overlaping_tracks(self, available_tracks: pd.DataFrame, current_instrument_indices: List[int],
                                   rng: np.random.Generator, max_n_tries=10, authorize_overlapping=False):
        """
        Finds a mix with tracks that don't have overlapping instrument indices with the current set.

        Args:
            available_tracks (pd.DataFrame): DataFrame containing available tracks to choose from
            current_instrument_indices (List[int]): List of instrument indices already selected
            rng (np.random.Generator): Random number generator for consistent sampling
            max_n_tries (int, optional): Maximum number of attempts to find non-overlapping tracks. Defaults to 10.

        Returns:
            pd.DataFrame or None: DataFrame of chosen tracks with no instrument overlap if found,
                                 None if no non-overlapping tracks can be found after max attempts
        """
        if len(available_tracks) == 0:
            return None
        # Keep sampling until we get one that's OK. with a limit... stop after a few attempts
        for _ in range(max_n_tries):
            chosen_mix_idx = rng.choice(available_tracks['mix_index'])  # Don't even need to "uniquify" mix indices
            chosen_tracks_df = available_tracks[available_tracks['mix_index'] == chosen_mix_idx]
            overlap = chosen_tracks_df['instrument_index'].apply(lambda x: x in current_instrument_indices)
            if not np.any(overlap):
                return chosen_tracks_df
        if not authorize_overlapping:
            return None  # If we reach this point: failure to find a non-overlapping mix after a few attempts
        else:
            return chosen_tracks_df

    def _get_random_minibatch(self, rng: np.random.Generator):
        # We consider that all tracks are available... Even if a track is randomly selected twice (which is unlikely
        # but could happen, we'll just keep trying until we find a non-overlapping instrument UIDs)
        tracks_df = self.data_source.precomputed_instruments.copy()
        tracks_df.rename(columns={'item_index': 'mix_index'}, inplace=True)

        batch_mix_indices = []
        instrument_indices = []
        for _ in range(self.n_mixes_per_minibatch):
            chosen_tracks_df = self._get_non_overlaping_tracks(
                tracks_df, instrument_indices,
                rng, max_n_tries=100
            )
            instrument_indices += chosen_tracks_df['instrument_index'].to_list()
            batch_mix_indices.append(chosen_tracks_df['mix_index'].values[0])

        return batch_mix_indices

    def __len__(self):
        """
        Length of the sampler.

        Returns:
            Number of samples in an epoch
        """
        return self.n_minibatches * self.n_mixes_per_minibatch

    def __iter__(self):
        """
        Iterator for the sampler.

        Returns:
            Iterator over dataset indices
        """
        if self.mix_indices_per_minibatch is not None:
            for minibatch_idx, mix_indices in enumerate(self.mix_indices_per_minibatch):
                for mix_idx in mix_indices:
                    yield mix_idx  # mix_idx is the index of an item for the MixDataset used with this sampler
        else:
            self._current_epoch_seed += 1
            rng = np.random.default_rng(seed=self._current_epoch_seed)
            for minibatch_idx in range(self.n_minibatches):
                mix_indices = self._get_random_minibatch(rng)  # mix_indices is actually a "minibatch"
                for mix_idx in mix_indices:
                    yield mix_idx



def main():
    """
    Create test, valid, and train datasets similar to mergeddataset.py

    Example usage:
      python mixdataset.py [-n-workers N] [-batch-size N]
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Create and test MixDataset instances')
    parser.add_argument('--n-workers', type=int, default=0, help='Number of workers for dataloader (default: 0)')
    parser.add_argument('--batch-size', type=int, default=48, help='Batch size for dataloader (default: 16)')
    args = parser.parse_args()
    print(f"{args.n_workers=} {args.batch_size=}")

    # Create datasets for each split
    for split in ['valid', 'test', 'train', ]:
        dataset = MixDataset(
            split=split, target_sr=16000, use_single_notes_as_references=True,
            use_cache=False, contrastive_dataloader=True,
        )
        print(f"\n - - - - - - - - - - {split.upper()} dataset - - - - - - - - - - -")
        print(dataset)

        # Debugging : tentative de lecture du dataset entier
        #for i in tqdm(range(len(dataset)), desc="Reading the entire dataset"):
        #    item = dataset[i]

        dataloader = dataset.get_dataloader(batch_size=args.batch_size, num_workers=args.n_workers)
        # Load all minibatches available with the dataloader
        #   For valid/test : first step can be VERY long
        for minibatch in tqdm(dataloader, desc=f"Loading {split} minibatches"):
            assert minibatch[0].shape[0] == args.batch_size
            break


if __name__ == "__main__":
    main()
