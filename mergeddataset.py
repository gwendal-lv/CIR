import copy
from pathlib import Path
import os
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torchaudio
import torch
import torch.utils.data
import torch.nn.functional as F


import multimodal_sounds_dataset.merged
import multimodal_sounds_dataset.midi
import multimodal_sounds_dataset.nsynth
import multimodal_sounds_dataset.surge



class MultimodalSoundsDataset(multimodal_sounds_dataset.merged.MergedDataset):
    def __init__(self, data_dir=None, sub_datasets=('nsynth', 'surge'), use_small_dataset=False,
                 split='all', target_sr=48000, max_audio_len_s=10.0, pad: Optional[str]=None,
                 exclude_augmented_train=False,
                 ):
        """
        Initialize the MultimodalSoundsDataset.

        Args:
            data_dir (str, optional): Path to the data directory. If None, uses './data' relative to this file.
            split: 'all', 'train', 'valid', or 'test'
            pad: If given, will pad audios with zeros such that all audios length will be max_audio_len_s
                    (in seconds). Possible values: 'right', ... TODO implement others...
        """
        if data_dir is not None:
            self.data_dir = Path(data_dir)
            if not self.data_dir.exists():
                raise ValueError(f"Provided data directory '{data_dir}' does not exist")
        else:
            self.data_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'data'
            if not self.data_dir.exists():
                raise ValueError(f"Default data directory '{self.data_dir}' does not exist. Maybe you forgot to create "
                                 f"a symlink from ./data to the actual parent dir for all datasets.")
        super().__init__(
            data_dir=self.data_dir, sub_datasets=sub_datasets, use_small_dataset=use_small_dataset, split=split,
            exclude_augmented_train=exclude_augmented_train, exclude_augmented_valid_test=True
        )

        self.orig_sr, self.target_sr = 48000, target_sr
        self.max_audio_len_s, self.pad = max_audio_len_s, pad
        self.max_audio_len_samples = int(np.round(max_audio_len_s * self.target_sr))

        # Proper distributions of MIDI notes and pitches
        self.midi_dataset = multimodal_sounds_dataset.midi.SlakhMIDI(self.data_dir)

        # TODO Assign weights/probabilities vs MIDI notes

    def __str__(self):
        return (f"{super().__str__()}\n|_ Data sampling rate: "
                f"original {self.orig_sr / 1000:.1f} kHz, resampled to {self.target_sr/1000:.1f} kHz.")

    @property
    def features_statistics(self):
        return copy.copy(multimodal_sounds_dataset.merged.FEATURES_STATISTICS)

    # - - - - - - - - - -       Labels, etc.       - - - - - - - - - -

    @property
    def id2datasource(self):
        return {i: k for i, k in enumerate(self.sub_datasets_dfs.keys())}

    @property
    def datasource2id(self):
        return {k: i for i, k in enumerate(self.sub_datasets_dfs.keys())}

    # - - - - - - - - - -       PyTorch Dataset and Dataloader        - - - - - - - - - -

    def __len__(self):
        return len(self.df)

    def get_audio(self, path: Path):
        audio, sr = torchaudio.load(self.data_dir / path)  # FIXME data_dir is already included... but pathlib handles this!
        assert sr == self.orig_sr, f"All audio files should be 48kHz (issue with {path})"
        assert len(audio.shape) == 2, "Audio is expected to be 2D when loaded (even if mono...)"
        assert audio.shape[0] == 1, "Audio must be mono"
        audio = audio[0, :]
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            audio = resampler(audio)
        # Pad all audios if required
        if self.pad is not None and audio.shape[0] < self.max_audio_len_samples:
            if self.pad.lower() == 'right':
                audio = F.pad(audio, (0, self.max_audio_len_samples - audio.shape[0]))
            else:
                raise ValueError(f"Unexpected pad argument {self.pad=}")
        # Finally check that audio is not too long
        if audio.shape[0] > self.max_audio_len_samples:
            audio = audio[:self.max_audio_len_samples]  # Brutal cut, no fadeout
        return audio

    def __getitem__(self, idx):
        item_row = self.df.iloc[idx]
        audio = self.get_audio(item_row['path'])
        # TODO Implement gain data augmentation (e.g. sample a gain in dB from a normal distrib, std +/- 6dB)
        #    also an initial delay maybe, if the audio is not too long already (for AST to learn to identify onsets
        #    anywhere in the tokens)
        # TODO Implement FX data augmentation (easy and/or fast: reverb, delay, chorus, ...)

        # Return only the audio file for this generic dataset
        return idx, self.datasource2id[item_row['data_source']], audio

    def get_dataloader(self, batch_size: int, seed=0, num_workers=0) -> torch.utils.data.DataLoader:
        assert self.pad is not None, "Batching will not properly work without padding audio waveforms"
        if self.split == 'all':
            raise ValueError("An entire dataset (not a train|valid|test split) should not be used with a dataloader")
        # Uniform sampling at the moment, TODO allow other options....
        # TODO At least: use the same amount of samples for each sub-dataset (e.g. don't use all NSynth samples
        #       at all epochs)
        # TODO Implement a weighted sampler for reweighting classes... (for the training set only, using training
        #    stats only)
        train_sampler = torch.utils.data.RandomSampler(
            data_source=self,
            generator=torch.Generator().manual_seed(seed),
            replacement=False,
            num_samples=len(self),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=(train_sampler if self.split == 'train' else None),
            shuffle=(False if self.split != 'train' else None),
            drop_last=(self.split == 'train'),
        )
        return dataloader

    def retrieve_single_note_from_instrument(
            self, instrument_UID: str, median_pitch=False, ref_velocity=100,
            rng: np.random.Generator = None,
    ):
        """ Utility method: retrieve a single note (single pitch) for a given instrument_UID """
        # first retrieve the sub-DF for that instrument
        instrument_df = self.df[self.df['instrument_UID'] == instrument_UID]
        instrument_df = instrument_df[instrument_df['midi_pitch'].notna()].copy().reset_index(drop=True)
        assert len(instrument_df) > 0, f"No audio with a single note and pitch found for instrument {instrument_UID}"
        # Then retrieve and return a single note
        if median_pitch:
            ref_pitch = instrument_df['midi_pitch'].median()  # float value - will be used for distances anyways
            instrument_df['pitch_distance'] = (instrument_df['midi_pitch'] - ref_pitch).abs()
            instrument_df['velocity_distance'] = (instrument_df['midi_velocity'] - ref_velocity).abs()
            min_pitch_distance = instrument_df['pitch_distance'].min()
            instrument_df = instrument_df[np.isclose(instrument_df['pitch_distance'], min_pitch_distance)]
            note_row = instrument_df.loc[instrument_df['velocity_distance'].idxmin()]
        else:
            assert rng is not None, "A random number generator is required to select a single note"
            note_row = instrument_df.iloc[rng.integers(0, len(instrument_df))]
        return note_row  # return the dataframe row - audio can be loaded later from the path


class MIDIInstrumentGroupClassificationDataset(MultimodalSoundsDataset):
    def __init__(self, **kwargs):  # TODO  excluded_MIDI_groups=("Drum Kit", )
        """
        Currently only works for MIDI Instrument Groups classifications (see multimodal_sounds_dataset/midi.py).
        This dataset forces the merge of different GT instrument families/categories into the main MIDI categories,
        and that merge has not been studied. This dataset should only be used for quick testing purposes.

        Args:
            kwargs: to be passed to the MultimodalSoundsDataset constructor
        """
        super().__init__(**kwargs)
        # Exclude audios which have no label (no clearly associated MIDI instrument)
        self.df = self.df[self.df['midi_instrument_group'] != '']
        # Check: only MIDI instruments are currently supported
        if not self.df['midi_instrument_group'].isin(list(multimodal_sounds_dataset.midi.MIDI_INSTRUMENT_GROUPS.keys())).all():
            raise ValueError("Some values in self.df['midi_instrument_group'] are not MIDI Instrument Groups' names")
        # Also check that all instruments are in the (currently fixed) list of MIDI group labels
        if not self.df['midi_instrument_group'].isin(self.label2id.keys()).all():
            raise ValueError("Some values in self.df['midi_instrument_group'] are not among available labels")

    @property
    def id2label(self):
        return {
            0: 'Bass', 1: 'Brass', 2: 'Chromatic Percussion', 3: 'Guitar', 4: 'Organ', 5: 'Piano', 6: 'Pipe',
            7: 'Reed', 8: 'Strings', 9: 'Synth Effects', 10: 'Synth Lead', 11: 'Synth Pad', 12: 'Voice'
        }

    @property
    def label2id(self):
        return {label: idx for idx, label in self.id2label.items()}

    @property
    def num_labels(self):
        return len(self.id2label)

    def __str__(self):
        return f"{super().__str__()}\n|_ Available labels: {list(self.label2id.keys())}"

    # - - - - - - - - - -       PyTorch Dataset and Dataloader features       - - - - - - - - - -

    def __getitem__(self, idx):
        item_idx, data_source_id, audio = super().__getitem__(idx)
        item_row = self.df.iloc[idx]
        targets = torch.zeros(self.num_labels)
        targets[self.label2id[item_row['midi_instrument_group']]] = 1.0
        return item_idx, data_source_id, audio, targets


class InstrumentDirectClassificationDataset(MultimodalSoundsDataset):
    def __init__(self, **kwargs):
        """
        Dataset that has as many labels as instruments in the dataset.

        Args:
            kwargs: to be passed to the MultimodalSoundsDataset constructor
        """
        super().__init__(**kwargs)
        self.id2label = {i: label for i, label in enumerate(self.df['instrument_UID'].unique().tolist())}
        self.label2id = {label: id for id, label in self.id2label.items()}

    @property
    def num_labels(self):
        return len(self.id2label)

    def __str__(self):
        return f"{super().__str__()}\n|_ Total available instruments: {self.num_labels=}"

    # - - - - - - - - - -       PyTorch Dataset and Dataloader features       - - - - - - - - - -

    def __getitem__(self, idx):
        item_idx, data_source_id, audio = super().__getitem__(idx)
        item_row = self.df.iloc[idx]
        targets = torch.zeros(self.num_labels)
        targets[self.label2id[item_row['instrument_UID']]] = 1.0
        return item_idx, data_source_id, audio, targets


class MultiLabelDataset(MultimodalSoundsDataset):
    def __init__(self, **kwargs):
        """
        Dataset which provides multiple labels for each sample, related to different aspect of the sound.
        Labels include:
        - the type of sound source/generation technique, called "instrument_source" in NSynth:
                synthetic, acoustic, electronic
        - the instrument family, with some families arbitrarily regrouped.... (see multimodal_sounds_dataset/merged.py)
        - the FX if available (e.g. reverb or not, delay, phaser, distorsion...)

        Args:
            kwargs: to be passed to the MultimodalSoundsDataset constructor
        """
        super().__init__(**kwargs)
        # We DO NOT exclude audios which have no instrument (all items have at least an "instrument_source")

        #  - - - Check labels consistency - - -
        # Instrument source: all items must have a label (and all labels must be found at least once)
        assert np.all(self.df['instrument_source'].isin(self.instrument_source_labels))
        for instr_source_label in self.instrument_source_labels:
            sub_df = self.df[self.df['instrument_source'] == instr_source_label]
            assert len(sub_df) > 0, f"{instr_source_label=} has no corresponding item in the dataset"
        # Instrument family: all values must be in available labels (expect the None family, some items do not
        #  have a clearly defined instrument),  and all available labels must have a corresponding item in the dataset
        all_observed_families = set(self.df['instrument_family'].values)
        all_observed_families = sorted([f for f in all_observed_families if f is not None])
        assert all_observed_families == sorted(self.instrument_family_labels), f"{all_observed_families=} is not {sorted(self.instrument_family_labels)=}"
        # FX: same as instrument families
        all_observed_fx = sorted(list(set(sum(self.df['fx'].values, []))))
        assert all_observed_fx == sorted(self.fx_labels), f"{all_observed_fx=} is not {sorted(self.fx_labels)=}"

    @property
    def id2label(self):
        return {
            # "Instrument source" (see NSynth)
            0: 'acoustic', 1: 'electronic', 2: 'synthetic',
            # Instrument families (arbitrarily merged from NSynth and Surge datasets)
            3: 'atmosphere', 4: 'bass', 5: 'brass', 6: 'effects', 7: 'guitar', 8: 'lead', 9: 'mallet', 10: 'organ',
            11: 'pad', 12: 'percussion', 13: 'piano', 14: 'pluck', 15: 'strings', 16: 'vocal', 17: 'wind',
            # FX... (also arbitrary groups)
            18: 'chorus', 19: 'delay', 20: 'distortion', 21: 'flanger', 22: 'frequency shifter', 23: 'phaser',
            24: 'resonator', 25: 'reverb', 26: 'rotary speaker',
        }

    @property
    def instrument_source_ids(self):
        """ IDs corresponding to labels describing an instrument source """
        return list(range(0, 3))

    @property
    def instrument_source_labels(self):
        return [self.id2label[i] for i in self.instrument_source_ids]

    @property
    def instrument_family_ids(self):
        """ IDs corresponding to labels describing an instrument family """
        return list(range(3, 18))

    @property
    def instrument_family_labels(self):
        return [self.id2label[i] for i in self.instrument_family_ids]

    @property
    def fx_ids(self):
        """ IDs corresponding to labels describing an audio effect """
        return list(range(18, 27))

    @property
    def fx_labels(self):
        return [self.id2label[i] for i in self.fx_ids]

    @property
    def label2id(self):
        return {label: idx for idx, label in self.id2label.items()}

    @property
    def num_labels(self):
        return len(self.id2label)

    def __str__(self):
        return (f"{super().__str__()}\n|_ Available labels:"
                f"\n    |_ {self.instrument_source_labels=}"
                f"\n    |_ {self.instrument_family_labels=}"
                f"\n    |_ {self.fx_labels=}")

    # - - - - - - - - - -       PyTorch Dataset and Dataloader features       - - - - - - - - - -

    def __getitem__(self, idx):
        item_idx, data_source_id, audio = super().__getitem__(idx)
        item_row = self.df.iloc[idx]
        targets = torch.zeros(self.num_labels)
        # Assign target probabilities
        targets[self.label2id[item_row['instrument_source']]] = 1.0
        # A single instrument family is currently supported (could be more in the future...)
        if item_row['instrument_family'] is not None:
            targets[self.label2id[item_row['instrument_family']]] = 1.0
        # Zero, one or Multiple FX
        for fx in item_row['fx']:
            targets[self.label2id[fx]] = 1.0
        return item_idx, data_source_id, audio, targets


class ContrastiveInstrumentDataset(MultimodalSoundsDataset):
    def __init__(self, hard_negatives_ratio=0.0, **kwargs):
        """
        Dataset for contrastive training, which can build Dataloaders for easy retrieval of positive
        (same instrument) and negative (other instrument) items

        :param kwargs: to be passed to the MultimodalSoundsDataset constructor
        """
        super().__init__(**kwargs)
        self.hard_negatives_ratio = hard_negatives_ratio
        if self.hard_negatives_ratio > 0.0:
            assert self.split == 'train', "Hard negative mining is only available for the training set"
        # Identify items which have no other positive example - can't be used to make pairs...
        self.df.insert(3, 'instrument_n_audios', -1)
        # Build some instrument *indexes* now, much faster to retrieve later. These indices should be used
        # by this class only (or the related sampler)
        instrument_UID_to_i = {uid: i for i, uid in enumerate(sorted(self.df['instrument_UID'].unique()))}
        self.df.insert(3, '_instrument_i', self.df['instrument_UID'].apply(lambda x: instrument_UID_to_i[x]))
        for i in range(len(instrument_UID_to_i)):
            instrument_mask = (self.df['_instrument_i'] == i)
            self.df.loc[instrument_mask, 'instrument_n_audios'] = int(instrument_mask.sum())
        self.df = self.df[self.df['instrument_n_audios'] >= 2]

    def __getitem__(self, idx):
        item_idx, data_source_id, audio = super().__getitem__(idx)
        instrument_i = self.df.iloc[item_idx]['_instrument_i']
        return item_idx, audio, instrument_i  # instrument_i should not be used... The dataloader will change this

    def get_dataloader(self, batch_size: int, seed=0, num_workers=0) -> torch.utils.data.DataLoader:
        assert self.pad is not None, "Batching will not properly work without padding audio waveforms"

        if self.hard_negatives_ratio > 0.0:  # Only available for the training dataset
            sampler = PairedRandomSampler(self, batch_size, hard_negatives_ratio=self.hard_negatives_ratio)
        else:
            sampler = PrecomputedPairedRandomSampler(self, batch_size, keep_seed=(self.split != 'train'))

        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
            collate_fn=self._dataloader_collate_fn,
        )
        return dataloader

    @staticmethod
    def _dataloader_collate_fn(batch):
        items_indices, audios, instruments_i = zip(*batch)
        current_group_i, instr_i_to_group_i = 0, {}
        for i in instruments_i:
            if i not in instr_i_to_group_i.keys():
                instr_i_to_group_i[i] = current_group_i
                current_group_i += 1
        groups_i = [instr_i_to_group_i[i] for i in instruments_i]
        return (torch.tensor(items_indices, dtype=torch.int64),
                torch.vstack(audios),
                torch.tensor(groups_i, dtype=torch.int32))


class PrecomputedPairedRandomSampler(torch.utils.data.Sampler):
    def __init__(self, data_source: ContrastiveInstrumentDataset, batch_size: int, seed=20250623, keep_seed=False):
        """
        A random sampler that will yield pairs of items from the dataset, tries to ensure to that all instruments
        and all audios can be found in an epoch. However, this is not 100% guaranteed.
        All minibatches are pre-computed when a new epoch starts.

        This sampler does not provide any functionality for semi-hard or hard negative mining based on
        instrument category.

        The last incomplete minibatch will always be dropped.

        :param data_source: TODO doc
        :param batch_size: TODO doc
        :param seed:
        :param keep_seed: If True, the same seed will be used for each epoch (use for valid and test subsets)
        """
        super().__init__()
        self.data_source, self.batch_size = data_source, batch_size
        assert self.batch_size % 2 == 0, "Batch size must be even, in order to make pairs in each batch"
        self.current_seed, self.keep_seed = seed, keep_seed
        # Make a copy of the dataset's dataframe, but only keep the necessary cols
        self.df = self.data_source.df[[
            'split', 'instrument_UID', '_instrument_i', 'instrument_n_audios', 'is_augmented',
            'midi_pitch', 'midi_velocity', 'midi_artist', 'midi_track',
        ]].copy()
        self.df['dataset_i'] = range(len(self.df))
        self.epoch_df_backup: Optional[pd.DataFrame] = None

    @property
    def n_minibatches(self):
        # Each audio presented twice ON AVERAGE during one epoch
        return len(self.data_source) // (self.batch_size // 2)

    def __len__(self):
        return self.n_minibatches * self.batch_size

    def __iter__(self):
        # Initialize the iterator: all minibatches are planned from the beginning (quite CPU intensive...)
        if self.keep_seed and self.epoch_df_backup is not None:
            epoch_df = self.epoch_df_backup.copy()  #    but Don't recompute epoch_df if the seed does not change...
        else:
            epoch_df = self._generate_epoch_df()
            self.epoch_df_backup = epoch_df.copy()
        assert len(epoch_df) == len(self), "Planned len should be the same as actual length after pos/neg mining"
        # Now yield (return) all indices one by one
        for i in range(len(self)):
            yield epoch_df.iloc[i]['dataset_i']

    def _generate_epoch_df(self):
        n_instr_per_batch = self.batch_size // 2
        if not self.keep_seed:
            self.current_seed += 1

        indices = list(range(len(self.data_source)))
        rng = np.random.default_rng(seed=self.current_seed)
        rng.shuffle(indices)

        # generate a temporary dataframe for this epoch
        epoch_df = self.df.copy().iloc[indices]
        instrument_indices = sorted(list(epoch_df['_instrument_i'].unique()))  # NOT the UIDs; those are "local" indices
        epoch_df.insert(0, 'used', False)
        # and split this big DF in minibatches
        minibatches_dfs = []
        for minibatch_i in range(self.n_minibatches):
            minibatches_dfs.append(
                epoch_df.iloc[n_instr_per_batch * minibatch_i:n_instr_per_batch * (minibatch_i+1)].copy())
            epoch_df.loc[minibatches_dfs[-1].index, 'used'] = True

        # - - - - Check that all instruments are different in all minibatches (ensure proper "negative mining") - - - -
        # Shuffling is almost NEVER properly done... So we'll just analyze all minibatches by hand, replace
        #   the instrument found twice (or more) by a random audio and its associated instrument_UID
        for minibatch_i, minibatch_df in enumerate(minibatches_dfs):
            if len(minibatch_df['_instrument_i'].unique()) != n_instr_per_batch:
                #  Row-by-row processing:
                #    Check if the instrument can be found in previous rows, and if it does, replace it
                for row_i in range(len(minibatch_df)):
                    previous_rows, row = minibatch_df.iloc[0:row_i], minibatch_df.iloc[row_i]
                    if row['_instrument_i'] in previous_rows['_instrument_i'].values:
                        # Just re-use some random audio
                        new_row = row
                        other_instruments_i = minibatch_df['_instrument_i'].unique()
                        while new_row['_instrument_i'] in other_instruments_i:
                            new_row = epoch_df.iloc[rng.integers(0, len(epoch_df))]
                        # Make this the minibatch's current (i-th) row
                        epoch_df.loc[new_row.name, 'used'] = True
                        minibatch_df.loc[row.name] = new_row
                        minibatch_df.rename(index={row.name: new_row.name}, inplace=True)
                        epoch_df.loc[row.name, 'used'] = False
        # Negative mining sanity check: no duplicate instruments in each minibatch?
        assert np.all([len(_df['instrument_UID'].unique()) == len(_df) for _df in minibatches_dfs])

        # - - - - - - - -     Build the pairs of sounds from each instrument ("positive mining")    - - - - - - - -
        # First, for computational speed, we absolutely need to pre-load a mapping from instrument_i to all
        # corresponding dataframe rows (we'll store the entire dataframe, we have more than enough RAM for this...)
        instrument_i_to_sub_df = {i: epoch_df[epoch_df['_instrument_i'] == i] for i in instrument_indices}
        # Random sampling with replacement
        for minibatch_i, minibatch_df in enumerate(minibatches_dfs):
            positives_df = list()
            for _idx, row in minibatch_df.iterrows():
                # The following operations are very expensive: better not to search the (big) dataframes,
                #        but to locate indices in advance
                instr_df = instrument_i_to_sub_df[row['_instrument_i']]
                positive_row = row
                while positive_row['dataset_i'] == row['dataset_i']:
                    positive_row = instr_df.iloc[rng.integers(0, len(instr_df))]
                positives_df.append(positive_row)
            positives_df = pd.DataFrame(positives_df, index=(1 + 2 * np.arange(len(positives_df))))
            positives_df['group_index'] = np.arange(len(positives_df))
            minibatch_df = minibatch_df.set_index(2 * np.arange(len(positives_df)))
            minibatch_df['group_index'] = np.arange(len(minibatch_df))
            assert len(positives_df) == len(minibatch_df) == (self.batch_size // 2)
            # Merge
            minibatches_dfs[minibatch_i] = pd.concat([minibatch_df, positives_df]).sort_index()
            minibatches_dfs[minibatch_i]['minibatch_index'] = minibatch_i
            assert len(minibatches_dfs[minibatch_i]) == self.batch_size

        #  Rebuild epoch_df from the minibatches (the unused--the last ditched batch-- won't be part of it...)
        epoch_df = pd.concat(minibatches_dfs, ignore_index=True).drop(columns=['used'])
        return epoch_df


class PairedRandomSampler(torch.utils.data.Sampler):
    def __init__(self,
                 data_source: ContrastiveInstrumentDataset, batch_size: int, seed=20250824,
                 hard_negatives_ratio=0.0, hard_negatives_category='instrument_family',
                 ):
        """
        TODO DOC

        The last incomplete minibatch will always be dropped.

        :param data_source: TODO doc
        :param batch_size: TODO doc
        :param seed:
        :param hard_negatives_ratio: TODO doc
        :param hard_negatives_category: Type of labels used for sampling semi-hard negatives belonging to the same class.
        """
        super().__init__()
        self.rng = np.random.default_rng(seed=seed)

        self.data_source, self.batch_size = data_source, batch_size
        self.hard_negatives_category, self.hard_negatives_ratio = hard_negatives_category, hard_negatives_ratio
        assert self.batch_size % 2 == 0, "Batch size must be even, in order to make pairs in each batch"
        # Number of hard negative and random sounds per minibatch
        self.n_hard_per_minibatch = self.batch_size * self.hard_negatives_ratio
        self.n_hard_per_minibatch = int(2 * round(self.n_hard_per_minibatch / 2))  # Round to the closest even integer
        self.n_random_per_minibatch = self.batch_size - self.n_hard_per_minibatch

        self.available_hard_negatives_categories = ('instrument_family', 'midi_instrument_group')
        assert hard_negatives_category in self.available_hard_negatives_categories
        # Make a copy of the dataset's dataframe, but only keep the necessary cols
        assert self.hard_negatives_category in self.data_source.df.columns
        self.df = self.data_source.df[[
            'instrument_UID', '_instrument_i', self.hard_negatives_category,
            'split', 'instrument_n_audios', 'is_augmented', 'midi_pitch', 'midi_velocity', 'midi_artist', 'midi_track',
        ]].copy()
        self.df['dataset_i'] = range(len(self.df))
        self.df.insert(0, 'n_epoch_draws', 0)

        self.n_minibatches = len(self.df) // self.batch_size

    def __len__(self):
        return self.n_minibatches * self.batch_size

    def __iter__(self):
        # When epoch starts: reinit usage counters
        self.df['n_epoch_draws'] = 0

        # Now yield (return) all indices one by one - on a "per-minibatch" basis
        for minibatch_i in range(self.n_minibatches):
            minibatch_df = list()

            # 1) Sample the hard negatives (positive pairs that are hard negatives to each other):
            #    1a) Randomly retrieve an instrument family/group (we don't care about the sound/instrument yet)
            #        and the associated sub-df - careful of the items with a None family! (cannot be used for mining)
            family = None
            while family is None or (isinstance(family, str) and family == ''):
               family = self.df.iloc[self.rng.integers(0, len(self.df))][self.hard_negatives_category]
            family_mask = (self.df[self.hard_negatives_category] == family)
            sub_df = self.df[family_mask]
            #    1b) Using this smaller DF: using normalized the probability distribution, sample the proper number of
            #        pairs of sounds for this family/group (also updates the usage counters 'n_epoch_draws')
            minibatch_df += self._sample_instrument_pairs(sub_df, self.n_hard_per_minibatch // 2)

            # 2) Randomly sample the others. Some hard negatives can be there too! In practice, it will often happen
            #     for big families such as bass or keys
            other_families_sub_df = self.df[~family_mask]
            minibatch_df += self._sample_instrument_pairs(other_families_sub_df, self.n_random_per_minibatch // 2)

            # 3) Sanity check(s)
            minibatch_df = pd.DataFrame(minibatch_df)
            assert np.all(minibatch_df['instrument_UID'].value_counts().values == 2), \
                "Each instrument must be found exactly twice in a minibatch"

            #  Finally, yield the indices one-by-one
            for row in minibatch_df.itertuples():
                yield row.dataset_i

    def _sample_instrument_pairs(self, sub_df: pd.DataFrame, n_pairs: int):
        sampled_rows = list()
        for _ in range(n_pairs):
            # Check that enough samples remain in sub_df...
            if len(sub_df) == 0:
                raise NotImplementedError("A break should happend, but the info should be forwarded to the caller")

            sampling_probs = np.exp(-1.0 * sub_df['n_epoch_draws'])
            sampling_probs = sampling_probs / sampling_probs.sum()
            sampled_df_index = self.rng.choice(sub_df.index, p=sampling_probs)
            anchor_row = sub_df.loc[sampled_df_index]
            sampled_instrument_mask = (sub_df['_instrument_i'] == anchor_row['_instrument_i'])
            # Get the positive item
            sampled_instrument_df = sub_df[sampled_instrument_mask]
            sampled_instrument_df = sampled_instrument_df[sampled_instrument_df.index != anchor_row.name]
            positive_row = sampled_instrument_df.iloc[self.rng.integers(0, len(sampled_instrument_df))]
            # store both (anchor and positive) and update their usage counters in the main DF
            sampled_rows += [anchor_row, positive_row]
            self.df.loc[[anchor_row.name, positive_row.name], 'n_epoch_draws'] += 1
            # Mask the sounds for the last sampled instrument
            sub_df = sub_df[~sampled_instrument_mask]
        return sampled_rows


if __name__ == "__main__":
    # Small tests
    from tqdm import tqdm

    for _split in []:  # ['all', 'train', 'valid', 'test']:
        _dataset = MultimodalSoundsDataset(sub_datasets=('nsynth', 'surge',), use_small_dataset=True, split=_split)
        print(_dataset)
        _item = _dataset[4000]

    for _split in ['test', 'valid', 'train', 'all']:
        #_dataset = MIDIInstrumentGroupClassificationDataset(use_small_dataset=True, split=_split, pad='right')
        #_dataset = MultiLabelDataset(use_small_dataset=True, split=_split, pad='right')
        _dataset = ContrastiveInstrumentDataset(use_small_dataset=True, split=_split, pad='right', target_sr=16000)
        print(_dataset)
        _item = _dataset[4000]
        if _split != 'all':
            _dl = _dataset.get_dataloader(32, num_workers=16)
            for _minibatch in tqdm(_dl, total=len(_dl), desc="Fetching all minibatches"):
                _df_row_indices, _audios, _instrument_groups = _minibatch
            pass

    pass
