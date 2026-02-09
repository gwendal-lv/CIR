# TODO WARNING: Don't modify this file (it will be automatically overwritten)
import copy
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
import torchaudio
from matplotlib.style.core import available

# See MIDI instruments groups defined in midi.py
INSTRUMENT_FAMILY_TO_MIDI_GROUP = {
    'bass': "Bass",
    'brass': "Brass",
    'flute':  "Pipe",
    'guitar':  "Guitar",
    'keyboard': "Piano",
    'mallet': "Chromatic Percussion",
    'organ': "Organ",
    'reed': "Reed",
    'string': "Strings",
    'synth_lead': "Synth Lead",
    'vocal': "Voice",
}

# Can be used to convert NSynth's families to a new set of families shared across all datasets in this repo
INSTRUMENT_FAMILY_TO_MERGED_FAMILY = {
    'bass': "bass",
    'brass': "brass",
    'flute':  "wind",  # flute and reed are not big families ; merged together
    'guitar':  "guitar",
    'keyboard': "piano",
    'mallet': "mallet",
    'organ': "organ",
    'reed': "wind",
    'string': "strings",
    'synth_lead': "lead",  # 'synth' info will be provided in the 'source' label (acoustic, electronic, synthetic)
    'vocal': "vocal",
}

# The list of string representation of all 'note qualities' (e.g. dark, fast_decay, ...). Seem to be
# automatically generated (no documentation about this...) but are often low-quality annotations
# (noisy labels, inconsistent, etc...).
#
# Those qualities might vary depending on the MIDI note (they do not remain constant for all notes of a given
# instrument, e.g. a 'bass' low note can be dark, but a high-pitched note possibly won't be).
AVAILABLE_QUALITIES = [
    'bright', 'dark', 'distortion', 'fast_decay', 'long_release', 'multiphonic', 'nonlinear_env', 'percussive',
    'reverb', 'tempo-synced'
]

# Some qualities can be considered as a simplified FX label (for multi-label classification tasks)
#   Even 'distortion' is excluded, the labelling is VERY inconsistent (e.g. for the same instrument, same velocity,
#       distortion looks randomly related to pitch...)
QUALITY_TO_FX = {
    q: (q if q in ['reverb'] else None) for q in AVAILABLE_QUALITIES
}


class NSynthDataset:
    def __init__(self, data_dir: Union[str, Path] = None, version="0.1", small=False, exclude_extreme_pitches=True):
        """
        :param data_dir: Directory where all datasets can be found (including e.g. nsynth_v0.1, etc...).
            If not provided, will automatically try to use a "data" folder located at this file's level.
        :param version:
        :param small: If True, a smaller version will be used (approximately 10% of the whole dataset)
        :param exclude_extreme_pitches:
        """
        nsynth_dir_name = f"nsynth_v{version}{'_small' if small else ''}"
        if data_dir is None:
            self.path = Path(__file__).parent / 'data' / nsynth_dir_name
        else:
            assert Path(data_dir).is_dir()
            self.path = Path(data_dir) / nsynth_dir_name
        assert self.path.exists(), f"{self.path} cannot be found and should contain the NSynth dataset"
        self._full_df = pd.read_csv(str(self.path / 'examples.csv'))  # With samples that will be eventually excluded
        # Exclusions: weird (mislabeled) instruments, extreme pitches
        self.df = self._full_df[~ self._full_df.instrument.isin(self.get_excluded_instruments())]
        if exclude_extreme_pitches:
            exclusion_mask = np.zeros(len(self.df), dtype=bool)
            for instrument_family in self.available_instrument_families_str:
                pitch_info = self.get_pitch_range(instrument_family)
                pitch_min, pitch_max = pitch_info['min'], pitch_info['max']
                exclusion_mask = exclusion_mask | (
                    (self.df.instrument_family == instrument_family) & (~self.df.pitch.between(pitch_min, pitch_max))
                )
            self.df = self.df[~exclusion_mask]
        # reset the index (the note index can always be used to retrieve a specific note in the original DF)
        self.df = self.df.reset_index(drop=True)

    def __str__(self):
        ratios = {k: 100.0 * np.count_nonzero(self.df.split == k) / len(self.df) for k in ['train', 'valid', 'test']}
        return (f"NSynth dataset with {len(self.df)} notes "
                f"(train: {ratios['train']:.1f}%, valid: {ratios['valid']:.1f}%, test: {ratios['test']:.1f}%)")

    # =================================== Data augmentation / reduction... =======================================

    # TODO data augmentation for text labels...  ??

    # TODO audio data augmentation, slight repitching ?
    #    and/or time stretching? (audios are ALWAYS 3s before release...)

    @staticmethod
    def get_pitch_range(family: str):
        """ Ranges for each instrument family that allow to get sounds that are a better representations
         of that family. E.g., very high-pitched bass sounds, or very low-pitched brass sounds are often
         degenerated synthesized or time-stretched samples.

         Also provides the 'middle' note, which is usually the most representative of a given type of instrument
         """
        return {
            'bass': {'min': 23, 'middle': 40, 'max': 63},
            'brass': {'min': 31, 'middle': 60, 'max': 89},
            # no upper limit: even though same acoustic flutes sound weird, synthetic ones are OK
            'flute': {'min': 35, 'middle': 72, 'max': 108},
            'guitar': {'min': 25, 'middle': 60, 'max': 86},
            # 21, 108 = A0, C8 : most extreme MIDI notes
            'keyboard': {'min': 21, 'middle': 60, 'max': 108},
            # Very high or very low notes do not really sound like a mallet perc...
            'mallet': {'min': 36, 'middle': 60, 'max': 108},
            'organ': {'min': 21, 'middle': 54, 'max': 96},
            'reed': {'min': 31, 'middle': 60, 'max': 96},
            'string': {'min': 24, 'middle': 67, 'max': 96},
            'synth_lead': {'min': 36, 'middle': 60, 'max': 84},
            'vocal': {'min': 38, 'middle': 60, 'max': 86},
        }[family]

    @staticmethod
    def get_excluded_instruments():
        """ Instruments that  or did not seem worth keeping or did not seem properly labeled
         when randomly listening to some items """
        instr = [
            'bass_synthetic_000',  # Weird sound, not a bass synth...
            'bass_synthetic_002'  # Some kind of thin synth (not a bass)
            'guitar_electronic_014', # Some kind of weird perc (more like a broken pan...)
        ]
        # Not 'vocal' but quite usual lead synths...
        instr += [f'vocal_synthetic_{i:03d}' for i in [0, 1, 2, 3, 7, 8, 9, 10, 12, ]]
        return instr

    # =================================== Labels =======================================

    @property
    def available_instrument_sources_str(self):
        """ The list of string representation of all 'instrument sources' (e.g. acoustic, electronic, ...)
        https://magenta.tensorflow.org/datasets/nsynth#note-qualities """
        return ['acoustic', 'electronic', 'synthetic']

    @property
    def available_instrument_families_str(self):
        """ The list of string representation of all NSynth 'instrument families' (e.g. bass, organ, ...) """
        return ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead',
                'vocal']



if __name__ == "__main__":

    from tqdm import tqdm

    ds = NSynthDataset(small=False)  # Will search into ./data/
    ds = NSynthDataset(data_dir="/media/gwendal/Data/Datasets/multimodal_sounds_dataset", small=False)  # Can use an absolute path also
    print(ds)
    # benchmark: try read all the dataset (and measure time...)
    if False:
        for index, row in tqdm(ds.df.iterrows(), total=len(ds.df)):
            y, sr = torchaudio.load(ds.path / row.path)

    ds = NSynthDataset(small=True)
    print(f"Smaller dataset: {ds}")
    for index, row in tqdm(ds.df.iterrows(), total=len(ds.df)):
        y, sr = torchaudio.load(ds.path / row.path)