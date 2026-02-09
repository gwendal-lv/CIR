# TODO WARNING: Don't modify this file (it will be automatically overwritten)
"""
Classes for using dataset of MIDI notes and distributions of MIDI pitch/velocity/duration.
Datasets themselves must be downloaded separately.

Also provides lists of MIDI instrument names and families
"""
import json
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd


MIDI_INSTRUMENTS = {
    # Piano
    0: "Acoustic Grand Piano", 1: "Bright Acoustic Piano", 2: "Electric Grand Piano", 3: "Honky-tonk Piano",
    4: "Electric Piano 1", 5: "Electric Piano 2", 6: "Harpsichord", 7: "Clavinet",
    # Chromatic Percussion
    8: "Celesta", 9: "Glockenspiel", 10: "Music Box", 11: "Vibraphone", 12: "Marimba",
    13: "Xylophone", 14: "Tubular Bells", 15: "Dulcimer",
    # Organ
    16: "Drawbar Organ", 17: "Percussive Organ", 18: "Rock Organ", 19: "Church Organ",
    20: "Reed Organ", 21: "Accordion", 22: "Harmonica", 23: "Tango Accordion",
    # Guitar
    24: "Acoustic Guitar (nylon)", 25: "Acoustic Guitar (steel)", 26: "Electric Guitar (jazz)", 27: "Electric Guitar (clean)",
    28: "Electric Guitar (muted)", 29: "Overdriven Guitar", 30: "Distortion Guitar", 31: "Guitar Harmonics",
    # Bass
    32: "Acoustic Bass", 33: "Electric Bass (finger)", 34: "Electric Bass (pick)", 35: "Fretless Bass",
    36: "Slap Bass 1", 37: "Slap Bass 2", 38: "Synth Bass 1", 39: "Synth Bass 2",
    # Strings - including Timpani in the MIDI standard! https://midi.org/general-midi Makes no sense...
    40: "Violin", 41: "Viola", 42: "Cello", 43: "Contrabass", 44: "Tremolo Strings",
    45: "Pizzicato Strings", 46: "Orchestral Harp", 47: "Timpani",
    # Ensemble
    48: "String Ensemble 1", 49: "String Ensemble 2", 50: "Synth Strings 1", 51: "Synth Strings 2",
    52: "Choir Aahs", 53: "Voice Oohs", 54: "Synth Voice", 55: "Orchestra Hit",
    # Brass
    56: "Trumpet", 57: "Trombone", 58: "Tuba", 59: "Muted Trumpet", 60: "French Horn",
    61: "Brass Section", 62: "Synth Brass 1", 63: "Synth Brass 2",
    # Reed
    64: "Soprano Sax", 65: "Alto Sax", 66: "Tenor Sax", 67: "Baritone Sax", 68: "Oboe",
    69: "English Horn", 70: "Bassoon", 71: "Clarinet",
    # Pipe
    72: "Piccolo", 73: "Flute", 74: "Recorder", 75: "Pan Flute", 76: "Blown Bottle",
    77: "Shakuhachi", 78: "Whistle", 79: "Ocarina",
    # Synth Lead
    80: "Lead 1 (square)", 81: "Lead 2 (sawtooth)", 82: "Lead 3 (calliope)", 83: "Lead 4 (chiff)",
    84: "Lead 5 (charang)", 85: "Lead 6 (voice)", 86: "Lead 7 (fifths)", 87: "Lead 8 (bass + lead)",
    # Synth Pad
    88: "Pad 1 (new age)", 89: "Pad 2 (warm)", 90: "Pad 3 (polysynth)", 91: "Pad 4 (choir)",
    92: "Pad 5 (bowed)", 93: "Pad 6 (metallic)", 94: "Pad 7 (halo)", 95: "Pad 8 (sweep)",
    # Synth Effects
    96: "FX 1 (rain)", 97: "FX 2 (soundtrack)", 98: "FX 3 (crystal)", 99: "FX 4 (atmosphere)",
    100: "FX 5 (brightness)", 101: "FX 6 (goblins)", 102: "FX 7 (echoes)", 103: "FX 8 (sci-fi)",
    # Ethnic
    104: "Sitar", 105: "Banjo", 106: "Shamisen", 107: "Koto", 108: "Kalimba",
    109: "Bagpipe", 110: "Fiddle", 111: "Shanai",
    # Percussive
    112: "Tinkle Bell", 113: "Agogo", 114: "Steel Drums", 115: "Woodblock", 116: "Taiko Drum",
    117: "Melodic Tom", 118: "Synth Drum",
    # Sound Effects
    119: "Reverse Cymbal", 120: "Guitar Fret Noise", 121: "Breath Noise", 122: "Seashore",
    123: "Bird Tweet", 124: "Telephone Ring", 125: "Helicopter", 126: "Applause", 127: "Gunshot",
    # The 129th program is used in the Slakh dataset to designate a Drumkit
    128: 'Drums',
}

# From the General MIDI specification - seems outdated, not used in Slakh for instance... http://www.slakh.com/#analysis
__MIDI_INSTRUMENT_GROUPS = {
    "Piano": list(range(0, 8)),
    "Chromatic Percussion": list(range(8, 16)),
    "Organ": list(range(16, 24)),
    "Guitar": list(range(24, 32)),
    "Bass": list(range(32, 40)),
    "Strings": list(range(40, 48)),
    "Ensemble": list(range(48, 56)),
    "Brass": list(range(56, 64)),
    "Reed": list(range(64, 72)),
    "Pipe": list(range(72, 80)),
    "Synth Lead": list(range(80, 88)),
    "Synth Pad": list(range(88, 96)),
    "Synth Effects": list(range(96, 104)),
    "Ethnic": list(range(104, 112)),
    "Percussive": list(range(112, 120)),
    "Sound Effects": list(range(120, 128))
}

# Slightly customized MIDI instrument groups
MIDI_INSTRUMENT_GROUPS = {
    "Piano": list(range(0, 8)),
    "Chromatic Percussion": list(range(8, 16)) + [47, 108],  # + timpani, kalimba
    "Organ": list(range(16, 24)),
    "Guitar": list(range(24, 32)) + [104, 105, 106, 107],  # + sitar, banjo, shamisen, koto (koto is borderline...)
    "Bass": list(range(32, 40)),
    "Strings": list(range(40, 47)) + list(range(48, 52)) + [110],  # + ensemble and synth strings, fiddle
    # "Ensemble": list(range(48, 56)),  # Entirely removed... string ensembles moved to string, voices as a new group
    "Voice": list(range(52, 55)),
    "Brass": list(range(56, 64)),
    "Reed": list(range(64, 72)) + [109, 111],  # + bagpipes, shanai
    "Pipe": list(range(72, 80)),
    "Synth Lead": list(range(80, 88)),
    "Synth Pad": list(range(88, 96)),
    "Synth Effects": list(range(96, 104)),
    # "Ethnic": list(range(104, 112)),  # Weird category... removed
    "Percussion": [55] + list(range(112, 120)),  # Renamed from "Percussive" for coherence
    "Sound Effects": list(range(120, 128)),
    "Drum Kit": [128],
}

MIDI_INSTRUMENT_TO_GROUP = [None, ] * 129
for group, instruments in MIDI_INSTRUMENT_GROUPS.items():
    for midi_program_value in instruments:
        MIDI_INSTRUMENT_TO_GROUP[midi_program_value] = group


class SlakhMIDI:
    def __init__(self, data_dir: Union[str, Path] = None, exclude_unnamed=True, exclude_duplicates=True):
        """
        MIDI-only (no audio) dataset of MIDI notes. See midi.ipynb for more information.

        :param data_dir: Folder where all audio and MIDI dataset are stored. If not provided, will use the
            './data/' folder. data_dir must contain the 'slakh2100_no_audio/' subdirectory.
        """
        slakh_dir_name = 'slakh2100_no_audio'
        if data_dir is None:
            self.path = Path(__file__).parent / 'data' / slakh_dir_name
        else:
            assert Path(data_dir).is_dir()
            self.path = Path(data_dir) / slakh_dir_name
        assert self.path.exists(), f"{self.path} cannot be found and should contain the Slakh MIDI-only dataset"

        # load the main DF with general info about the MIDI tracks available
        self._full_df = pd.read_csv(self.path / 'slakh_no_audio.csv').fillna('')
        # Sub-dataframe with MIDI songs included (may exclude duplicates and/or non-named tracks...)
        self.df = self._full_df.copy()
        if exclude_duplicates:
            self.df = self.df[(~self.df['lmd_name_duplicate']) & (self.df['split'] != 'omitted')]
        if exclude_unnamed:
            self.df = self.df[self.df['lmd_track'] != '']

        # Pre-load available instruments for each MIDI song
        # TODO ensure that each track has enough notes...??? reading all files now would be very long
        self.instruments_df = pd.read_csv(self.path / 'midi_instruments.csv')
        self.instruments_df = self.instruments_df[self.instruments_df['slakh_index'].isin(self.df.index)]

        # Load distributions of MIDI notes
        with open(self.path / 'notes_distributions.json', 'r') as f:
            self.notes_distributions = json.load(f)

    def get_random_note(self, instrument_group: str, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        # sample pitch, velocity and duration independently
        note_info = {'pitch': -1, 'velocity': -1, 'duration_s': -1.0}
        for k in ['pitch', 'velocity']:
            note_info[k] = rng.choice(
                self.notes_distributions[k][instrument_group]['values'],
                p=self.notes_distributions[k][instrument_group]['probabilities']
            )
        distribution = self.notes_distributions['log_duration_s'][instrument_group]
        bin_index = rng.choice(np.arange(len(distribution['probabilities'])), p=distribution['probabilities'])
        bin_lower, bin_upper = distribution['bin_edges'][bin_index], distribution['bin_edges'][bin_index+1]
        note_info['duration_s'] = 10 ** rng.uniform(bin_lower, bin_upper)

        return note_info


    def __str__(self):
        return f'SlakhMIDI dataset with {len(self.df)} songs and {len(self.instruments_df)} instrument tracks'


if __name__ == "__main__":
    # Basic example
    slakh_midi_dataset = SlakhMIDI()
    note = slakh_midi_dataset.get_random_note('Bass')
    pass