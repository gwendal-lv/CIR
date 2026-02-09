# TODO WARNING: Don't modify this file (it will be automatically overwritten)
""" Utility classes and constants for an easy use of the dataset after generation """
import pandas as pd

REFERENCE_MIDI_NOTE = {'pitch': 57, 'velocity': 100}  # A3 (220Hz) note, velocity 100

SURGE_FAMILY_TO_MERGED_FAMILY = {
    'Basses': 'bass',
    'Brass': 'brass',
    'FX': "effects",
    'Keys': 'piano',
    'Leads': 'lead',
    'Pads': 'pad',
    'Percussion': 'percussion',
    'Plucks': 'pluck',
    'Polysynths': None,  # FIXME could be either synth lead or pad...
    'Sequences': None,  # Excluded, not to mix timbre with rhythm / MIDI sequences... (same for e.g. arps)
    'Winds': 'wind',
    'Drums': 'percussion',
    'Guitars': 'guitar',
    'Organs': 'organ',
    'Synths': 'lead',
    'Rhythms': None,
    'Vox': 'vocal',
    'Arps': None,
    'Atmospheres': 'atmosphere',
    'Ambiance': 'atmosphere',
    'Mallets': 'mallet',
    'Soundscapes': 'atmosphere',
    'Strings': 'strings',
    'Voices': 'vocal',
    'Bells': 'mallet',  # Bells should be their own category... But mallet is MUCH bigger
    'Drones': 'effects',
}

# Shouldn't be used, lots of things to be fixed here...
SURGE_FAMILY_TO_MIDI_TIMBRE_GROUP = {
    'Basses': 'Bass',
    'Brass': 'Brass',
    'FX': "Synth Effects",
    'Keys': 'Piano',
    'Leads': 'Synth Lead',
    'Pads': 'Synth Pad',
    'Percussion': 'Chromatic Percussion',
    'Plucks': 'Guitar',
    'Polysynths': None,  # FIXME could be either synth lead or pad...
    'Sequences': None,  # TODO Where to put this... (check also other rythmic labels)
    'Winds': 'Reed',  # FIXME maybe Pipe also, depends on the preset...
    'Drums': 'Chromatic Percussion',
    'Guitars': 'Guitar',
    'Organs': 'Organ',
    'Synths': 'Synth Lead',
    'Rhythms': None,
    'Vox': 'Voice',
    'Arps': None,
    'Atmospheres': 'Synth Effects',
    'Ambiance': 'Synth Effects',
    'Mallets': 'Chromatic Percussion',
    'Soundscapes': 'Synth Effects',
    'Strings': 'Strings',
    'Voices': 'Voice',
    'Bells': 'Chromatic Percussion',
    'Drones': 'Synth Effects',
}

# Used for mapping MIDI tracks and distributions of notes to Surge patches (so we get consistent, audible audio outs)
#   Can't be considered as proper categories for timbre
SURGE_FAMILY_TO_MIDI_GROUP = {
    'Basses': 'Bass',
    'Brass': 'Brass',
    'FX': 'Piano',  # TODO check maybe...
    'Keys': 'Piano',
    'Leads': 'Synth Lead',
    'Pads': 'Synth Pad',
    'Percussion': 'Chromatic Percussion',
    'Plucks': 'Guitar',
    'Polysynths': 'Piano',  # TODO check with examples...
    'Sequences': 'Synth Pad',  # For longer notes...
    'Winds': 'Reed',  # Could have been "Pipe" also...
    'Drums': 'Chromatic Percussion',
    'Guitars': 'Guitar',
    'Organs': 'Organ',
    'Synths': 'Synth Lead',
    'Rhythms': 'Synth Pad',  # For longer notes...
    'Vox': 'Voice',
    'Arps': 'Synth Pad',
    # Maybe strings for some of those ? (MIDI strings have the longuest notes...) BUT Synth Pad has a better pitch distribution...
    'Atmospheres': 'Synth Pad',
    'Ambiance': 'Synth Pad',
    'Mallets': 'Chromatic Percussion',
    'Soundscapes': 'Synth Pad',
    'Strings': 'Strings',
    'Voices': 'Voice',
    'Bells': 'Chromatic Percussion',
    'Drones': 'Synth Pad',
}

# Effects documentation: https://surge-synthesizer.github.io/manual-xt/#effects
AVAILABLE_FX = pd.DataFrame([
    # ---    Effects found in surgepy.constants.__dict__    ---
    {'name': 'Off', 'param_value': 0, 'category': 'Off'},
    {'name': 'Delay', 'param_value': 1, 'category': 'Time & Space'},
    {'name': 'Reverb 1', 'param_value': 2, 'category': 'Time & Space'},
    {'name': 'Phaser', 'param_value': 3, 'category': 'Modulation'},
    {'name': 'Rotary Speaker', 'param_value': 4, 'category': 'Modulation'},
    {'name': 'Distortion', 'param_value': 5, 'category': 'Distortion'},
    {'name': 'EQ', 'param_value': 6, 'category': 'Filtering'},
    {'name': 'Frequency Shifter', 'param_value': 7, 'category': 'Mangling'},
    {'name': 'Conditioner', 'param_value': 8, 'category': 'Multieffects'},
    {'name': 'Chorus', 'param_value': 9, 'category': 'Modulation'},
    {'name': 'Vocoder', 'param_value': 10, 'category': 'Mangling'},
    {'name': 'Reverb 2', 'param_value': 11, 'category': 'Time & Space'},
    {'name': 'Flanger', 'param_value': 12, 'category': 'Modulation'},
    {'name': 'Ring Modulation', 'param_value': 13, 'category': 'Mangling'},
    {'name': 'Airwindows', 'param_value': 14, 'category': 'Multieffects'},  # 56 effects inside...
    {'name': 'Neuron', 'param_value': 15, 'category': 'Distortion'},

    # ---    Effects identified by looking at the plugin directly (for specific presets using an effect ID)    ---
    {'name': 'Graphic EQ', 'param_value': 16, 'category': 'Filtering'},
    {'name': 'Resonator', 'param_value': 17, 'category': 'Filtering'},
    {'name': 'CHOW', 'param_value': 18, 'category': 'Distortion'},  # effect quite unclear...
    {'name': 'Exciter', 'param_value': 19, 'category': 'Filtering'},
    {'name': 'Ensemble', 'param_value': 20, 'category': 'Modulation'},
    {'name': 'Combulator', 'param_value': 21, 'category': 'Mangling'},
    {'name': 'Nimbus', 'param_value': 22, 'category': 'Mangling'},  # granular texture effect
    {'name': 'Tape', 'param_value': 23, 'category': 'Distortion'},
    {'name': 'Treemonster', 'param_value': 24, 'category': 'Mangling'},  # complex FX with pitch tracking and ring mod
    {'name': 'Waveshaper', 'param_value': 25, 'category': 'Distortion'},
    # Different filters (low- and high-cuts) for the mid and side components (L-R // M-S transforms)
    {'name': 'Mid-Side Tool', 'param_value': 26, 'category': 'Multieffects'},
    {'name': 'Spring Reverb', 'param_value': 27, 'category': 'Time & Space'},
    {'name': 'Bonsai', 'param_value': 28, 'category': 'Distortion'},
    {'name': 'Audio Input', 'param_value': 29, 'category': 'Multieffects'},
])

# Mapping to a reduced set of known and fully-defined FX
SIMPLIFIED_FX_MAP = {
    'Off': None,
    'Delay': 'delay',
    'Reverb 1': 'reverb',
    'Phaser': 'phaser',
    'Rotary Speaker': 'rotary speaker',
    'Distortion': 'distortion',
    'EQ': None,
    'Frequency Shifter': 'frequency shifter',
    'Conditioner': None,  # EQ
    'Chorus': 'chorus',
    'Vocoder': None,
    'Reverb 2': 'reverb',
    'Flanger': 'flanger',
    'Ring Modulation': None,  # Quite specific and very few examples in the dataset...
    'Airwindows': None,
    'Neuron': None,
    'Graphic EQ': None,
    'Resonator': 'resonator',
    'CHOW': 'distortion',
    'Exciter': 'resonator',  # Harmonic exciter. Almost no item: merged with resonator
    'Ensemble': 'chorus',  # only a few patches use it... and is actually chorus-based
    'Combulator': 'flanger',  # Very few of those... Merged with flanger
    'Nimbus': None,
    'Tape': 'distortion',  # TODO Maybe a specific tape distortion?
    'Treemonster': None,
    'Waveshaper': 'distortion',  # Very few of those... merged w/ disto
    'Mid-Side Tool': None,
    'Spring Reverb': 'reverb',
    'Bonsai': 'distortion',
    'Audio Input': None,
}