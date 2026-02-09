# TODO WARNING: Don't modify this file (it will be automatically overwritten)
import ast
import os
from pathlib import Path

import torch.utils.data
import pandas as pd
from natsort import natsorted

try:  # Try/except for compatibility with notebooks...
    from . import nsynth
    from . import surge
except ImportError:
    import nsynth
    import surge


# Statistics about the audio datasets or features extracted from the raw audio
#      (training stats only)
FEATURES_STATISTICS = {
    # AST Mel-spectrograms, computed using the default extractor (normalization disabled) from HuggingFace
    'AST_default_mel': {
        'nsynth': {'mean': -11.070530804913503, 'std': 5.6745043497320955},
        'surge': {'mean': -8.51401349612029, 'std': 6.022350417278173},
    }
}


class MergedDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir=None, sub_datasets=('nsynth', 'surge'), use_small_dataset=False, split='all',
                 exclude_augmented_train=False, exclude_augmented_valid_test=True):
        """
        MergedDataset that loads the .csv listing audio files of sub-datasets,
         and tries to merge those into a single big DataFrame.

        Args:
            data_dir (str, optional): Path to the data directory. If None, uses './data' relative to this file.
            split: 'all', 'train', 'valid', or 'test'
            exclude_augmented_valid_test: If True, excluded augmented samples from the validation and test sets.
        """
        super().__init__()
        if data_dir is not None:
            self.data_dir = Path(data_dir)
            if not self.data_dir.exists():
                raise ValueError(f"Provided data directory '{data_dir}' does not exist")
        else:
            self.data_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'data'
            if not self.data_dir.exists():
                raise ValueError(f"Default data directory '{self.data_dir}' does not exist. Maybe you forgot to create "
                                 f"a symlink from ./data to the actual parent dir for all datasets.")

        # Search for all available datasets
        found_dataset_dirs = []
        for item in self.data_dir.iterdir():
            if item.is_dir():
                for dataset_name in sub_datasets:
                    if item.name.startswith(f"{dataset_name}_v"):
                        is_small = item.name.endswith("_small")
                        folder_name = item.name
                        version_part = folder_name[:-6] if is_small else folder_name
                        version = version_part.split("_v")[1]
                        found_dataset_dirs.append({
                            "name": dataset_name, "version": version, "small": is_small, "path": str(item.absolute())
                        })
        self.available_datasets = pd.DataFrame(found_dataset_dirs)
        self.available_datasets = self.available_datasets.set_index('path').loc[natsorted(self.available_datasets['path'])].reset_index()
        # And retrieve the paths for the latest versions of each one
        self.sub_datasets_info = {}
        for dataset_name in sub_datasets:
            dataset_rows = self.available_datasets[
                (self.available_datasets['name'] == dataset_name) & (self.available_datasets['small'] == use_small_dataset)
            ]
            if dataset_rows.empty:
                raise ValueError(f"Requested dataset '{dataset_name}' with {use_small_dataset=} not found in available datasets")
            # Already version-sorted
            self.sub_datasets_info[dataset_name] = dataset_rows.iloc[-1]

        # If required: keep only the requested split  FIXME this should be done first... ? For faster loading and processing
        self.split = split
        assert self.split in ['all', 'train', 'valid', 'test']

        # Load the DFs for all audio files...
        self.sub_datasets_dfs = {}
        for ds_name, ds_info in self.sub_datasets_info.items():

            if ds_name == 'nsynth':
                nsynth_instance = nsynth.NSynthDataset(
                    self.data_dir, ds_info['version'], use_small_dataset,
                    exclude_extreme_pitches=True  # TODO make this an arg... maybe don't exclude but use weighted samplers
                )
                nsynth_df = nsynth_instance.df.copy()
                # Reduce the dataframe's size now - for (much) faster processing below
                if self.split != 'all':
                    nsynth_df = nsynth_df[nsynth_df['split'] == split]

                nsynth_df['path'] = nsynth_df['path'].apply(lambda x: Path(ds_info['path']) / x)
                # --- Build commons columns (values comparable with other sub-datasets) ---
                nsynth_df['instrument_UID'] = nsynth_df['instrument'].apply(lambda x: f'nsynth__{x}')
                nsynth_df.rename(columns={'pitch': 'midi_pitch', 'velocity': 'midi_velocity'}, inplace=True)
                nsynth_df['midi_instrument_group'] = nsynth_df['instrument_family'].apply(
                    lambda x: nsynth.INSTRUMENT_FAMILY_TO_MIDI_GROUP[x])
                nsynth_df['original_instrument_family'] = nsynth_df['instrument_family'].copy()
                nsynth_df['instrument_family'] = nsynth_df['instrument_family'].apply(
                    lambda x: nsynth.INSTRUMENT_FAMILY_TO_MERGED_FAMILY[x])
                # Effects: only consider some of those at the moment...
                fx = []
                for i, qualities in nsynth_df['qualities'].items():
                    qualities = ast.literal_eval(qualities)  # Will be a list of strings
                    fx.append([nsynth.QUALITY_TO_FX[q] for q in qualities if nsynth.QUALITY_TO_FX[q] is not None])
                nsynth_df.insert(6, 'fx', fx)
                nsynth_df['is_augmented'] = False  # No augmented samples in NSynth... was big enough
                self.sub_datasets_dfs[ds_name] = nsynth_df

            elif ds_name == 'surge':
                surge_df = pd.read_csv(Path(ds_info['path']) / 'audio.csv')
                # Reduce the dataframe's size now - for (much) faster processing below
                if self.split != 'all':
                    surge_df = surge_df[surge_df['split'] == split]

                surge_df['patch_fx'] = surge_df['patch_fx'].fillna('')
                surge_df['path'] = surge_df['path'].apply(lambda x: Path(ds_info['path']) / x)
                # --- Build commons columns (values comparable with other sub-datasets) ---
                surge_df['instrument_source'] = 'synthetic'  # Consistency with NSynth's labels
                surge_df['instrument_family'] = surge_df['patch_family'].apply(
                    lambda x: surge.SURGE_FAMILY_TO_MERGED_FAMILY[x])
                surge_df['instrument_UID'] = surge_df['patch_UID'].apply(lambda x: f'surge__{x}')
                surge_df['midi_instrument_group'] = surge_df['patch_family'].apply(
                    lambda x: surge.SURGE_FAMILY_TO_MIDI_TIMBRE_GROUP[x])
                # TODO Regroup effects into big categories or subsets... Which are not those of Surge
                fx = []
                for i, original_fx_list in surge_df['patch_fx'].items():
                    fx.append([])
                    for orig_fx in original_fx_list.split(','):
                        if orig_fx != '' and surge.SIMPLIFIED_FX_MAP[orig_fx] is not None:
                            fx[-1].append(surge.SIMPLIFIED_FX_MAP[orig_fx])
                surge_df.insert(4, 'fx', fx)
                # Store which ones are augmented (even for training) - Exclude those from validation and test sets...
                surge_df['is_augmented'] = (surge_df['patch_UID'].str.endswith("_aftertouch") | surge_df['patch_UID'].str.endswith("_no_fx"))
                if exclude_augmented_train:
                    surge_df = surge_df[~((surge_df['split'] == 'train') & surge_df['is_augmented'])]
                if exclude_augmented_valid_test:
                    surge_df = surge_df[~(surge_df['split'].isin(['test', 'valid']) & surge_df['is_augmented'])]
                self.sub_datasets_dfs[ds_name] = surge_df

            else:
                raise NotImplementedError(f"Unsupported dataset '{ds_name}'")
            self.sub_datasets_dfs[ds_name]['data_source'] = ds_name

        # Try to merge those DFs as best as possible - create a new index (concat allows duplicates !!!)
        self.df = pd.concat([df for _, df in self.sub_datasets_dfs.items()], axis=0, ignore_index=True)
        common_cols = [
            'split', 'data_source', 'instrument_UID',
            'instrument_family', 'fx',
            'midi_pitch', 'midi_velocity', 'path'
            # 'midi_instrument_group',  # Should not even be used, really messy/inconsistent labels
        ]
        others_cols = [c for c in self.df.columns if c not in common_cols]
        self.df = self.df.loc[:, common_cols + others_cols]
        self.df['midi_instrument_group'] = self.df['midi_instrument_group'].fillna('')
        # TODO a common probability column !!!!! (Normalized for each sub-dataset then weighted between sub-datasets)
        #          Also weight the probabilities for instrument families?

        # TODO Maybe an arg to exclude drums?

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> pd.Series:
        return self.df.iloc[idx]

    def __str__(self):
        if self.split == 'all':
            s = (f"{self.__class__.__name__} with {len(self.df)} audio files total: "
                 f"{self._get_train_valid_test_str(self.df)}.\n|_ The sub-datasets are: ")
            for ds_name in self.sub_datasets_dfs.keys():
                sub_df = self.df[self.df['data_source'] == ds_name]
                s += f"\n    |_ {ds_name.ljust(15)} {len(sub_df)} audio files: {self._get_train_valid_test_str(sub_df)}."
        else:
            s = (f"{self.__class__.__name__}: '{self.split}' split with {len(self.df)} audio files "
                 f"({len(set(self.df['instrument_UID'].values))} instruments)")
            for ds_name in self.sub_datasets_dfs.keys():
                sub_df = self.df[self.df['data_source'] == ds_name]
                s += f"\n    |_ {ds_name.ljust(15)} {len(sub_df)} audio files, {len(set(sub_df['instrument_UID'].values))} instruments"
        return s

    def _get_train_valid_test_str(self, df):
        s = ""
        for split in ['train', 'valid', 'test']:
            split_df = df[df['split'] == split]
            s += f"{split} {100.0 * (len(split_df) / len(df)):.1f}% ({len(set(split_df['instrument_UID'].values))} instruments), "
        return s[:-2]


if __name__ == "__main__":
    # Small tests
    from datetime import datetime

    for _split in ['train', 'valid', 'test']:  # 'all',
        t0 = datetime.now()
        _dataset = MergedDataset(sub_datasets=('nsynth', 'surge',), use_small_dataset=False, split=_split)
        print(f"{_split} dataset loaded in {(datetime.now() - t0).total_seconds():.1f}s")
        print(_dataset)