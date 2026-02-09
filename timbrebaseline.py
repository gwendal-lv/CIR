"""
This module provides baseline models for timbre search evaluation.
It includes a simple MFCC-based model and a Timbre Toolbox-based model that can be used with the TimbreSearchEvaluator.

Example usage:
    ```python
    from timbrebaseline import MFCCModel, TimbreToolboxModel
    from timbresearch import TimbreSearchEvaluator

    # Create an MFCC model with desired parameters
    model = MFCCModel(sr=16000, n_mfcc=40)
    # Or create a Timbre Toolbox model (always uses 44100 Hz)
    model = TimbreToolboxModel()
    # Use with TimbreSearchEvaluator
    evaluator = TimbreSearchEvaluator(model=model, sr=16000)
    results = evaluator.perform_eval()
    ```
"""
import multiprocessing
import threading
import warnings
from functools import cached_property
from typing import Sequence, Optional, Union, Protocol, runtime_checkable, List
from abc import ABC, abstractmethod
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchaudio
import wandb
import yaml
import numpy as np

import mergeddataset
import timbresearch
from ttb import peeTimbreToolbox
import utils


class TimbreBaselineModel(ABC):
    """
    Abstract base class for all timbre baseline models.

    This class provides common functionality for device handling and defines
    the interface that all timbre baseline models should implement.

    All subclasses must:
    1. Call super().__init__() in their __init__ method
    2. Implement _move_model_to_device() to move model-specific components to the device
    3. Implement compute_audio_embeddings() to process audio and return embeddings
    """

    def __init__(self, sr: int):
        # Set the device to CPU by default
        self.device = torch.device('cpu')
        self.sr = sr

    def to(self, device: Union[str, torch.device]) -> 'TimbreBaselineModel':
        """
        Move the model to the specified device.
        Only CPU is supported. Raises ValueError for CUDA or other devices.

        Args:
            device: The device to move the model to (must be 'cpu')

        Returns:
            The model itself for chaining

        Raises:
            ValueError: If device is not CPU
        """
        device_str = str(device)
        if 'cuda' in device_str or 'gpu' in device_str:
            raise ValueError("CUDA/GPU is not supported. Only CPU is allowed.")
        if device_str != 'cpu':
            raise ValueError(f"Unsupported device: {device_str}. Only CPU is allowed.")

        self.device = torch.device('cpu')
        self._move_model_to_device()
        return self

    def _move_model_to_device(self):
        """
        Move model-specific components to the device.
        This method should be overridden by subclasses.
        """
        pass

    def ensure_mono(self, audio: torch.Tensor) -> torch.Tensor:
        """ Ensure the audio is in mono format. """
        if len(audio.shape) == 2:
            assert audio.shape[0] == 1, "Only mono audio is supported"
            audio = audio[0, :]
        else:
            assert len(audio.shape) == 1, "Audio must be 1D or 2D with first dimension of size 1"
        return audio

    @abstractmethod
    def compute_audio_embeddings(self, audios: Sequence[torch.Tensor], sr: int, paths: Optional[Sequence[Path]] = None) -> torch.Tensor:
        """
        Compute embeddings for a batch of audio signals.

        Args:
            audios: A sequence of audio tensors
            sr: Sample rate of the audio signals (must match the model's sample rate)
            paths: Optional sequence of paths to the audio files (for caching - most models won't use this)

        Returns:
            A tensor of embeddings
        """
        pass


class MFCCModel(TimbreBaselineModel):
    """
    A simple model that computes MFCC features from audio signals.
    This model can be used with TimbreSearchEvaluator for baseline timbre search evaluation.

    The MFCC computation is performed by PyTorch and can run on CPU.
    """

    def __init__(self, sr: int, n_mfcc: int = 40, normalize=True):
        super().__init__(sr)
        self.n_mfcc = n_mfcc
        self.normalize = normalize
        # Default mel spectrogram parameters
        self.melkwargs = {
            'n_fft': 1024,  # Defaults: torch 400, librosa 2048
            'hop_length': 256,  # Defaults: torch 160, librosa 512
            'n_mels': 128,
        }

        # Create the MFCC transform
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sr, n_mfcc=self.n_mfcc,
            log_mels=True, melkwargs=self.melkwargs
        )
        self.normalization_stats = (self._get_normalization_stats() if self.normalize else None)

        # Move components to the device
        self._move_model_to_device()

    def _get_normalization_stats(self):
        if self.n_mfcc == 40 and self.sr == 16000:
            return {
                'mean': torch.Tensor([
                    # MFCCs Means
                    -8.5089e+01, 1.5571e+01, -2.6223e+00, 1.3299e+00, -2.4607e+00,
                    -8.1953e-01, -2.1949e+00, -7.2183e-01, -1.5913e+00, -5.0569e-01,
                    -1.2427e+00, -4.3838e-01, -9.8294e-01, -3.6136e-01, -8.1017e-01,
                    -1.0444e-01, -5.1890e-01, -1.2025e-01, -4.0863e-01, -3.4877e-02,
                    -3.5081e-01, 9.1711e-02, -2.5345e-01, -5.7540e-03, -4.0925e-01,
                    -2.2243e-01, -5.6323e-01, -3.3784e-01, -4.6410e-01, -9.7917e-02,
                    -1.9026e-01, 1.5280e-01, -4.2493e-02, 1.2168e-01, -2.6614e-01,
                    -1.2911e-01, -1.3205e-01, 2.4542e-01, 1.2546e-01, 3.2386e-01,
                    # MFCCs SDs
                    3.2768e+01, 1.0041e+01, 6.0289e+00, 3.9171e+00, 3.2816e+00,
                    2.5884e+00, 2.5334e+00, 2.2168e+00, 2.1975e+00, 2.0352e+00,
                    2.0363e+00, 1.9698e+00, 1.9633e+00, 1.8825e+00, 1.8481e+00,
                    1.7862e+00, 1.7976e+00, 1.7762e+00, 1.7720e+00, 1.7669e+00,
                    1.7459e+00, 1.7205e+00, 1.6877e+00, 1.6369e+00, 1.6267e+00,
                    1.6031e+00, 1.6088e+00, 1.6243e+00, 1.5918e+00, 1.5428e+00,
                    1.4818e+00, 1.5122e+00, 1.4765e+00, 1.4356e+00, 1.4536e+00,
                    1.4405e+00, 1.4102e+00, 1.3633e+00, 1.3897e+00, 1.3464e+00
                ]),
                'std': torch.Tensor([
                    # MFCCs Means
                    42.7248, 19.2136, 11.3715,  7.1817,  5.6140,  4.6094,  4.1912,  3.8205,
                     3.6929,  3.6379,  3.5310,  3.6200,  3.5455,  3.5482,  3.4326,  3.4356,
                     3.4326,  3.4417,  3.4356,  3.5483,  3.5519,  3.5338,  3.4346,  3.2928,
                     3.2089,  3.1950,  3.2833,  3.4713,  3.2535,  3.0781,  2.8646,  3.0175,
                     2.9529,  2.7702,  2.8275,  2.8578,  2.7097,  2.6593,  2.9005,  2.6695,
                    # MFCCs SDs
                    14.1973,  6.2504,  3.4360,  2.3256,  1.6266,  1.3475,  1.2817,  1.1855,
                     1.1457,  1.1329,  1.1295,  1.1574,  1.1391,  1.1379,  1.1022,  1.0910,
                     1.0834,  1.0977,  1.0987,  1.0986,  1.0862,  1.0856,  1.0414,  1.0244,
                     1.0308,  1.0196,  1.0187,  1.0397,  1.0082,  1.0015,  0.9198,  0.9690,
                     0.9384,  0.9211,  0.9384,  0.9077,  0.9033,  0.8371,  0.8956,  0.8210
                ])
            }
        else:
            raise ValueError(f"Stats not available for {self.n_mfcc=} and {self.sr=}")

    def _move_model_to_device(self):
        if hasattr(self, 'mfcc_transform'):
            self.mfcc_transform = self.mfcc_transform.to(self.device)

    def compute_audio_embeddings(self, audios: Sequence[torch.Tensor], sr: int, paths: Optional[Sequence[Path]] = None) -> torch.Tensor:
        """Compute MFCC embeddings for a batch of audio signals.
        Processes each audio individually without padding.

        Args:
            audios: A sequence of audio tensors
            sr: Sample rate of the audio signals (must match the model's sample rate)
            paths: Optional sequence of paths to the audio files (not used by this model)

        Returns:
            A tensor of MFCC embeddings with shape (len(audios), n_mfcc * 2)
        """
        assert sr == self.sr, f"Sample rate mismatch: expected {self.sr}, got {sr}"

        # Process each audio individually without padding
        all_embeddings = []
        for audio in audios:
            # Convert to mono if needed
            audio = self.ensure_mono(audio)

            # Process each audio individually
            audio_batch = audio.unsqueeze(0).to(self.device)  # Add batch dimension
            # Compute MFCCs for this single audio
            with torch.no_grad():
                mfcc = self.mfcc_transform(audio_batch)  # Shape: (1, n_mfcc, time)
                mean, std = torch.mean(mfcc, dim=2), torch.std(mfcc, dim=2)  # Shape: (1, n_mfcc)
                embedding = torch.cat([mean, std], dim=1)
                if self.normalize:  # Broadcast computation (no batch dim for the stats)
                    embedding = (embedding - self.normalization_stats['mean']) / self.normalization_stats['std']
                all_embeddings.append(embedding)

        return torch.cat(all_embeddings, dim=0)  # Shape: (batch_size, n_mfcc*2)


class MFCCTimbreEvaluator:
    """ A class that evaluates MFCC-based timbre search using TimbreSearchEvaluator. """

    def __init__(self, config, secrets_path="secrets.yaml",
                 eval_split='test', distance_metrics=('l2', 'cosine', ), use_cuda=False):
        self.config, self.sr, self.n_mfcc = config, config['model']['sr'], config['model']['num_mfccs']
        self.distance_metrics = distance_metrics
        with open(secrets_path, "r") as f:
            self.secrets = yaml.safe_load(f)
        wandb.login(key=self.secrets['wandb']['api_key'])

        self.model = MFCCModel(sr=self.sr, n_mfcc=self.n_mfcc, normalize=True)

        self.eval_split = eval_split
        assert self.eval_split in ['valid', 'test']
        ref_datasets = ['train', 'valid',] + (['test'] if self.eval_split == 'test' else [])
        self.evaluator = timbresearch.TimbreSearchEvaluator(
            model=self.model, sr=self.sr, batch_size=1,
            use_small_dataset=config['dataset']['use_small_dataset'],
            eval_split=self.eval_split, reference_datasets=ref_datasets,
            verbose=True, use_cuda=use_cuda
        )

    def perform_eval(self):
        # Create the W&B run now
        # The valid is ignored; the "dataset" as a whole is considered small if either the train or the test is small
        _small_dataset = (self.config['dataset']['use_small_dataset']['train'] or self.config['dataset']['use_small_dataset']['test'])
        self.wandb_run = wandb.init(
            entity=self.secrets['wandb']['team'],
            project=self.config['run']['project'],
            name=self.config['run']['name'],
            config={
                "model_type": f"{self.config['model']['type']}{self.config['model']['num_mfccs']}",
                "sr": self.sr,
                "use_small_dataset": _small_dataset,
            }
        )

        eval_sounds_df, metrics = self.evaluator.perform_eval(distance_metrics=self.distance_metrics)
        figs = self.evaluator.plot_eval_sounds_df(eval_sounds_df)

        # Log metrics and figures to wandb
        log_data = {}
        for metric_name, metric_value in metrics.items():
            log_data[f'timbresearch/{metric_name}/{self.eval_split}'] = metric_value
        for fig_name, fig in figs.items():
            log_data[f'timbresearchfig/{fig_name}/{self.eval_split}'] = wandb.Image(fig)
        self.wandb_run.log(data=log_data)

        return eval_sounds_df, metrics, figs

    def __del__(self):
        """Ensure wandb run is properly closed when the object is destroyed."""
        if hasattr(self, 'wandb_run') and self.wandb_run is not None:
            self.wandb_run.finish()


class TimbreToolboxModel(TimbreBaselineModel):
    """
    A model that computes Timbre Toolbox features from audio signals.
    This model can be used with TimbreSearchEvaluator for timbre search evaluation.

    The Timbre Toolbox computation is performed by the peeTimbreToolbox module and runs on CPU.

    Note: The Timbre Toolbox is designed to work with a 44100 Hz sample rate.
    This model always uses 44100 Hz internally and expects input audio to be at 44100 Hz.
    """

    def __init__(self, normalize=True, exclude_loudness_pitch_duration_descriptors=True, n_workers=1):
        # Always use 44100 Hz for the Timbre Toolbox
        super().__init__(44100)
        self.normalize, self.exclude_loudness_pitch_duration_descriptors = normalize, exclude_loudness_pitch_duration_descriptors
        self.normalization_stats = None
        self.n_workers = n_workers

        # Move components to the device
        self._move_model_to_device()

        self.cache_dir = Path(__file__).parent.resolve() / 'logs/TTB-embeddings-cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        assert len(self.descriptors_name) == len(self.descriptors_mean) == len(self.descriptors_std)

    @cached_property
    def descriptors_categories(self):
        return [
            'TEE',  # Temporal Energy Envelope
            'STFTmag',  # Discard other spectral descriptors... To keep a reasonable amount of spectral features
            # 'STFTpow',
            'Harmonic',
            # 'ERBfft', 'ERBgam',
        ]

    @cached_property
    def descriptors_name(self):
        return [
            'TEE-Att', 'TEE-Dec', 'TEE-Rel', 'TEE-LAT', 'TEE-AttSlope', 'TEE-DecSlope', 'TEE-TempCent', 'TEE-EffDur',
            'TEE-FreqMod', 'TEE-AmpMod', 'TEE-RMSEnv-median', 'TEE-RMSEnv-iqr',
            'STFTmag-SpecCent-median', 'STFTmag-SpecCent-iqr', 'STFTmag-SpecSpread-median', 'STFTmag-SpecSpread-iqr', 'STFTmag-SpecSkew-median', 'STFTmag-SpecSkew-iqr',
            'STFTmag-SpecKurt-median', 'STFTmag-SpecKurt-iqr', 'STFTmag-SpecSlope-median', 'STFTmag-SpecSlope-iqr', 'STFTmag-SpecDecr-median', 'STFTmag-SpecDecr-iqr',
            'STFTmag-SpecRollOff-median', 'STFTmag-SpecRollOff-iqr', 'STFTmag-SpecVar-median', 'STFTmag-SpecVar-iqr', 'STFTmag-FrameErg-median', 'STFTmag-FrameErg-iqr',
            'STFTmag-SpecFlat-median', 'STFTmag-SpecFlat-iqr', 'STFTmag-SpecCrest-median', 'STFTmag-SpecCrest-iqr',
            # 'STFTpow-SpecCent-median', 'STFTpow-SpecCent-iqr', 'STFTpow-SpecSpread-median', 'STFTpow-SpecSpread-iqr', 'STFTpow-SpecSkew-median', 'STFTpow-SpecSkew-iqr', 'STFTpow-SpecKurt-median', 'STFTpow-SpecKurt-iqr', 'STFTpow-SpecSlope-median', 'STFTpow-SpecSlope-iqr', 'STFTpow-SpecDecr-median', 'STFTpow-SpecDecr-iqr', 'STFTpow-SpecRollOff-median', 'STFTpow-SpecRollOff-iqr', 'STFTpow-SpecVar-median', 'STFTpow-SpecVar-iqr', 'STFTpow-FrameErg-median', 'STFTpow-FrameErg-iqr', 'STFTpow-SpecFlat-median', 'STFTpow-SpecFlat-iqr', 'STFTpow-SpecCrest-median', 'STFTpow-SpecCrest-iqr',
            'Harmonic-HarmErg-median', 'Harmonic-HarmErg-iqr', 'Harmonic-NoiseErg-median',
            'Harmonic-NoiseErg-iqr', 'Harmonic-Noisiness-median', 'Harmonic-Noisiness-iqr', 'Harmonic-F0-median',
            'Harmonic-F0-iqr', 'Harmonic-InHarm-median', 'Harmonic-InHarm-iqr', 'Harmonic-TriStim1-median',
            'Harmonic-TriStim1-iqr', 'Harmonic-TriStim2-median', 'Harmonic-TriStim2-iqr', 'Harmonic-TriStim3-median',
            'Harmonic-TriStim3-iqr', 'Harmonic-HarmDev-median', 'Harmonic-HarmDev-iqr', 'Harmonic-OddEveRatio-median',
            'Harmonic-OddEveRatio-iqr', 'Harmonic-SpecCent-median', 'Harmonic-SpecCent-iqr',
            'Harmonic-SpecSpread-median', 'Harmonic-SpecSpread-iqr', 'Harmonic-SpecSkew-median',
            'Harmonic-SpecSkew-iqr', 'Harmonic-SpecKurt-median', 'Harmonic-SpecKurt-iqr', 'Harmonic-SpecSlope-median',
            'Harmonic-SpecSlope-iqr', 'Harmonic-SpecDecr-median', 'Harmonic-SpecDecr-iqr',
            'Harmonic-SpecRollOff-median', 'Harmonic-SpecRollOff-iqr', 'Harmonic-SpecVar-median',
            'Harmonic-SpecVar-iqr', 'Harmonic-FrameErg-median', 'Harmonic-FrameErg-iqr'
            # 'ERBfft-SpecCent-median', 'ERBfft-SpecCent-iqr', 'ERBfft-SpecSpread-median', 'ERBfft-SpecSpread-iqr', 'ERBfft-SpecSkew-median', 'ERBfft-SpecSkew-iqr', 'ERBfft-SpecKurt-median', 'ERBfft-SpecKurt-iqr', 'ERBfft-SpecSlope-median', 'ERBfft-SpecSlope-iqr', 'ERBfft-SpecDecr-median', 'ERBfft-SpecDecr-iqr', 'ERBfft-SpecRollOff-median', 'ERBfft-SpecRollOff-iqr', 'ERBfft-SpecVar-median', 'ERBfft-SpecVar-iqr', 'ERBfft-FrameErg-median', 'ERBfft-FrameErg-iqr', 'ERBfft-SpecFlat-median', 'ERBfft-SpecFlat-iqr', 'ERBfft-SpecCrest-median', 'ERBfft-SpecCrest-iqr',
            #'ERBgam-SpecCent-median', 'ERBgam-SpecCent-iqr', 'ERBgam-SpecSpread-median', 'ERBgam-SpecSpread-iqr', 'ERBgam-SpecSkew-median', 'ERBgam-SpecSkew-iqr', 'ERBgam-SpecKurt-median', 'ERBgam-SpecKurt-iqr', 'ERBgam-SpecSlope-median', 'ERBgam-SpecSlope-iqr', 'ERBgam-SpecDecr-median', 'ERBgam-SpecDecr-iqr', 'ERBgam-SpecRollOff-median', 'ERBgam-SpecRollOff-iqr', 'ERBgam-SpecVar-median', 'ERBgam-SpecVar-iqr', 'ERBgam-FrameErg-median', 'ERBgam-FrameErg-iqr', 'ERBgam-SpecFlat-median', 'ERBgam-SpecFlat-iqr', 'ERBgam-SpecCrest-median', 'ERBgam-SpecCrest-iqr'
        ]

    @cached_property
    def loudness_pitch_duration_descriptors(self):
        return [
            'TEE-TempCent', 'TEE-EffDur',
            # Not the IQR for these, we only exclude the median
            'TEE-RMSEnv-median', 'STFTmag-FrameErg-median',
            'Harmonic-HarmErg-median', 'Harmonic-NoiseErg-median',  # The relationship between these two (noisiness) is kept
            'Harmonic-F0-median', 'Harmonic-FrameErg-median',
        ]

    @cached_property
    def loudness_pitch_duration_descriptors_mask(self):
        return torch.tensor([d in self.loudness_pitch_duration_descriptors for d in self.descriptors_name])

    @cached_property
    def descriptors_mean(self):
        return torch.tensor([0.09378033131361008, 0.606132447719574, 2.3251681327819824, -0.8972233533859253, 8.99434757232666, -1.789367437362671, 1.3436615467071533, 2.144475221633911, 4.24691915512085, 0.10508067905902863, 0.18200890719890594, 0.1610337197780609, 0.23753927648067474, 0.010283468291163445, 0.3257954716682434, 0.015794318169355392, 1.35615074634552, 0.05746081843972206, 1.9318474531173706, 0.0904107615351677, -3.8527414290001616e-05, 3.828828994301148e-05, -0.02342260256409645, 0.05196380987763405, 0.4958915412425995, 0.0010821152245625854, 0.05993000045418739, 0.04982660338282585, 0.019829565659165382, 0.016113432124257088, 0.1343519389629364, 0.07311363518238068, 108.72360229492188, 35.35969161987305, 0.08157128840684891, 0.06462990492582321, 0.673788845539093, 0.5139181613922119, 0.8698013424873352, 0.04673357307910919, 210.64340209960938, 27.86553382873535, 0.04315129667520523, 0.028599567711353302, 0.32706013321876526, 0.12492049485445023, 0.35006535053253174, 0.12005992233753204, 0.2460978925228119, 0.0980512797832489, 0.006485172547399998, 0.0051910667680203915, 419.0547180175781, 496.53460693359375, 743.5712280273438, 196.137451171875, 554.8678588867188, 149.7090301513672, 2.821758508682251, 1.0811331272125244, 19.339244842529297, 10.893824577331543, -1.8113192709279247e-05, 4.890913714916678e-06, -5.067453384399414, 3.881977081298828, 1715.73779296875, 489.3162841796875, 0.06808064132928848, 0.08031246811151505, 0.7567353248596191, 0.5774176716804504])

    @cached_property
    def descriptors_std(self):
        return torch.tensor([0.2304910123348236, 0.7115341424942017, 1.3778265714645386, 0.3999353349208832, 4.081834316253662, 3.384608745574951, 0.7254782319068909, 1.3107779026031494, 1.963175654411316, 0.12030947208404541, 0.21346570551395416, 0.12839728593826294, 0.05021042749285698, 0.038719113916158676, 0.06875088810920715, 0.05220308154821396, 0.2844914495944977, 0.22363866865634918, 0.4132397174835205, 0.3239278197288513, 0.0001776932622306049, 0.00013387018407229334, 0.2277335822582245, 0.17635966837406158, 0.00739692896604538, 0.003981470596045256, 0.19804328680038452, 0.15249252319335938, 0.02220558561384678, 0.014775951392948627, 0.2036484330892563, 0.13693034648895264, 77.77043151855469, 42.59767150878906, 0.17168287932872772, 0.10014322400093079, 1.2798044681549072, 0.6950480937957764, 0.18091046810150146, 0.13997957110404968, 123.7750015258789, 60.43332290649414, 0.18040598928928375, 0.1588144153356552, 0.2852189838886261, 0.1248091384768486, 0.2283778190612793, 0.10534588247537613, 0.23376253247261047, 0.10469552874565125, 0.009586581960320473, 0.005788127426058054, 2345.15234375, 3854.912841796875, 632.8953247070312, 267.4530944824219, 366.2196044921875, 174.30630493164062, 2.2695183753967285, 1.0761570930480957, 24.415634155273438, 15.87020206451416, 1.7269630916416645e-05, 1.0978436876030173e-05, 13.930851936340332, 9.62724781036377, 1312.00146484375, 606.8704223632812, 0.20946156978607178, 0.1775669902563095, 1.4374059438705444, 0.7888889908790588])

    def _move_model_to_device(self):
        # No specific components to move for this model
        pass

    def _compute_audio_embedding(self, audio: torch.Tensor, path: Optional[Path]=None) -> torch.Tensor:
        """ Compute Timbre Toolbox embedding for a single audio signal and caches it locally, and loads the
                embedding vector from the cache if it exists. """
        if path is not None:
            rel_path = path.relative_to(Path(__file__).parent / 'data')  # Go back 3 dirs
            embed_path = self.cache_dir / rel_path.with_suffix(f"{rel_path.suffix}.pt")
        else:
            embed_path = None
        # Try to load the embedding from the precomputed embeddings cache
        if embed_path is not None and embed_path.exists():
            embedding = torch.load(embed_path)
        # Otherwise compute it and store it
        else:
            audio = self.ensure_mono(audio)
            # Convert to numpy for Timbre Toolbox processing
            audio_np = audio.cpu().numpy()
            # Compute Timbre Toolbox descriptors
            try:
                # Compute all descriptors using 44100 Hz sample rate
                descHub_d = peeTimbreToolbox.F_computeAllDescriptor(audio_np, 44100)
                # Apply temporal modeling to get statistics
                descHub_d = peeTimbreToolbox.F_temporalModeling(descHub_d)
                # FIXME Many missing features here... !
                # Extract features from the descriptors
                features = {}
                # Process each descriptor category
                for category in self.descriptors_categories:
                    if category in descHub_d:
                        for desc_name in descHub_d[category]:
                            # Identify single-value features (e.g., Attack time, Release time, ...)
                            if descHub_d[category][desc_name]['value'].shape == (1, 1):
                                features[f"{category}-{desc_name}"] = descHub_d[category][desc_name]['value'][0, 0]
                            # For time-varying features: Use median and IQR as features
                            else:
                                for stat_type in ['median', 'iqr']:
                                    assert stat_type in descHub_d[category][desc_name]
                                    val = descHub_d[category][desc_name][stat_type]
                                    val = val.flatten()
                                    assert len(val) == 1
                                    features[f"{category}-{desc_name}-{stat_type}"] = val[0]
                # Check the names of features/descriptors, then convert features to tensor
                for i, feature_name in enumerate(features.keys()):
                    assert (self.descriptors_name[i] == feature_name), f"Descriptor name mismatch ({i=}): found {feature_name=}, expected {self.descriptors_name[i]=}"
                embedding = torch.tensor([v for k, v in features.items()], dtype=torch.float32).unsqueeze(0)
            except Exception as e:
                warnings.warn(f"Error computing embedding for audio {path=}: {e}")
                embedding = None

        # Save if a path was provided
        if embed_path is not None:
            if not embed_path.parent.exists():
                embed_path.parent.mkdir(parents=True, exist_ok=False)
            utils.save_pytorch_data_safely(embedding, embed_path)

        if embedding is not None:
            if self.normalize:
                embedding = (embedding - self.descriptors_mean) / self.descriptors_std
            if self.exclude_loudness_pitch_duration_descriptors:
                embedding = embedding[:, ~self.loudness_pitch_duration_descriptors_mask]
        return embedding

    def compute_audio_embeddings(self, audios: Sequence[torch.Tensor], sr: int, paths: Optional[Sequence[Path]] = None) -> torch.Tensor:
        """Compute Timbre Toolbox embeddings for a batch of audio signals.
        Processes each audio individually without padding.

        Args:
            audios: A sequence of audio tensors
            sr: Sample rate of the audio signals (must be 44100 Hz)
            paths: Optional sequence of paths to the audio files (can be used by this model to retrieve pre-computed embeddings)

        Returns:
            A tensor of Timbre Toolbox embeddings
        """
        assert sr == 44100, f"Sample rate must be 44100 Hz, got {sr=}"
        if paths is None:
            warnings.warn("paths arg not provided, so all timbre features will be recomputed (very slow!)")
            paths = [None] * len(audios)

        # Process audio files in parallel if n_workers > 1
        all_embeddings = []

        if self.n_workers > 1:
            # print(f"Multiprocessing enabled ({self.n_workers=})")
            # Use spawn method instead of fork to avoid "too many files opened" errors
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(processes=self.n_workers) as pool:
                # Create a list of (audio, path) tuples for processing
                tasks = [(audios[i], paths[i]) for i in range(len(audios))]
                # Map the _compute_audio_embedding_wrapper function to the tasks
                all_embeddings = pool.starmap(self._compute_audio_embedding, tasks)
        else:
            # print("Multiprocessing disabled (n_workers==1)")
            for i, (audio, path) in enumerate(zip(audios, paths)):
                embedding = self._compute_audio_embedding(audio, path)
                all_embeddings.append(embedding)

        return self.concat_embeddings(all_embeddings)

    @staticmethod
    def concat_embeddings(all_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """ Concatenate embeddings from a batch of audio signals, replacing None embeddings
            (computation failures) with NaNs of the proper shape. """
        # First, find the first non-None embedding to get a shape
        embeds_shape = None
        for embed in all_embeddings:
            if embed is not None:
                embeds_shape = embed.shape
                break
        # Now check size and handle the None embeddings (computation failures...)
        for i in range(len(all_embeddings)):
            if all_embeddings[i] is not None:
                assert all_embeddings[i].shape == embeds_shape, f"Invalid embedding shape: {all_embeddings[i].shape=} != {embeds_shape=}"
            else:  # failures stored as NaNs
                all_embeddings[i] = torch.empty(embeds_shape)
                all_embeddings[i][:] = float('nan')
        # Concatenate all embeddings now
        return torch.cat(all_embeddings, dim=0)


class TimbreToolboxEvaluator:
    """ A class that evaluates Timbre Toolbox-based timbre search using TimbreSearchEvaluator. """

    def __init__(self, config, secrets_path="secrets.yaml",
                 eval_split='test', distance_metrics=('l2', 'cosine'), use_cuda=False):
        self.config, self.sr = config, config['model']['sr']
        self.distance_metrics = distance_metrics
        self.n_workers = config['model'].get('n_workers', 1)

        with open(secrets_path, "r") as f:
            self.secrets = yaml.safe_load(f)
        wandb.login(key=self.secrets['wandb']['api_key'])

        self.model = TimbreToolboxModel(
            normalize=True,
            exclude_loudness_pitch_duration_descriptors=True,
            n_workers=self.n_workers
        )

        self.eval_split = eval_split
        assert self.eval_split in ['valid', 'test']
        ref_datasets = ['train', 'valid',] + (['test'] if self.eval_split == 'test' else [])
        # Use a batch size that does not saturate the RAM, does not open too many files simultaneously, ...
        self.evaluator = timbresearch.TimbreSearchEvaluator(
            model=self.model, sr=self.sr, batch_size=256,
            use_small_dataset=config['dataset']['use_small_dataset'],
            eval_split=self.eval_split, reference_datasets=ref_datasets,
            verbose=True, use_cuda=use_cuda
        )

    def get_embeddings_stats(self):
        normalize_backup, exclude_loudness_pitch_descriptors_backup = self.model.normalize, self.model.exclude_loudness_pitch_duration_descriptors
        self.model.normalize, self.model.exclude_loudness_pitch_duration_descriptors = False, False
        # This is extremely slow... so we'll just use stats from non-test pre-computed embeddings
        # small hack: Provide the path only (everything must have been cached already...), don't load the audio
        paths = self.evaluator.sounds_df[self.evaluator.sounds_df['is_reference']]['path'].to_list()
        embeddings = [self.model._compute_audio_embedding(None, p) for p in paths]
        embeddings = self.model.concat_embeddings(embeddings)
        partial_failure_mask = torch.any(torch.isnan(embeddings), dim=1)  # Include failures of a few features only
        means = embeddings[~partial_failure_mask, :].mean(dim=0)
        stds = embeddings[~partial_failure_mask, :].std(dim=0)
        print("========== Means ==========")
        print([float(v) for v in means])
        print("========== SDs ==========")
        print([float(v) for v in stds])
        self.model.normalize, self.model.exclude_loudness_pitch_duration_descriptors = normalize_backup, exclude_loudness_pitch_descriptors_backup
        return means, stds

    def perform_eval(self):
        # Create the W&B run now
        # The valid is ignored; the "dataset" as a whole is considered small if either the train or the test is small
        _small_dataset = (self.config['dataset']['use_small_dataset']['train'] or self.config['dataset']['use_small_dataset']['test'])
        self.wandb_run = wandb.init(
            entity=self.secrets['wandb']['team'],
            project=self.config['run']['project'],
            name=self.config['run']['name'],
            config={
                "model_type": f"{self.config['model']['type']}",
                "sr": self.sr,
                "use_small_dataset": _small_dataset,
            }
        )

        print(f"[TimbreToolboxModel/Evaluator] embeddings cache: {self.model.cache_dir}")
        # Track advancement of embeddings' computation using a parallel thread
        # Create an event to signal when to stop the tracking thread
        stop_event = threading.Event()
        # Start the tracking thread
        tracking_thread = threading.Thread(
            target=utils.track_files_progress,
            args=(self.model.cache_dir, self.evaluator.embeddings_count),
            kwargs={'stop_event': stop_event, 'update_interval': 30}
        )
        tracking_thread.daemon = True  # Thread will exit when main thread exits
        tracking_thread.start()

        try:
            # Perform the evaluation
            eval_sounds_df, metrics = self.evaluator.perform_eval(distance_metrics=self.distance_metrics)
            figs = self.evaluator.plot_eval_sounds_df(eval_sounds_df, distance_metric='cosine')
            log_data = {}
            for metric_name, metric_value in metrics.items():
                log_data[f'timbresearch/{metric_name}/{self.eval_split}'] = metric_value
            for fig_name, fig in figs.items():
                log_data[f'timbresearchfig/{fig_name}/{self.eval_split}'] = wandb.Image(fig)
            self.wandb_run.log(data=log_data)

            return eval_sounds_df, metrics, figs
        finally:
            # Signal the tracking thread to stop
            stop_event.set()
            tracking_thread.join(timeout=1.0)  # Wait for the thread to finish

    def __del__(self):
        """Ensure wandb run is properly closed when the object is destroyed."""
        if hasattr(self, 'wandb_run') and self.wandb_run is not None:
            self.wandb_run.finish()


if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Model for timbre search')
    parser.add_argument('config_file', type=str, help='Path to config file (e.g., config/MFCC-baseline.yaml) containing model and dataset parameters')
    parser.add_argument('--compute-stats', action='store_true', help='Compute statistics for the training dataset')
    args = parser.parse_args()

    # Load config file (mandatory)
    with open(Path(args.config_file), 'r') as f:
        _config = yaml.safe_load(f)
    print(f"Loaded config from {Path(args.config_file)}")
    print(f"{_config['model']['type']=}")

    if _config['model']['type'] == 'MFCC':
        _sr, _n_mfcc = _config['model']['sr'], _config['model']['num_mfccs']

        if not args.compute_stats:
            # Create and use MFCCTimbreEvaluator
            _evaluator = MFCCTimbreEvaluator(_config, secrets_path="secrets.yaml")
            _eval_sounds_df, _metrics, _figs = _evaluator.perform_eval()
            print(f"Metrics:\n" + ', '.join([f'{k}: {v:.3f}' for k, v in _metrics.items()]))
            plt.show()  # Display plots locally in addition to wandb

        # In this section: compute embeddings for the whole training dataset (to compute normalization statistics)
        else:
            with utils.measure_time("Computing training dataset stats"):
                with utils.measure_time("Loading training dataset"):
                    # No padding: use individual audio files (no real batching here)
                    _dataset = mergeddataset.MultimodalSoundsDataset(use_small_dataset=True, split='train', pad=None, target_sr=_sr)
                print(_dataset)

                # Create MFCC model with parameters from config
                _model = MFCCModel(sr=_sr, n_mfcc=_n_mfcc, normalize=False)
                print(f"Created MFCC model with {_model.n_mfcc} coefficients")

                _training_embeddings = []
                for _i in tqdm(range(len(_dataset)), desc="Processing audio files", mininterval=0.5, total=len(_dataset)):
                    _, _, _audio = _dataset[_i]
                    _embedding = _model.compute_audio_embeddings([_audio], _model.sr, paths=None)
                    _training_embeddings.append(_embedding)
                _training_embeddings = torch.cat(_training_embeddings, dim=0)
            print("---------- Means ----------\n", _training_embeddings.mean(dim=0),
                  "\n---------- Standard Deviations ----------\n", _training_embeddings.std(dim=0))

    elif _config['model']['type'] == 'TimbreToolBox':
        _evaluator = TimbreToolboxEvaluator(_config, secrets_path="secrets.yaml")
        if not args.compute_stats:
            _eval_sounds_df, _metrics, _figs = _evaluator.perform_eval()  # Will use a cache for embeddings
            print(f"Metrics:\n" + ', '.join([f'{k}: {v:.3f}' for k, v in _metrics.items()]))
            plt.show()
        else:
            with utils.measure_time("Computing training dataset stats"):
                means, stds = _evaluator.get_embeddings_stats()

    else:
        print(f"Model type '{_config['model']['type']}' is not implemented.\nCurrently supported model types:\n  - MFCC\n  - TimbreToolBox\n\nExample usage:\n  python timbrebaseline.py config/MFCC-baseline.yaml [--compute-stats]\n  python timbrebaseline.py config/TTB-baseline.yaml [--compute-stats]\n  python timbrebaseline.py config/TTB-baseline.yaml --precompute-embeddings")
