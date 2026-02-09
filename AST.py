"""
This module provides a customized implementation of the Audio Spectrogram Transformer (AST) model for audio processing.
It can be imported to use AST models in your code. The module extends HuggingFace's implementation with additional 
functionality for fine-tuning and audio embedding extraction. For command-line operations such as loading/testing 
model checkpoints, uploading models to Weights & Biases (wandb), or downloading models from wandb, see ASTwandb.py.
"""
import pathlib
import warnings
from typing import Any, Optional, Union, Iterable, Sequence

import torch
import torch.nn.functional as F
import librosa

from transformers import ASTForAudioClassification, ASTFeatureExtractor
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import ASTMLPHead, ASTConfig

import utils


class ASTFineTuned(ASTForAudioClassification):
    @staticmethod
    def from_custom_config(custom_config_from_yaml: dict[str, Any]):
        """ Similar to HF's from_pretrained(...), but using a yaml config (already loaded) rather than a checkpoint """
        # This will call the __init__ below (for the actual instance)
        model = ASTFineTuned.from_pretrained(custom_config_from_yaml['model']['pretrained'])
        model.prepare_for_downstream_task(custom_config_from_yaml)
        return model

    def __init__(self, config: ASTConfig):
        super().__init__(config)
        self.custom_config = {}  # Custom config, originally in a local .yaml file
        self.feature_extractor: Optional[ASTFeatureExtractor] = None

    @property
    def sr(self):
        return self.custom_config['model']['sr']

    def prepare_for_downstream_task(self, custom_config_from_yaml: dict[str, Any]):
        """
        Changes the number of output logits, instantiates a new classification head with a different number of classes.
        Must be called after some config and weights have been preloaded.
        """
        # Custom additional config
        self.custom_config = custom_config_from_yaml

        # modify the parent class' .config and .classifier
        self.config.id2label = self.custom_config['dataset']['id2label']  # May be an empty dict if not a classifier
        self.config.label2id = {label: idx for idx, label in self.config.id2label.items()}

        if self.custom_config['model']['type'] in [
            'MultiLabelClassification', 'InstrumentGroupClassification', 'InstrumentDirectClassification',
            'MultiEncoder',  # Not really a classifier... but we need custom-sized output vectors (even if these are not considered as logits)
        ]:
            # Modifications to the model itself... classifier only
            # The head will configure itself from the config (with auto-updated num_labels at this point)
            self.classifier = ASTMLPHead(self.config)

        elif self.custom_config['model']['type'] in ['ContrastiveEmbeddings', 'ContrastiveMixEmbeddings']:
            self.classifier = torch.nn.Identity()

        else:
            raise NotImplementedError(f'Unexpected model type: {self.custom_config["model"]["type"]}')

        # Pre-processing will be performed on the CPU - OK in practice (the transformer on GPU is quite large)
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(self.custom_config['model']['pretrained'])
        # Check now that all feature extractors have compatible parameters
        assert self.feature_extractor.sampling_rate == self.custom_config['model']['sr']
        assert self.feature_extractor.num_mel_bins == self.custom_config['model']['num_mel_bins']
        assert self.feature_extractor.max_length == self.custom_config['model']['max_num_mel_frames']

    def forward(self, audios: torch.Tensor):
        """ Override of the default forward, in order to also provide the last embeddings (the vectors used
        at the input of the classification layer) """

        # Compute mel-specs on the CPU... Then move them on the GPU
        #     AST Feature extractors only properly work with numpy audio (because of what it uses to check
        #     dimensions/bacthing).  Returns a single-key dict: No attention mask returned
        mel_spectrograms = self.feature_extractor(
            audios.cpu().numpy(), sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt",
        )['input_values']
        mel_spectrograms = mel_spectrograms.to(self.device)

        # Apply the main transformer model
        outputs = super().forward(input_values=mel_spectrograms, output_hidden_states=True)

        # Code extracted from the HuggingFace implementation
        last_hidden_sequence = outputs.hidden_states[-1]
        last_hidden_sequence = self.base_model.layernorm(last_hidden_sequence)
        # Final embeddings used for classification, named "pooled_output" in the original implementation
        last_hidden_embeddings = (last_hidden_sequence[:, 0] + last_hidden_sequence[:, 1]) / 2  # OK, double-checked

        return outputs, last_hidden_embeddings

    def compute_audio_embeddings(self, audios: Sequence[torch.Tensor], sr: int, paths: Optional[Sequence[pathlib.Path]] = None) -> torch.Tensor:
        # Perform some checks, retrieve the longest audio
        # TODO Move to GPU here? (maybe move self. also) if necessary and available
        assert sr == self.feature_extractor.sampling_rate, f"{sr}, {self.feature_extractor.sampling_rate} mismatch"
        max_len = -1
        padded_audios = list()
        for a in audios:
            if len(a.shape) == 2:
                assert a.shape[0] == 1, "Mono only expected"
                a = a[0, :]
            else:
                assert len(a.shape) == 1
            max_len = max(a.shape[0], max_len)
            padded_audios.append(a)  # Not actually padded yet...
        # Pad audios then Generate a proper batch - to  Apply the model in one call
        padded_audios = [F.pad(a, (0, max_len - a.shape[0]), mode='constant', value=0.0) for a in padded_audios]
        padded_audios = torch.stack(padded_audios)
        with torch.no_grad():
            outputs, last_hidden_embeddings = self.forward(padded_audios)
        return last_hidden_embeddings

    def test_with_dummy_audio(self):
        """
        Tests the model with a dummy audio (trumpet sample from librosa) to verify functionality.
        Measures execution time and prints information about the processed audio.
        Returns the input audio tensor and the computed embeddings.
        """
        with utils.measure_time("Testing the model with a dummy audio"):
            # Load trumpet sample from librosa, resampled to the model's sample rate
            input_audio, sr = librosa.load(librosa.ex('trumpet'), sr=self.sr)
            # Limit to 10 seconds
            input_audio = input_audio[:(10 * sr)]
            # Convert to torch tensor with batch dimension
            input_audio = torch.tensor(input_audio).unsqueeze(0)
            # Process through the model
            with torch.no_grad():
                embeddings = self.compute_audio_embeddings([input_audio], sr)
        return embeddings

    def save_checkpoint(self, path: pathlib.Path):
        checkpoint = {'model_state_dict': self.state_dict(), 'custom_config': self.custom_config}
        torch.save(checkpoint, path)

    @staticmethod
    def from_checkpoint(checkpoint_path: Union[str, pathlib.Path]):
        checkpoint = torch.load(checkpoint_path)
        # Check that git config is OK...
        current_git_state, checkpoint_git_state = utils.get_git_info(), checkpoint['custom_config']['git']
        for k in current_git_state.keys():
            if current_git_state[k] != checkpoint_git_state[k]:
                warnings.warn(f"\nGit {k} is different in the checkpoint and in the current code\n\tcurrent: {current_git_state[k]}, checkpoint: {checkpoint_git_state[k]}")
        model = ASTFineTuned.from_custom_config(checkpoint['custom_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


def main():
    model = ASTFineTuned.from_checkpoint('./ICASSP26_Triplet-full_bs48_ckpt_final.pt')
    embeddings = model.test_with_dummy_audio()
    print(f"{embeddings.shape=}")


if __name__ == '__main__':
    main()

