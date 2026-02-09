"""
This module provides command-line utilities for working with Audio Spectrogram Transformer (AST) models.
It allows operations such as loading/testing model checkpoints, uploading models to Weights & Biases (wandb),
or downloading models from wandb.
"""

import argparse
import pathlib
import os
import warnings
from typing import Union

import utils
import yaml
import wandb

from AST import ASTFineTuned


def wandb_upload(checkpoint_path: Union[str, pathlib.Path], model: ASTFineTuned):
    """
    Upload the current checkpoint to W&B (not done automatically after each run because it would use all
    available storage quickly). Requires a valid W&B API key and team name in secrets.yaml.

    :param checkpoint_path: Path to the checkpoint to upload.
    :param model: Required to retrieve run info, e.g., model.custom_config['run']['wandb_id'].
    """
    with open("secrets.yaml", "r") as f:
        secrets = yaml.safe_load(f)
    with utils.measure_time(f"Uploading model checkpoint to W&B"):
        wandb.login(key=secrets['wandb']['api_key'])
        wandb_run = wandb.init(
            entity=secrets['wandb']['team'],
            project=model.custom_config['run']['project'],
            id=model.custom_config['run']['wandb_id'],
            resume='must'
        )
        checkpoint_path = pathlib.Path(checkpoint_path)
        wandb_run.save(str(checkpoint_path), base_path=checkpoint_path.parent, policy='now')
        wandb_run.finish()


def wandb_download(wandb_run_path: str, download_dir: Union[str, pathlib.Path]=None) -> pathlib.Path:
    """
    Download a checkpoint from W&B. Requires a valid W&B API key in secrets.yaml.

    :param wandb_run_path: Path to the W&B run in the format "entity/project/run-id".
    :param download_dir: Optional path to a directory where the checkpoint will be downloaded.
                       If None, uses "./wandb_downloads/entity/project/run-id" (relative to the current __file__)
    :return: Path to the downloaded checkpoint file.
    """
    # Parse the run path to extract entity, project and run ID
    parts = wandb_run_path.split('/')
    if len(parts) != 3:
        raise ValueError(f"Invalid wandb_run_path format: {wandb_run_path}. Expected format: 'entity/project/run_id'")
    entity, wandb_project, wandb_id = parts
    with open("secrets.yaml", "r") as f:
        secrets = yaml.safe_load(f)

    # Set up the target directory for downloaded data
    if download_dir is None or download_dir == '':
        current_dir = pathlib.Path(__file__).parent
        download_dir = current_dir / "wandb_downloads" / entity / wandb_project / wandb_id
        print("No target directory specified. Using default: ", download_dir, "\n(To change, specify --download-dir in the command line)")
    else:
        download_dir = pathlib.Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    # Check for the run ID in W&B (before trying to download anything)
    wandb.login(key=secrets['wandb']['api_key'])
    if secrets['wandb']['team'] != entity:
        warnings.warn(
            f"W&B team name in secrets.yaml ({secrets['wandb']['team']}) does not match the entity in the run path ({entity}).")
    wandb_run = wandb.init(entity=entity, project=wandb_project, id=wandb_id, resume='must')
    print(f'----------   {wandb_run.name=}   ----------')
    wandb_run.finish()

    with utils.measure_time(f"Downloading model checkpoint(s) from W&B (entity: {entity}, project: {wandb_project}, run ID: {wandb_id})"):
        checkpoint_file = None
        possible_checkpoints_names = ['ckpt_best.pt', 'ckpt_final.pt']
        for checkpoint_name in possible_checkpoints_names:
            try:
                checkpoint_file = wandb.restore(checkpoint_name, run_path=wandb_run_path, replace=True, root=download_dir)
                print(f"========== Checkpoint downloaded to '{checkpoint_file.name=}' ==========")
                # break # (Don't) DO keep loading other checkpoints if one was found
            except ValueError as e:
                warnings.warn(f"\nCould not find checkpoint '{checkpoint_name}' in W&B run path: {wandb_run_path}.")

        if checkpoint_file is None:
            raise FileNotFoundError(f"No checkpoint file found in {wandb_run_path=}. ({possible_checkpoints_names=})")
        abs_path = pathlib.Path(checkpoint_file.name).resolve()
        print(f"========== Absolute path: '{abs_path}' ==========")
        return abs_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AST model operations for testing checkpoints and W&B download/upload')
    parser.add_argument('--local-checkpoint', type=str,
                         help='Path to a checkpoint currently saved on disk. Triggers a model load and a model test with a dummy audio. Use --help for usage information.')
    parser.add_argument('--upload', action='store_true',
                         help='Upload the local checkpoint to W&B (not done by default after training to save W&B space). Requires --local-checkpoint.')
    parser.add_argument('--download', action='store_true',
                         help='Trigger a download from W&B. Requires --wandb-run-path. Can be used with the optional --download-dir parameter.')
    parser.add_argument('--wandb-run-path', type=str,
                         help='W&B Run Path in the format "<entity>/<project>/<run-id>". To obtain this run path from wandb.ai: click on a Run, go to the "Overview" tab, copy the "Run path". Required when --download is used.')
    parser.add_argument('--download-dir', type=str,
                         help='Directory where the checkpoint will be downloaded. Only valid when --download is used. If not specified, uses a default path based on the run path.')
    args = parser.parse_args()

    if args.local_checkpoint:
        assert not args.download, "Cannot specify both local checkpoint (--local-checkpoint) and download (--download)"
        with utils.measure_time(f"Loading model from local checkpoint: {args.local_checkpoint}"):
            _model = ASTFineTuned.from_checkpoint(args.local_checkpoint)
        _model.test_with_dummy_audio()
        # If --upload is specified, upload the checkpoint to W&B
        if args.upload:
            wandb_upload(args.local_checkpoint, _model)

    elif args.download:
        assert not args.upload, "Cannot specify both download (--download) and upload (--upload)"
        assert args.wandb_run_path, "Must specify W&B run path (--wandb-run-path) when using --download"

        checkpoint_path = wandb_download(args.wandb_run_path, args.download_dir)
        # Load the model from the downloaded checkpoint - and test it
        with utils.measure_time(f"Loading model from downloaded checkpoint: {checkpoint_path}"):
            _model = ASTFineTuned.from_checkpoint(checkpoint_path)
        _model.test_with_dummy_audio()

    elif args.upload and not args.local_checkpoint:
        print("Cannot upload to W&B without a local checkpoint. Use --help for usage information.")
    elif args.wandb_run_path and not args.download:
        print("If --wandb-run-path is provided, --download must be used as well. Use --help for usage information.")
    elif args.download_dir and not args.download:
        print("If --download-dir is provided, --download must be used as well. Use --help for usage information.")
    else:
        print("No operation specified. Use --local-checkpoint (with --upload) or --download with --wandb-run-path (and optional --download-dir). Use --help for usage information.")

    # TODO: Evaluate embeddings or perform other operations with the loaded model
