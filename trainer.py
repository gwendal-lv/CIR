import abc
import re
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

import numpy as np
import torch
import yaml

# Additional imports needed for the unified run loop
import wandb
import transformers
from tqdm import tqdm
import mergeddataset

import utils


class TrainerBase(abc.ABC):
    def __init__(self, config: Dict[str, Any], secrets_path="secrets.yaml"):
        self.config = config
        with open(secrets_path, "r") as f:
            self.secrets = yaml.safe_load(f)

        # First, sanitize the run name (will be used a dir name to store checkpoints)
        self.config['run']['name'] = utils.sanitize_file_name(config['run']['name'])
        self.run_storage_dir = self.prepare_logs_storage()  # Can modify the run name to make it unique
        self.config['run']['local_logs_dir'] = str(self.run_storage_dir.resolve())

        # Store git info (WandB doesn't do it automatically, although it is supposed to... maybe
        #  because of the conda venv?)
        self.config['git'] = utils.get_git_info()

        self.wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
        self.model: Optional[torch.nn.Module] = None

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.datasets: Dict[str, torch.utils.data.Dataset] = {}
        self.dataloaders: Dict[str, torch.utils.data.DataLoader] = {}

    def prepare_logs_storage(self) -> Path:
        """
        Prepare the storage dir.... Check if that exact combination of run and project name already exist.
        Can modify the run name to make it unique

        :return: Path to directory where all data related to the current run will be stored.
        """
        current_file_path = Path(__file__).resolve()
        project_dir = current_file_path.parent / f'logs/{self.config["run"]["project"]}'
        project_dir.mkdir(parents=True, exist_ok=True)
        new_run_name = self.find_available_run_dir_name(project_dir, self.config['run']['name'])
        if new_run_name != self.config['run']['name']:
            warnings.warn(f"\nThe storage directory for {self.config['run']['name']=} already exists.\n"
                          f"Using {new_run_name=} instead.")
            self.config['run']['name'] = new_run_name
        run_dir = project_dir / f'{self.config["run"]["name"]}'
        run_dir.mkdir(parents=False, exist_ok=False)
        print(f"========== Local logs storage: {run_dir} ==========")
        if run_dir != run_dir.resolve():
            print(f"(Corresponding resolved path: {run_dir.resolve()})")
        return run_dir

    @staticmethod
    def find_available_run_dir_name(parent_dir: Path, initial_run_name: str) -> str:
        if not (parent_dir / initial_run_name).exists():
            return initial_run_name
        # If exists already: try to Extract the number at the end of the base_name
        match = re.match(r'(.+)-(\d+)$', initial_run_name)
        if match:  # If a number is found, separate the prefix and the number
            prefix, index = match.groups()
            index = int(index)
        else:  # Otherwise, use the base_name as the prefix and start from 0
            prefix = initial_run_name
            index = 0
        # Check if the folder already exists and increment the index until an available name is found
        while True:
            new_name = f"{prefix}-{index}"
            if not (parent_dir / new_name).exists():
                return new_name
            index += 1

    def save_config(self):
        """ Saves self.config as a local .yaml file, and logs it into W&B. """
        config_yaml_path = self.run_storage_dir / 'config.yaml'
        with open(config_yaml_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        self.wandb_run.save(glob_str=config_yaml_path, base_path=self.run_storage_dir)

    def save_checkpoint(self, step_i=-1, suffix=''):
        """
            Saves the checkpoint to disk, in ./logs/{project_name}/{run_name}/ckpt{step}.pt
        """
        # Create the project/run dir if not exists... raise a warning if dir exists already
        checkpoint_path = self.run_storage_dir / f"ckpt{step_i if step_i >= 0 else ''}{suffix}.pt"
        self.model.save_checkpoint(checkpoint_path)
        print(f"========== Checkpoint saved to {checkpoint_path} ==========")

    @property
    def device(self):
        return self.config['training']['device']

    @property
    def is_classifier(self):
        return 'Classification' in self.config['model']['type']

    def init_optimizer(self):
        """ Requires self.model to be initialized first. The optimizer will be stored in self.optimizer,
         and its .step method will be called at the end of each training step (not after each epoch). """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['training']['start_lr'])

    def init_scheduler(self, warmup_n_steps_ratio=0.05, n_annealing_cycles=4):
        """ Requires self.optimizer to be initialized first. The scheduler will be stored in self.scheduler,
         and its .step method will be called at the end of each training step (not after each epoch). """

        end_start_lr_ratio = self.config['training']['end_lr'] / self.config['training']['start_lr']
        n_steps = self.config['training']['num_steps']
        n_warmup = int(warmup_n_steps_ratio * n_steps)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=end_start_lr_ratio, end_factor=1.0, total_iters=n_warmup
        )
        annealing_cycle_n_steps = int(np.ceil((n_steps - n_warmup) / n_annealing_cycles))

        if self.config['training']['scheduler'] in ['Exp', 'Exponential']:
            # FIXME possible numerical instability for large num_steps...
            gamma = np.exp((1.0 / (n_steps - n_warmup)) * np.log(end_start_lr_ratio))
            main_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

        elif self.config['training']['scheduler'] in ['Lin', 'Linear']:  # With warmup
            main_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=end_start_lr_ratio, total_iters=(n_steps - n_warmup)
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[n_warmup]
            )

        elif self.config['training']['scheduler'] in ['Cos', 'Cosine']:  # With warmup and warm restarts
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=annealing_cycle_n_steps,  # Length of each cycle
                T_mult=1,  # Keep cycles the same length
                eta_min=self.config['training']['end_lr']
            )

        else:
            raise ValueError(f"Unsupported scheduler: {self.config['training']['scheduler']}")

        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[n_warmup]
        )
        return self.scheduler

    @abc.abstractmethod
    def train_step(self, step_i: int, minibatch):
        """Perform one training step for a minibatch. Must log training metrics and step scheduler."""
        raise NotImplementedError

    @abc.abstractmethod
    def eval_performance(self, training_step_i: int, dataset_split: str = 'valid'):
        """Evaluate model on a full split (valid/test) and log metrics. """
        raise NotImplementedError

    def run(self):
        # FIXME First train the last layer alone (random initial weights will mess w/ the gradients),
        #          then train the full model
        transformers.set_seed(seed=0, deterministic=False)  # Deterministic not allowed with the current CUDA install...

        # Init logging
        wandb.login(key=self.secrets['wandb']['api_key'])
        # The valid is ignored; the "dataset" as a whole is considered small if either the train or the test is small
        _small_dataset = (self.config['dataset']['use_small_dataset']['train']
                          or self.config['dataset']['use_small_dataset']['test'])
        self.wandb_run = wandb.init(
            entity=self.secrets['wandb']['team'],
            project=self.config['run']['project'],
            name=self.config['run']['name'],
            # TODO Automatically provide all the config... ?
            config={
                "lr_start": self.config['training']['start_lr'],
                "lr_end": self.config['training']['end_lr'],
                "model_type": self.config['model']['type'],
                "pretrained": self.config['model']['pretrained'],
                "num_steps": self.config['training']['num_steps'],
                "sched": self.config['training']['scheduler'],
                "criterion": self.config['training']['criterion'],
                "batch_size": self.config['training']['batch_size'],
                "small_dataset": _small_dataset,
                "train_data_aug": self.config['dataset']['training_data_augmentation'],
            },
        )
        self.config['run']['wandb_id'] = self.wandb_run.id
        self.save_config()
        # Make some prints, to be stored in wandb (after the run has been init)
        for k, ds in self.datasets.items():
            print(f"\n{ds}")
            print(f"'{k}' dataloader: {len(self.dataloaders[k])} minibatches available")
        print("\n")

        # Init scheduler and optimizer, and move everything to the GPU
        self.model.to(self.device)
        # We train the full model directly (randomly init last layer + main transformer). Initial partial training
        # (only the last layer) has been tried but seems to have no effect.
        self.init_optimizer()
        self.init_scheduler(warmup_n_steps_ratio=self.config['training'].get('warmup_ratio', 0.0))

        # Training and validation, in steps
        training_dataloader_iter = iter(self.dataloaders['train'])
        step_i = 0  # Initialize step_i outside the loop to ensure it's always defined
        if isinstance(self.datasets['train'], mergeddataset.ContrastiveInstrumentDataset):
            print("ContrastiveInstrumentDataset: the first step will actually start after mining positives and negatives... (may take some time)")
        try:
            for step_i in tqdm(range(self.config['training']['num_steps']), total=self.config['training']['num_steps'],
                               desc="Training step", position=0):

                # Get the next mini-batch, perform 1 training step with it
                try:
                    minibatch = next(training_dataloader_iter)
                except StopIteration:  # Reset the iterator when all mini-batches have been processed
                    training_dataloader_iter = iter(self.dataloaders['train'])
                    minibatch = next(training_dataloader_iter)
                self.train_step(step_i, minibatch)

                # Validation on the entire valid dataset, periodically or after the final training step
                is_last_step = (step_i == (self.config['training']['num_steps'] - 1))
                if ((step_i > 0 and step_i % self.config['validation']['period_in_steps'] == 0)
                        or is_last_step or (step_i == 0 and self.config['validation']['valid_first_step'])):
                    self.eval_performance(step_i, dataset_split='valid')

        except KeyboardInterrupt:
            print("\nTraining interrupted by user (Ctrl-C). Performing clean stop...")
            self.eval_performance(step_i, dataset_split='valid')

        # No finally here, otherwise exceptions don't seem properly caught by the debugguer
        self.eval_performance(step_i, dataset_split='test')
        self.save_checkpoint(suffix="_final")
        self.wandb_run.finish()
