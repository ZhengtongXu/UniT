"""
Training Workspace for Memory-Augmented VQ-VAE (AlignTac)

This workspace handles the training of MemoryVQVAE model for dual-objective learning:
1. RGB reconstruction (geometry branch)
2. MD-Map prediction (dynamics branch with memory)

Author: Claude Code
Date: 2025-12-17
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import hydra
import pathlib
import random
import argparse
import datetime
import os
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger
import copy
import wandb

from UniT.workspace.base_workspace import BaseWorkspace
from UniT.model.aligntac.memory_vqvae import MemoryVQVAE
from UniT.dataset.mdmap_dataset import MDMapDataModule
from UniT.taming.logging_util import instantiate_from_config, config_to_parser


OmegaConf.register_new_resolver("eval", eval, replace=True)


class WandbImageLogger(Callback):
    """
    PyTorch Lightning callback for logging model-generated images to wandb

    This callback calls model.log_images() at regular intervals and logs
    the resulting images to wandb for visual monitoring of training progress.
    """

    def __init__(self, log_interval: int = 1000, max_images: int = 4):
        """
        Args:
            log_interval: Log images every N training steps
            max_images: Maximum number of images to log per batch
        """
        super().__init__()
        self.log_interval = log_interval
        self.max_images = max_images

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called after training_step, before optimizer step"""
        if trainer.global_step % self.log_interval == 0:
            self._log_images(trainer, pl_module, batch, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called after validation_step"""
        # Only log from first batch of validation
        if batch_idx == 0:
            self._log_images(trainer, pl_module, batch, split="val")

    def _log_images(self, trainer, pl_module, batch, split="train"):
        """Generate and log images to wandb"""
        # Check if model has log_images method
        if not hasattr(pl_module, 'log_images'):
            return

        # Check if wandb logger is available
        wandb_logger = None
        # PyTorch Lightning 1.x uses trainer.logger (singular)
        if hasattr(trainer, 'logger') and isinstance(trainer.logger, WandbLogger):
            wandb_logger = trainer.logger
        # Fallback to loggers (plural) for newer versions
        elif hasattr(trainer, 'loggers'):
            for logger in trainer.loggers:
                if isinstance(logger, WandbLogger):
                    wandb_logger = logger
                    break

        if wandb_logger is None:
            return

        # Generate images
        pl_module.eval()
        with torch.no_grad():
            try:
                images_dict = pl_module.log_images(batch)
            except Exception as e:
                print(f"Warning: Failed to generate log images: {e}")
                return
        pl_module.train()

        # Log to wandb
        wandb_images = {}
        for key, tensor in images_dict.items():
            # tensor shape: (N, C, H, W) in [0, 1] range
            # Convert to (N, H, W, C) for wandb
            if tensor.dim() == 4:
                tensor_np = tensor.permute(0, 2, 3, 1).cpu().numpy()  # (N, H, W, C)

                # Create wandb Image objects
                images = [wandb.Image(img) for img in tensor_np[:self.max_images]]
                wandb_images[f"{split}/{key}"] = images

        # Log all images in one call
        wandb_logger.experiment.log(wandb_images, step=trainer.global_step)


class TrainMemoryVqvaeWorkspace(BaseWorkspace):
    """
    Training workspace for Memory-Augmented VQ-VAE
    """
    
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        # Set seed for reproducibility
        seed = cfg.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        seed_everything(seed)

        # Configure GPU
        trainer_config = {}
        trainer_config["gpus"] = cfg.gpus

        # Configure MemoryVQVAE model
        print("=" * 70)
        print("Initializing Memory-Augmented VQ-VAE (AlignTac)")
        print("=" * 70)
        
        model = MemoryVQVAE(**cfg.model)
        self.model = model

        # Print model size
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model size: {total_params / 1e6:.2f}M parameters")
        print(f"Trainable: {trainable_params / 1e6:.2f}M parameters")
        
        # Print architecture summary
        print(f"\nArchitecture:")
        print(f"  Geometry embedding: {cfg.model.geo_embed_dim}")
        print(f"  Dynamics embedding: {cfg.model.dyn_embed_dim}")
        print(f"  Codebook size: {cfg.model.n_embed}")
        print(f"  Memory window K: {cfg.model.memory_K}")
        print(f"  Memory attention heads: {cfg.model.memory_num_heads}")
        print(f"\nLoss weights:")
        print(f"  RGB reconstruction: {cfg.model.lambda_rgb}")
        print(f"  MD-Map prediction: {cfg.model.lambda_mdmap}")
        print("=" * 70)

        # Setup directories
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        parser = argparse.ArgumentParser()
        parser = Trainer.add_argparse_args(parser)
        parser = config_to_parser(cfg.trainer, parser)
        opt, unknown = parser.parse_known_args()

        nowname = now + "_memory_vqvae_" + opt.postfix
        logdir = os.path.join("representation_models", nowname)
        ckptdir = os.path.join(logdir, "checkpoints")

        # Lightning config
        configs = {}
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        trainer_config_dict = dict(trainer_config)
        lightning_config.trainer = trainer_config_dict

        # Setup callbacks
        trainer_kwargs = {}
        
        # Default callbacks
        default_callbacks_cfg = {}
        
        # Periodic checkpoint callback
        default_callbacks_cfg["periodic_checkpoint"] = {
            "target": "UniT.taming.logging_util.PeriodicCheckpointCallback",
            "params": {
                "ckptdir": ckptdir,
                "save_interval": cfg.get("save_every", 1),
                "save_last_n": 3,
                "verbose": True,
            }
        }

        # Model checkpoint callback (save best model)
        default_callbacks_cfg["model_checkpoint"] = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "best-{epoch:02d}-{val/total_loss:.4f}",
                "monitor": "val/total_loss",
                "save_top_k": 3,
                "mode": "min",
                "save_last": True,
            }
        }

        # Learning rate monitor
        default_callbacks_cfg["lr_monitor"] = {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
            }
        }

        callbacks_cfg = lightning_config.get('callbacks') or OmegaConf.create()
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        # Add WandbImageLogger if wandb is enabled
        if cfg.get("wandb", {}).get("enabled", False):
            image_log_interval = cfg.wandb.get("image_log_interval", 1000)
            wandb_image_logger = WandbImageLogger(
                log_interval=image_log_interval,
                max_images=4
            )
            trainer_kwargs["callbacks"].append(wandb_image_logger)
            print(f"WandbImageLogger enabled: logging every {image_log_interval} steps")

        # Setup wandb logger if enabled
        if cfg.get("wandb", {}).get("enabled", False):
            project_name = cfg.wandb.get("project", "aligntac_memory_vqvae")
            run_name = nowname  # Use same name as checkpoint dir

            wandb_logger = WandbLogger(
                project=project_name,
                name=run_name,
                save_dir=logdir,
                log_model=False,  # Don't log model checkpoints to wandb
            )
            trainer_kwargs["logger"] = wandb_logger
            print(f"WandbLogger configured: project='{project_name}', run='{run_name}'")

        # Additional trainer kwargs
        trainer_kwargs["max_epochs"] = cfg.get("max_epochs", 100)
        trainer_kwargs["log_every_n_steps"] = cfg.get("log_every_n_steps", 100)
        trainer_kwargs["val_check_interval"] = cfg.get("val_check_interval", 1000)
        
        if "gradient_clip_val" in cfg:
            trainer_kwargs["gradient_clip_val"] = cfg.gradient_clip_val

        # Create trainer
        trainer = Trainer.from_argparse_args(
            argparse.Namespace(**trainer_config_dict),
            **trainer_kwargs
        )
        
        self.trainer = trainer
        self.lightning_config = lightning_config
        self.ckptdir = ckptdir
        self.opt = opt

    def run(self):
        """Run training"""
        cfg = copy.deepcopy(self.cfg)
        opt = self.opt

        # Setup data module
        print("\n" + "=" * 70)
        print("Setting up data module")
        print("=" * 70)
        
        data_module = MDMapDataModule(
            data_root=cfg.dataset.data_root,
            mdmap_root=cfg.dataset.mdmap_root,
            batch_size=cfg.batch_size,
            num_workers=cfg.dataloader.num_workers,
            image_size=tuple(cfg.dataset.image_size),
            mdmap_size=tuple(cfg.dataset.mdmap_size),
            val_ratio=cfg.dataset.val_ratio,
            augment=cfg.dataset.augment,
        )
        
        data_module.setup()
        
        print(f"Train dataset size: {len(data_module.train_dataset)}")
        print(f"Val dataset size: {len(data_module.val_dataset)}")
        print("=" * 70)

        # Setup learning rate
        bs = cfg.batch_size
        base_lr = cfg.base_learning_rate
        ngpu = len(self.lightning_config.trainer['gpus'].strip(",").split(','))
        accumulate_grad_batches = self.lightning_config.trainer.get('accumulate_grad_batches', 1)
        
        self.model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        
        print(f"\nLearning rate: {self.model.learning_rate:.2e}")
        print(f"  = {accumulate_grad_batches} (accumulate_grad_batches)")
        print(f"  * {ngpu} (num_gpus)")
        print(f"  * {bs} (batch_size)")
        print(f"  * {base_lr:.2e} (base_lr)")
        print("=" * 70)

        # Train model
        if opt.train:
            print("\nStarting training...")
            try:
                self.trainer.fit(
                    self.model,
                    datamodule=data_module
                )
                print("\nTraining completed!")
            except KeyboardInterrupt:
                print("\nTraining interrupted by user")
            except Exception as e:
                print(f"\nTraining failed with error: {e}")
                raise

        # Test model
        if not opt.no_test and not self.trainer.interrupted:
            print("\nRunning validation...")
            self.trainer.validate(
                self.model,
                dataloaders=data_module.val_dataloader()
            )


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    """Main entry point"""
    workspace = TrainMemoryVqvaeWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
