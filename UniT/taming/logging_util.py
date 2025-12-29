
import os
import hydra
import torch
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
import torchvision
from PIL import Image as PILImage
import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
import wandb
import copy

class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)


            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))


            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        raise ValueError("No way wandb")
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids)

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)

            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            PILImage.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()

                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")


class EpochImageLogger(Callback):
    """
    每N个epoch记录一次图像可视化

    Args:
        epoch_frequency: 每N个epoch记录一次
        max_images: 最多记录几张图像
        first_epoch_log_steps: 第一个epoch内记录几次可视化（默认5次）
    """
    def __init__(self, epoch_frequency=10, max_images=4, first_epoch_log_steps=5):
        super().__init__()
        self.epoch_frequency = epoch_frequency
        self.max_images = max_images
        self.first_epoch_log_steps = first_epoch_log_steps
        self.vis_batch = None  # Cached batch for consistent visualization
        self.initial_logged = False  # Track if initial visualization is done
        self.first_epoch_step_count = 0  # Track steps in first epoch
        self.total_train_steps = None  # Total steps in an epoch (for first epoch logging)

    def on_validation_epoch_start(self, trainer, pl_module):
        """
        在每个验证epoch开始时采样一个新的随机batch
        不再缓存同一个batch,以展示更多样化的验证结果
        """
        # Get validation dataloader
        val_dataloader = trainer.val_dataloaders if trainer.val_dataloaders is not None else trainer.val_dataloader
        if isinstance(val_dataloader, list):
            val_dataloader = val_dataloader[0]

        # Sample a random batch index
        import random
        num_batches = len(val_dataloader)
        # 限制在较小的范围内，避免遍历整个dataloader太慢
        max_idx = min(num_batches - 1, 20)
        random_idx = random.randint(0, max_idx)

        # Fetch that specific batch
        for i, batch in enumerate(val_dataloader):
            if i == random_idx:
                self.vis_batch = batch
                break

    def _log_batch(self, pl_module, batch, split, epoch):
        """辅助方法：记录单个batch的图像"""
        # 移动到设备
        if isinstance(batch, dict):
            batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
        else:
            batch = batch.to(pl_module.device)

        # 生成图像
        with torch.no_grad():
            images = pl_module.log_images(batch)

        # 限制图像数量并智能归一化
        for k in images:
            batch_size = images[k].shape[0]
            if batch_size > self.max_images:
                # 使用均匀间隔采样
                indices = torch.linspace(0, batch_size - 1, self.max_images).long().to(images[k].device)
                images[k] = images[k][indices]
            
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()

                # 智能归一化：根据实际范围判断
                img_min, img_max = images[k].min().item(), images[k].max().item()

                if 'visual' in k:
                    # Visual使用ImageNet标准化，范围约[-2, +2]
                    # 需要反归一化
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                    images[k] = images[k] * std + mean  # 恢复到[0,1]
                    images[k] = torch.clamp(images[k], 0., 1.)

                elif img_min >= -0.1 and img_max <= 1.1:
                    # 已经在[0, 1]范围，无需变换
                    images[k] = torch.clamp(images[k], 0., 1.)

                elif img_min >= -1.1 and img_max <= 1.1:
                    # 在[-1, 1]范围，转换到[0, 1]
                    images[k] = torch.clamp(images[k], -1., 1.)
                    images[k] = (images[k] + 1.0) / 2.0

                else:
                    # 未知范围，使用min-max归一化
                    images[k] = (images[k] - img_min) / (img_max - img_min + 1e-8)

        # 保存到本地
        if hasattr(pl_module.logger, 'save_dir'):
            save_dir = pl_module.logger.save_dir
            root = os.path.join(save_dir, "images", split)
            os.makedirs(root, exist_ok=True)

            for k in images:
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)

                # 特殊处理不同的epoch标记
                if epoch == -1:
                    # 初始可视化（未训练）
                    filename = f"{k}_initial.png"
                elif 0 < epoch < 1:
                    # 第一个epoch内的可视化（epoch=0.1, 0.2, ...）
                    step_num = int(epoch * 10)
                    filename = f"{k}_epoch0-step{step_num}.png"
                else:
                    # 正常的epoch可视化
                    filename = f"{k}_epoch-{int(epoch):04d}.png"
                path = os.path.join(root, filename)
                PILImage.fromarray(grid).save(path)

        # 记录到wandb
        if isinstance(pl_module.logger, pl.loggers.WandbLogger):
            wandb_images = {}
            for k in images:
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                wandb_images[f"{split}/{k}"] = wandb.Image(grid)

            pl_module.logger.experiment.log(wandb_images, step=pl_module.global_step)
            print(f"Epoch {epoch} ({split}): 记录了 {len(images)} 张可视化图像到wandb")


    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        """在训练开始时记录初始可视化（未训练的模型）"""
        if self.initial_logged:
            return

        # 检查模型是否有log_images方法
        if not hasattr(pl_module, "log_images") or not callable(pl_module.log_images):
            return

        print("记录初始可视化（未训练的模型）...")

        # 获取训练集步数，用于第一个epoch的可视化
        try:
            train_dataloader = None
            if hasattr(trainer, "train_dataloader") and trainer.train_dataloader is not None:
                train_dataloader = trainer.train_dataloader
            elif hasattr(trainer, "datamodule") and trainer.datamodule is not None:
                train_dataloader = trainer.datamodule.train_dataloader()

            if train_dataloader is not None:
                self.total_train_steps = len(train_dataloader)
                print(f"第一个epoch将在 {self.first_epoch_log_steps} 个位置记录可视化（总步数: {self.total_train_steps}）")
        except:
            pass

        # 切换到eval模式
        was_training = pl_module.training
        if was_training:
            pl_module.eval()

        # 获取验证集的一个batch
        try:
            val_dataloader = trainer.val_dataloaders if trainer.val_dataloaders is not None else trainer.val_dataloader
            if isinstance(val_dataloader, list):
                val_dataloader = val_dataloader[0]

            if val_dataloader is not None:
                val_batch = next(iter(val_dataloader))
                self._log_batch(pl_module, val_batch, "val", -1)  # Use epoch=-1 for initial
        except Exception as e:
            print(f"无法记录初始验证集图像: {e}")

        # 获取训练集的一个batch
        try:
            train_dataloader = None
            if hasattr(trainer, "train_dataloader") and trainer.train_dataloader is not None:
                train_dataloader = trainer.train_dataloader
            elif hasattr(trainer, "datamodule") and trainer.datamodule is not None:
                train_dataloader = trainer.datamodule.train_dataloader()

            if train_dataloader is not None:
                train_batch = next(iter(train_dataloader))
                self._log_batch(pl_module, train_batch, "train", -1)  # Use epoch=-1 for initial
        except Exception as e:
            print(f"无法记录初始训练集图像: {e}")

        # 恢复训练模式
        if was_training:
            pl_module.train()

        self.initial_logged = True
        print("初始可视化记录完成")

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """在第一个epoch的训练批次中记录可视化"""
        # 仅在第一个epoch进行
        if trainer.current_epoch != 0:
            return

        # 检查模型是否有log_images方法
        if not hasattr(pl_module, "log_images") or not callable(pl_module.log_images):
            return

        # 如果没有总步数信息，直接返回
        if self.total_train_steps is None:
            return

        # 计算应该记录的步数位置（均匀分布）
        log_steps = [int(i * self.total_train_steps / self.first_epoch_log_steps)
                     for i in range(1, self.first_epoch_log_steps + 1)]

        # 检查当前步数是否应该记录
        if batch_idx not in log_steps:
            return

        # 计算是第几次记录（用于文件命名）
        log_index = log_steps.index(batch_idx) + 1

        print(f"记录第一个epoch的第 {log_index}/{self.first_epoch_log_steps} 次可视化（步数: {batch_idx}/{self.total_train_steps}）...")

        # 切换到eval模式
        was_training = pl_module.training
        if was_training:
            pl_module.eval()

        # 使用特殊的epoch标记（0.x表示第一个epoch的第x次记录）
        epoch_marker = 0.0 + log_index / 10.0  # 0.1, 0.2, 0.3, 0.4, 0.5

        # 记录验证集图像
        try:
            if self.vis_batch is not None:
                self._log_batch(pl_module, self.vis_batch, "val", epoch_marker)
        except Exception as e:
            print(f"无法记录验证集图像: {e}")

        # 记录训练集图像（使用当前batch）
        try:
            self._log_batch(pl_module, batch, "train", epoch_marker)
        except Exception as e:
            print(f"无法记录训练集图像: {e}")

        # 恢复训练模式
        if was_training:
            pl_module.train()

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        """在验证epoch结束时记录图像"""
        epoch = trainer.current_epoch

        # 检查是否应该记录
        if (epoch + 1) % self.epoch_frequency != 0:
            return

        # 检查模型是否有log_images方法
        if not hasattr(pl_module, "log_images") or not callable(pl_module.log_images):
            return

        # 切换到eval模式
        was_training = pl_module.training
        if was_training:
            pl_module.eval()

        # 1. 记录验证集图像
        if self.vis_batch is not None:
            self._log_batch(pl_module, self.vis_batch, "val", epoch)

        # 2. 记录训练集图像 (采样一个新的batch)
        try:
            # 尝试获取训练dataloader
            train_dataloader = None
            if hasattr(trainer, "train_dataloader") and trainer.train_dataloader is not None:
                train_dataloader = trainer.train_dataloader
            elif hasattr(trainer, "datamodule") and trainer.datamodule is not None:
                train_dataloader = trainer.datamodule.train_dataloader()

            if train_dataloader is not None:
                # 获取一个batch (这里假设是Iterable)
                # 注意: 这里会触发一次数据加载，可能会比较慢
                train_batch = next(iter(train_dataloader))
                self._log_batch(pl_module, train_batch, "train", epoch)

        except Exception as e:
            print(f"无法记录训练集图像: {e}")

        # 恢复训练模式
        if was_training:
            pl_module.train()


class PeriodicCheckpointCallback(Callback):
    def __init__(self, ckptdir, save_interval=5, save_last_n=3, verbose=True):
        super().__init__()
        self.ckptdir = ckptdir
        self.save_interval = save_interval
        self.save_last_n = save_last_n
        self.verbose = verbose
        self.checkpoints = []

    def on_train_epoch_end(self, trainer, pl_module, outputs=None):
        """
        PyTorch Lightning 1.0.8 passes outputs parameter, newer versions don't
        Make it optional for compatibility
        """
        epoch = trainer.current_epoch
        if (epoch + 1) % self.save_interval == 0:
            filepath = os.path.join(self.ckptdir, f"checkpoint-epoch{epoch}.ckpt")
            trainer.save_checkpoint(filepath)
            self.checkpoints.append(filepath)

            while len(self.checkpoints) > self.save_last_n:
                oldest_ckpt = self.checkpoints.pop(0)
                try:
                    os.remove(oldest_ckpt)
                    if self.verbose:
                        print(f"Deleted old checkpoint: {oldest_ckpt}")
                except OSError:
                    pass

            if self.verbose:
                print(f"Checkpoint saved: {filepath}")


def config_to_parser(config, parser):
    # Iterate over each key-value pair in the dictionary
    for key, value in config.items():
        if f'--{key}' in parser._option_string_actions:
            del parser._option_string_actions[f'--{key}']
        # Add the argument to the parser
        if isinstance(value, bool):
            parser.add_argument(f'--{key}', action='store', default=value)
        elif isinstance(value, list):
            parser.add_argument(f'--{key}', nargs='*', default=value)
        else:
            parser.add_argument(f'--{key}', type=type(value), default=value)
            #pass
    return parser

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))