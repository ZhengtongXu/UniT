"""
Residual VQ-VAE 训练工作空间
=====================================

训练Residual Tactile Representation Learning模型

使用:
    python UniT/workspace/train_residual_vqvae_workspace.py

    # 或指定配置
    python UniT/workspace/train_residual_vqvae_workspace.py \
        model.lambda_sparse=0.001 \
        batch_size=32
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
import datetime
import copy
import os
from omegaconf import OmegaConf, ListConfig
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from UniT.workspace.base_workspace import BaseWorkspace
from UniT.dataset.residual_tactile_dataset import ResidualTactileDataModule
from UniT.taming.models.residual_vqmodel import ResidualVQModel
from UniT.taming.logging_util import PeriodicCheckpointCallback, EpochImageLogger
from UniT.taming.curriculum_callback import CurriculumLearningCallback


class TrainResidualVqvaeWorkspace(BaseWorkspace):
    """
    训练Residual VQ-VAE的工作空间

    复用PyTorch Lightning + Hydra框架
    """

    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        # 设置随机种子
        seed = cfg.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        pl.seed_everything(seed)

        # 配置模型
        print("\n初始化Residual VQ-VAE模型...")
        self.model = ResidualVQModel(**cfg.model)

        # 打印模型大小
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"模型参数: 总计={total_params/1e6:.2f}M, 可训练={trainable_params/1e6:.2f}M")

        # 配置Trainer
        print("\n配置PyTorch Lightning Trainer...")
        self.trainer = self._setup_trainer(cfg)

        self.cfg = cfg

    def _setup_trainer(self, cfg: OmegaConf) -> Trainer:
        """
        配置PyTorch Lightning Trainer

        Args:
            cfg: Hydra配置对象
        Returns:
            trainer: 配置好的Trainer
        """
        # 时间戳
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

        # 日志目录
        nowname = now + cfg.trainer.name + cfg.trainer.postfix
        logdir = os.path.join("representation_models", nowname)
        ckptdir = os.path.join(logdir, "checkpoints")

        os.makedirs(ckptdir, exist_ok=True)
        print(f"  Checkpoint目录: {ckptdir}")

        # Callbacks
        callbacks = []

        # 1. 定期保存checkpoint
        save_interval = cfg.get('validation', {}).get('checkpoint_save_interval', cfg.trainer.save_every)
        keep_last_n = cfg.get('validation', {}).get('keep_last_n_checkpoints', 3)

        periodic_callback = PeriodicCheckpointCallback(
            ckptdir=ckptdir,
            save_interval=save_interval,
            save_last_n=keep_last_n,
            verbose=True
        )
        callbacks.append(periodic_callback)

        # 2. 学习率监控 (仅在有logger时添加)
        if not cfg.trainer.debug:
            lr_monitor = LearningRateMonitor(logging_interval='step')
            callbacks.append(lr_monitor)

        # 3. 课程学习回调 (Curriculum Learning)
        if cfg.get('curriculum_learning', {}).get('enabled', False):
            phase_configs = cfg.curriculum_learning.get('phases', [
                {'end_epoch': 15, 'lambda_sparse': 0.001, 'name': 'warmup'},
                {'end_epoch': 45, 'lambda_sparse': 0.01, 'name': 'sparse'},
                {'end_epoch': 100, 'lambda_sparse': 0.05, 'name': 'finetune'}
            ])
            # 转换OmegaConf为普通列表
            if hasattr(phase_configs, '__iter__'):
                phase_configs = [OmegaConf.to_container(p, resolve=True) for p in phase_configs]

            curriculum_callback = CurriculumLearningCallback(
                phase_configs=phase_configs,
                verbose=True
            )
            callbacks.append(curriculum_callback)
            print(f"  课程学习: 启用 ({len(phase_configs)}个阶段)")

        # 4. 图像可视化回调 (与验证频率对齐)
        if not cfg.trainer.debug:
            # Get image logging interval from validation config, fallback to val_check_interval
            val_interval = cfg.get('validation', {}).get('val_check_interval', 3.0)
            image_interval = cfg.get('validation', {}).get('image_log_interval', val_interval)

            image_logger = EpochImageLogger(
                epoch_frequency=int(image_interval),  # Same as validation frequency
                max_images=4
            )
            callbacks.append(image_logger)
            print(f"  图像可视化: 每{int(image_interval)}个epoch记录一次 (与验证频率对齐)")

        # Logger (wandb)
        logger = None
        if not cfg.trainer.debug:
            logger = WandbLogger(
                project=cfg.trainer.project,
                name=nowname,
                save_dir=logdir
            )
            print(f"  Wandb项目: {cfg.trainer.project}")

        # GPU配置
        if isinstance(cfg.gpus, int):
            gpus = [cfg.gpus] if cfg.gpus >= 0 else None
            strategy = None
        elif isinstance(cfg.gpus, ListConfig):
            gpus = OmegaConf.to_container(cfg.gpus, resolve=True)
            strategy = 'ddp' if len(gpus) > 1 else None
        elif isinstance(cfg.gpus, str):
            gpus_str = cfg.gpus.strip(',')
            if gpus_str:
                gpus = [int(x) for x in gpus_str.split(',') if x.strip()]
                strategy = 'ddp' if len(gpus) > 1 else None
            else:
                gpus = None
                strategy = None
        else:
            gpus = None
            strategy = None

        # Get validation interval from config
        val_check_interval = int(cfg.get('validation', {}).get('val_check_interval', 3))
        print(f"  验证频率: 每{val_check_interval}个epoch验证一次")

        # 创建Trainer (使用PyTorch Lightning 1.9 API)
        trainer_kwargs = {
            'max_epochs': cfg.max_epochs,
            'callbacks': callbacks,
            'logger': logger,
            'log_every_n_steps': 50,
            'check_val_every_n_epoch': val_check_interval,  # 从配置读取验证频率
            'gradient_clip_val': 1.0,   # 梯度裁剪
            'precision': 32,            # 使用FP32 (可改为16使用混合精度)
        }

        # 设置GPU配置 (使用新的accelerator/devices API)
        if gpus:
            trainer_kwargs['accelerator'] = 'gpu'
            trainer_kwargs['devices'] = gpus
            if strategy == 'ddp':
                trainer_kwargs['strategy'] = 'ddp'

        trainer = Trainer(**trainer_kwargs)

        return trainer

    def run(self):
        """运行训练流程"""
        cfg = copy.deepcopy(self.cfg)

        # 创建数据模块
        print("\n加载数据集...")
        datamodule = ResidualTactileDataModule(
            dataset_path=cfg.dataset.dataset_path,
            batch_size=cfg.batch_size,
            num_workers=cfg.dataloader.num_workers,
            visual_size=tuple(cfg.dataset.visual_size),
            tactile_size=tuple(cfg.dataset.tactile_size),
            normalize_robot_state=cfg.dataset.normalize_robot_state,
            val_ratio=cfg.dataset.val_ratio,
            seed=cfg.seed
        )

        # 计算学习率
        bs = cfg.batch_size
        base_lr = cfg.base_learning_rate

        # 计算GPU数量
        if isinstance(cfg.gpus, int):
            ngpu = 1 if cfg.gpus >= 0 else 1
        elif isinstance(cfg.gpus, str):
            gpus_str = cfg.gpus.strip(',')
            ngpu = len([x for x in gpus_str.split(',') if x.strip()]) if gpus_str else 1
        else:
            ngpu = 1

        self.model.learning_rate = ngpu * bs * base_lr

        print(f"\n学习率设置:")
        print(f"  {self.model.learning_rate:.2e} = "
              f"{ngpu} (GPUs) × {bs} (batch) × {base_lr:.2e} (base_lr)")

        print(f"\n{'='*60}")
        print(f"开始训练...")
        print(f"  模型: {cfg.trainer.name}")
        print(f"  Epochs: {cfg.max_epochs}")
        print(f"  Lambda_sparse: {cfg.model.lambda_sparse}")
        print(f"  预训练模型: {cfg.model.vq_model_ckpt}")
        print(f"{'='*60}\n")

        # 训练 (PL 1.0 uses train_dataloader/val_dataloaders, later versions use datamodule)
        try:
            self.trainer.fit(
                self.model,
                datamodule=datamodule
            )

        except KeyboardInterrupt:
            print("\n训练被用户中断")
        except Exception as e:
            print(f"\n训练出错: {e}")
            raise

        # 测试
        if not cfg.trainer.no_test and not self.trainer.interrupted:
            print("\n运行测试...")
            self.trainer.test(
                self.model,
                datamodule=datamodule
            )

        print(f"\n{'='*60}")
        print(f"✓ 训练完成!")
        print(f"  Checkpoints: {self.trainer.log_dir}/checkpoints")
        print(f"{'='*60}")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name='residual_vqvae_representation'
)
def main(cfg: OmegaConf):
    """主入口"""
    workspace = TrainResidualVqvaeWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
