#!/usr/bin/env python3
"""
恢复 v3 训练脚本
从 Epoch 100 checkpoint 继续训练，保持在同一个 wandb run
"""
import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import hydra
from omegaconf import OmegaConf
import torch

# 修复 PyTorch 2.6+ weights_only 问题 - 必须在导入其他模块前设置
import torch.serialization
from omegaconf import DictConfig, ListConfig
from omegaconf.base import ContainerMetadata, Node
import typing
torch.serialization.add_safe_globals([
    DictConfig, ListConfig, ContainerMetadata, Node,
    typing.Any
])

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb

from UniT.workspace.base_workspace import BaseWorkspace
from UniT.dataset.residual_tactile_dataset import ResidualTactileDataModule
from UniT.taming.models.residual_vqmodel import ResidualVQModel
from UniT.taming.logging_util import PeriodicCheckpointCallback, EpochImageLogger
from UniT.taming.curriculum_callback import CurriculumLearningCallback


# 配置信息
CHECKPOINT_PATH = "representation_models/2025-12-25T11-48-42residual_vqvae_v3_anti_overfit/checkpoints/checkpoint-epoch99.ckpt"
WANDB_RUN_ID = "3ijlxpfh"
WANDB_RUN_PATH = "representation_models/2025-12-25T11-48-42residual_vqvae_v3_anti_overfit"
MAX_EPOCHS = 200  # 从100延长到200


def main():
    # 加载配置
    config_path = pathlib.Path("UniT/config")
    with hydra.initialize(version_base=None, config_path=str(config_path)):
        cfg = hydra.compose(config_name="residual_vqvae_representation_v3_resume")

    print("=" * 80)
    print("恢复 Residual VQ-VAE v3 训练")
    print("=" * 80)
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  WandB Run ID: {WANDB_RUN_ID}")
    print(f"  继续训练: Epoch 100 → {MAX_EPOCHS}")
    print("=" * 80)
    print()

    # 设置随机种子
    pl.seed_everything(cfg.seed)

    # 创建数据模块
    print("加载数据集...")
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

    # 从checkpoint加载模型
    print(f"\n从checkpoint加载模型: {CHECKPOINT_PATH}")

    #直接加载checkpoint字典 (明确设置weights_only=False)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)

    # 从config创建模型
    model = ResidualVQModel(**cfg.model)

    # 加载权重
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # 恢复optimizer等状态（如果需要的话，PyTorch Lightning会自动处理）
    print(f"  ✓ 已从 Epoch {checkpoint['epoch']} 恢复")

    # 更新学习率
    ngpu = 1
    bs = cfg.batch_size
    base_lr = cfg.base_learning_rate
    model.learning_rate = ngpu * bs * base_lr
    print(f"  学习率: {model.learning_rate:.2e}")

    # 恢复 wandb logger (关键: 使用 resume='must' 和 id)
    print(f"\n恢复 WandB Run: {WANDB_RUN_ID}")
    logger = WandbLogger(
        project="residual_tactile",
        name="2025-12-25T11-48-42residual_vqvae_v3_anti_overfit",
        save_dir=WANDB_RUN_PATH,
        id=WANDB_RUN_ID,
        resume="must"  # 关键: 必须恢复这个run
    )
    print(f"  ✓ WandB logger 已恢复")

    # Callbacks
    callbacks = []

    # 1. Checkpoint callback
    ckptdir = os.path.join(WANDB_RUN_PATH, "checkpoints")
    periodic_callback = PeriodicCheckpointCallback(
        ckptdir=ckptdir,
        save_interval=5,
        save_last_n=5,
        verbose=True
    )
    callbacks.append(periodic_callback)

    # 2. 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # 3. 课程学习回调
    if cfg.curriculum_learning.enabled:
        phase_configs = [OmegaConf.to_container(p, resolve=True)
                        for p in cfg.curriculum_learning.phases]
        curriculum_callback = CurriculumLearningCallback(
            phase_configs=phase_configs,
            verbose=True
        )
        callbacks.append(curriculum_callback)
        print(f"\n课程学习: 启用 ({len(phase_configs)}个阶段)")

    # 4. 图像可视化
    image_interval = cfg.validation.image_log_interval
    image_logger = EpochImageLogger(
        epoch_frequency=int(image_interval),
        max_images=4
    )
    callbacks.append(image_logger)

    # 创建 Trainer
    print(f"\n创建 Trainer (Max Epochs: {MAX_EPOCHS})...")
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        check_val_every_n_epoch=cfg.validation.val_check_interval,
        gradient_clip_val=1.0,
        precision=32,
        accelerator='gpu',
        devices=[1],  # GPU 1
    )

    print("\n" + "=" * 80)
    print("开始恢复训练...")
    print(f"  当前 Epoch: 100")
    print(f"  目标 Epoch: {MAX_EPOCHS}")
    print(f"  剩余训练: {MAX_EPOCHS - 100} epochs")
    print("=" * 80)
    print()

    # 恢复训练
    try:
        trainer.fit(
            model,
            datamodule=datamodule,
            ckpt_path=CHECKPOINT_PATH  # PyTorch Lightning会自动恢复epoch等状态
        )
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练出错: {e}")
        raise

    print("\n" + "=" * 80)
    print("✓ 训练完成!")
    print(f"  最终 Epoch: {trainer.current_epoch}")
    print(f"  Checkpoints: {ckptdir}")
    print(f"  WandB Run: https://wandb.ai/pweiii_github/residual_tactile/runs/{WANDB_RUN_ID}")
    print("=" * 80)


if __name__ == "__main__":
    main()
