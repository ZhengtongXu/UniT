"""
AlignTac Training Workspace (单卡版)

支持:
1. 课程学习 (3阶段)
2. 单GPU训练
3. 混合精度训练
4. Wandb/Tensorboard日志
5. 模型检查点保存
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from omegaconf import OmegaConf

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from UniT.model.aligntac.aligntac_vqvae import AlignTacVQVAE
from UniT.dataset.aligntac_dataset import AlignTacDataModule
from UniT.taming.logging_util import instantiate_from_config


class CurriculumLearningCallback(pl.Callback):
    """
    课程学习回调
    
    Phase 1: RGB重建 (0 ~ phase1_epochs)
    Phase 2: +光流预测 (phase1_epochs ~ phase2_epochs)
    Phase 3: +阶段分类 (phase2_epochs ~ end)
    """
    
    def __init__(
        self,
        phase1_epochs: int = 30,
        phase2_epochs: int = 60,
    ):
        super().__init__()
        self.phase1_epochs = phase1_epochs
        self.phase2_epochs = phase2_epochs
    
    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        
        if epoch < self.phase1_epochs:
            new_phase = 1
        elif epoch < self.phase2_epochs:
            new_phase = 2
        else:
            new_phase = 3
        
        if pl_module.training_phase != new_phase:
            pl_module.set_training_phase(new_phase)
            trainer.logger.log_hyperparams({'training_phase': new_phase})


class ImageLogCallback(pl.Callback):
    """图像日志回调"""
    
    def __init__(self, log_every_n_epochs: int = 5):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            # 获取一个batch
            val_loader = trainer.val_dataloaders
            batch = next(iter(val_loader))
            
            # 生成可视化
            images = pl_module.log_images(batch)
            
            # 记录到logger
            for key, img in images.items():
                if isinstance(trainer.logger, WandbLogger):
                    import wandb
                    trainer.logger.experiment.log({
                        f"images/{key}": wandb.Image(img[:4].cpu())
                    })


def parse_args():
    parser = argparse.ArgumentParser(description='Train AlignTac VQ-VAE (单卡)')
    
    # 配置文件
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to config file')
    
    # 覆盖配置
    parser.add_argument('--data_path', type=str, default=None,
                        help='Override data path')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    
    # 课程学习
    parser.add_argument('--phase1_epochs', type=int, default=30)
    parser.add_argument('--phase2_epochs', type=int, default=60)
    parser.add_argument('--skip_curriculum', action='store_true',
                        help='Skip curriculum learning, train with all losses')
    
    # 其他
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置随机种子
    pl.seed_everything(args.seed)
    
    # 加载配置
    config = OmegaConf.load(args.config)
    
    # 命令行覆盖
    if args.data_path:
        config.data.data_path = args.data_path
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.lr:
        config.model.base_learning_rate = args.lr
    if args.epochs:
        config.trainer.max_epochs = args.epochs
    
    # 实验名称
    exp_name = args.exp_name or f"aligntac_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    OmegaConf.save(config, output_dir / 'config.yaml')
    
    print(f"[AlignTac] Output directory: {output_dir}")
    print(f"[AlignTac] Config:\n{OmegaConf.to_yaml(config)}")
    
    # ==================== 数据模块 ====================
    data_module = AlignTacDataModule(
        **config.data
    )
    data_module.setup()
    
    # ==================== 模型 ====================
    model = instantiate_from_config(config.model)
    model.learning_rate = config.model.base_learning_rate
    
    # 跳过课程学习
    if args.skip_curriculum:
        model.set_training_phase(3)
    
    # ==================== 回调 ====================
    callbacks = [
        # 模型检查点
        ModelCheckpoint(
            dirpath=output_dir / 'checkpoints',
            filename='aligntac-{epoch:03d}-{val/rgb_loss:.4f}',
            monitor='val/rgb_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
        ),
        
        # 学习率监控
        LearningRateMonitor(logging_interval='step'),
        
        # 图像日志
        ImageLogCallback(log_every_n_epochs=5),
    ]
    
    # 课程学习回调
    if not args.skip_curriculum:
        callbacks.append(
            CurriculumLearningCallback(
                phase1_epochs=args.phase1_epochs,
                phase2_epochs=args.phase2_epochs,
            )
        )
    
    # 早停 (可选)
    if config.trainer.get('early_stopping', False):
        callbacks.append(
            EarlyStopping(
                monitor='val/rgb_loss',
                patience=20,
                mode='min',
            )
        )
    
    # ==================== Logger ====================
    loggers = [
        TensorBoardLogger(
            save_dir=output_dir,
            name='tensorboard',
        )
    ]
    
    # Wandb (可选)
    if config.get('wandb', {}).get('enabled', False):
        loggers.append(
            WandbLogger(
                project=config.wandb.get('project', 'aligntac'),
                name=exp_name,
                save_dir=output_dir,
                config=OmegaConf.to_container(config),
            )
        )
    
    # ==================== Trainer (单卡) ====================
    trainer_kwargs = {
        'max_epochs': config.trainer.max_epochs,
        'accelerator': 'gpu',
        'devices': 1,  # 单卡训练
        'precision': config.trainer.get('precision', '16-mixed'),
        'callbacks': callbacks,
        'logger': loggers,
        'log_every_n_steps': 50,
        'gradient_clip_val': config.trainer.get('gradient_clip_val', 1.0),
        'accumulate_grad_batches': config.trainer.get('accumulate_grad_batches', 1),
        'check_val_every_n_epoch': config.trainer.get('check_val_every_n_epoch', 1),
    }
    
    # Debug模式
    if args.debug:
        trainer_kwargs['fast_dev_run'] = True
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # ==================== 训练 ====================
    print(f"[AlignTac] Starting training (单卡)...")
    print(f"[AlignTac] Train samples: {len(data_module.train_dataset)}")
    print(f"[AlignTac] Val samples: {len(data_module.val_dataset)}")
    
    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
        ckpt_path=args.resume,
    )
    
    print(f"[AlignTac] Training completed!")
    print(f"[AlignTac] Best model: {callbacks[0].best_model_path}")


if __name__ == '__main__':
    main()
