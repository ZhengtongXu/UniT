"""
Spatiotemporal Residual VQ-VAE 训练工作空间
============================================

训练时空残差触觉表征学习模型

"Feeling the Unexpected" (感受那份意外)

使用:
    python train.py --config-dir=./UniT/config --config-name=spatiotemporal_residual_vqvae
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
from UniT.dataset.spatiotemporal_tactile_dataset import SpatiotemporalTactileDataModule
from UniT.taming.models.spatiotemporal_residual_vqmodel import SpatiotemporalResidualVQModel
from UniT.taming.logging_util import PeriodicCheckpointCallback


class TrainSpatiotemporalWorkspace(BaseWorkspace):
    """
    训练Spatiotemporal Residual VQ-VAE的工作空间

    特性:
    - 三阶段课程学习 (重建 → VQ → FiLM)
    - force_identity参数动态切换
    - 视频可视化 (可选)
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
        print("\n" + "=" * 60)
        print("初始化Spatiotemporal Residual VQ-VAE模型...")
        print("=" * 60)
        print("核心理念: \"Feeling the Unexpected\"")
        print("  - Prophet预测预期的平滑趋势")
        print("  - Residual捕捉意外事件")
        print("  - VQ离散化为Event Code")
        print("  - Dynamic FiLM调制解码")
        print("=" * 60)

        self.model = SpatiotemporalResidualVQModel(**cfg.model)

        # 打印模型大小
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"模型参数: 总计={total_params/1e6:.2f}M, 可训练={trainable_params/1e6:.2f}M")

        # 配置Trainer
        print("\n配置PyTorch Lightning Trainer...")
        self.trainer = self._setup_trainer(cfg)

        self.cfg = cfg

    def _setup_trainer(self, cfg: OmegaConf) -> Trainer:
        """配置PyTorch Lightning Trainer"""
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

        # 2. 学习率监控
        if not cfg.trainer.debug:
            lr_monitor = LearningRateMonitor(logging_interval='step')
            callbacks.append(lr_monitor)

        # 3. 课程学习回调
        if cfg.get('curriculum_learning', {}).get('enabled', False):
            phase_configs = cfg.curriculum_learning.get('phases', [])
            if hasattr(phase_configs, '__iter__'):
                phase_configs = [OmegaConf.to_container(p, resolve=True) for p in phase_configs]

            curriculum_callback = SpatiotemporalCurriculumCallback(
                phase_configs=phase_configs,
                verbose=True
            )
            callbacks.append(curriculum_callback)
            print(f"  课程学习: 启用 ({len(phase_configs)}个阶段)")

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
        elif isinstance(cfg.gpus, ListConfig):
            gpus = OmegaConf.to_container(cfg.gpus, resolve=True)
        elif isinstance(cfg.gpus, str):
            gpus_str = cfg.gpus.strip(',')
            if gpus_str:
                gpus = [int(x) for x in gpus_str.split(',') if x.strip()]
            else:
                gpus = None
        else:
            gpus = None

        # 验证频率
        val_check_interval = int(cfg.get('validation', {}).get('val_check_interval', 5))
        print(f"  验证频率: 每{val_check_interval}个epoch验证一次")

        # 创建Trainer (PyTorch Lightning 1.9.0 API)
        trainer_kwargs = {
            'max_epochs': cfg.max_epochs,
            'callbacks': callbacks,
            'logger': logger,
            'log_every_n_steps': 50,
            'check_val_every_n_epoch': val_check_interval,
            'gradient_clip_val': 1.0,
            'precision': 32,
        }

        # 设置GPU
        if gpus:
            trainer_kwargs['gpus'] = gpus

        trainer = Trainer(**trainer_kwargs)

        return trainer

    def run(self):
        """运行训练流程"""
        cfg = copy.deepcopy(self.cfg)

        # 创建数据模块
        print("\n加载数据集...")
        datamodule = SpatiotemporalTactileDataModule(
            dataset_path=cfg.dataset.dataset_path,
            batch_size=cfg.batch_size,
            num_workers=cfg.dataloader.num_workers,
            seq_length=cfg.dataset.seq_length,
            visual_size=tuple(cfg.dataset.visual_size),
            tactile_size=tuple(cfg.dataset.tactile_size),
            normalize_robot_state=cfg.dataset.normalize_robot_state,
            val_ratio=cfg.dataset.val_ratio,
            seed=cfg.seed
        )

        # 计算学习率
        bs = cfg.batch_size
        base_lr = cfg.base_learning_rate

        # GPU数量
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
        print(f"开始训练 Spatiotemporal Residual VQ-VAE...")
        print(f"  模型: {cfg.trainer.name}")
        print(f"  Epochs: {cfg.max_epochs}")
        print(f"  Batch Size: {bs}")
        print(f"  Sequence Length: {cfg.dataset.seq_length}")
        print(f"{'='*60}\n")

        # 训练
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


class SpatiotemporalCurriculumCallback(pl.Callback):
    """
    时空架构的课程学习回调

    动态调整:
    - force_identity (Stage 1: True, Stage 2/3: False)
    - aux_loss_weight
    - hybrid_loss_weight
    - lambda_sparse
    """

    def __init__(self, phase_configs: list, verbose: bool = True):
        super().__init__()
        self.phase_configs = phase_configs
        self.verbose = verbose
        self.current_phase = None

    def on_train_epoch_start(self, trainer, pl_module):
        """在每个epoch开始时检查是否需要切换阶段"""
        current_epoch = trainer.current_epoch

        # 查找当前应该在哪个阶段
        for phase in self.phase_configs:
            if current_epoch < phase['end_epoch']:
                # 如果阶段改变, 更新参数
                if self.current_phase != phase.get('name', 'unnamed'):
                    self.current_phase = phase.get('name', 'unnamed')
                    self._update_phase(pl_module, phase, current_epoch)
                break

    def _update_phase(self, pl_module, phase: dict, epoch: int):
        """更新模型参数到新阶段"""
        # 更新Loss权重
        if 'aux_loss_weight' in phase:
            pl_module.aux_loss_weight = phase['aux_loss_weight']

        if 'hybrid_loss_weight' in phase:
            pl_module.hybrid_loss_weight = phase['hybrid_loss_weight']

        if 'lambda_sparse' in phase:
            pl_module.lambda_sparse = phase['lambda_sparse']

        # 更新force_identity (关键!)
        if 'force_identity' in phase:
            pl_module.force_identity = phase['force_identity']

        # 打印信息
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"课程学习: 切换到阶段 '{self.current_phase}' (Epoch {epoch})")
            print(f"  force_identity: {getattr(pl_module, 'force_identity', False)}")
            print(f"  aux_loss_weight: {pl_module.aux_loss_weight}")
            print(f"  hybrid_loss_weight: {pl_module.hybrid_loss_weight}")
            print(f"  lambda_sparse: {pl_module.lambda_sparse}")
            print(f"{'='*60}\n")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name='spatiotemporal_residual_vqvae'
)
def main(cfg: OmegaConf):
    """主入口"""
    workspace = TrainSpatiotemporalWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
