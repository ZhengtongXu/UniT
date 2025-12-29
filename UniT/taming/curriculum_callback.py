"""
Curriculum Learning Callback for Residual VQ-VAE Training
==========================================================

动态课程学习策略：在训练过程中自动调整超参数

支持调整的参数:
- aux_loss_weight: 辅助重建Loss权重
- lambda_sparse: 稀疏Loss权重
- z_exp_dropout: Prophet预测的dropout率 (强制VQ学习)

使用方式:
    trainer = Trainer(
        callbacks=[
            CurriculumLearningCallback(
                phase_configs=[
                    {
                        'end_epoch': 30,
                        'aux_loss_weight': 1.0,
                        'z_exp_dropout': 0.3,  # 早期强制VQ学习
                        'name': 'teacher_forcing'
                    },
                    {
                        'end_epoch': 100,
                        'aux_loss_weight': 0.1,
                        'z_exp_dropout': 0.0,  # 后期移除dropout
                        'name': 'student_adaptation'
                    }
                ]
            )
        ]
    )
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only


class CurriculumLearningCallback(Callback):
    """
    课程学习回调：根据训练阶段动态调整lambda_sparse

    Args:
        phase_configs: 阶段配置列表，每个配置包含:
            - end_epoch: 该阶段结束的epoch (exclusive)
            - lambda_sparse: 该阶段的稀疏损失权重
            - name: 阶段名称 (用于日志)
        verbose: 是否打印阶段切换信息
    """

    def __init__(self, phase_configs: list, verbose: bool = True):
        super().__init__()
        self.phase_configs = phase_configs
        self.verbose = verbose
        self.current_phase_idx = 0

        # 验证配置
        assert len(phase_configs) > 0, "至少需要一个阶段配置"
        for i, config in enumerate(phase_configs):
            assert 'end_epoch' in config, f"Phase {i}: 缺少end_epoch"
            assert 'name' in config, f"Phase {i}: 缺少name"
            # 不再强制检查 lambda_sparse，允许任意参数更新

        # 检查epoch顺序
        for i in range(1, len(phase_configs)):
            assert phase_configs[i]['end_epoch'] > phase_configs[i-1]['end_epoch'], \
                f"Phase {i}的end_epoch必须大于Phase {i-1}"

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        """训练开始时打印课程学习计划"""
        if self.verbose:
            print("\n" + "=" * 60)
            print("📚 课程学习计划 (Curriculum Learning Schedule)")
            print("=" * 60)
            for i, config in enumerate(self.phase_configs):
                start_epoch = 0 if i == 0 else self.phase_configs[i-1]['end_epoch']
                end_epoch = config['end_epoch']
                name = config['name']
                params = {k: v for k, v in config.items() if k not in ['end_epoch', 'name']}
                print(f"  Phase {i+1} [{name}]: Epochs {start_epoch}-{end_epoch-1}, Params={params}")
            print("=" * 60 + "\n")

        # 设置初始参数
        self._update_params(trainer, pl_module, 0)

    def on_train_epoch_start(self, trainer, pl_module):
        """每个epoch开始时检查是否需要切换阶段"""
        current_epoch = trainer.current_epoch

        # 检查当前epoch属于哪个阶段
        new_phase_idx = self._get_phase_index(current_epoch)

        # 如果进入新阶段，更新参数
        if new_phase_idx != self.current_phase_idx:
            self._switch_phase(trainer, pl_module, new_phase_idx, current_epoch)
            self.current_phase_idx = new_phase_idx

    def _get_phase_index(self, epoch: int) -> int:
        """根据epoch确定当前阶段索引"""
        for i, config in enumerate(self.phase_configs):
            if epoch < config['end_epoch']:
                return i
        # 如果超过最后一个阶段的end_epoch，保持最后一个阶段
        return len(self.phase_configs) - 1

    @rank_zero_only
    def _update_params(self, trainer, pl_module, phase_idx: int):
        """更新模型的参数"""
        config = self.phase_configs[phase_idx]

        updates = []
        for param_name, param_value in config.items():
            if param_name in ['end_epoch', 'name']:
                continue

            # 特殊处理: freeze_obs_encoder需要调用方法
            if param_name == 'freeze_obs_encoder' and hasattr(pl_module, 'update_freeze_status'):
                old_value = pl_module.freeze_obs_encoder
                pl_module.update_freeze_status(param_value)
                updates.append((param_name, old_value, param_value))
                continue

            if hasattr(pl_module, param_name):
                old_value = getattr(pl_module, param_name)
                setattr(pl_module, param_name, param_value)
                updates.append((param_name, old_value, param_value))

                # 同步到hparams (用于checkpoint保存)
                if hasattr(pl_module, 'hparams'):
                    setattr(pl_module.hparams, param_name, param_value)
            else:
                print(f"⚠️ 警告: 模型没有属性 '{param_name}', 无法更新 (Phase {phase_idx})")

        return updates

    @rank_zero_only
    def _switch_phase(self, trainer, pl_module, new_phase_idx: int, epoch: int):
        """切换到新阶段"""
        config = self.phase_configs[new_phase_idx]
        updates = self._update_params(trainer, pl_module, new_phase_idx)

        if self.verbose:
            print("\n" + "🔄" * 30)
            print(f"  📍 Epoch {epoch}: 进入新阶段!")
            print(f"  阶段名称: {config['name']}")
            for name, old, new in updates:
                print(f"  {name}: {old} → {new}")
            print(f"  持续到: Epoch {config['end_epoch']-1}")
            print("🔄" * 30 + "\n")

        # 记录阶段切换到wandb
        if trainer.logger is not None:
            metrics = {'curriculum/phase': new_phase_idx}
            for name, _, new in updates:
                metrics[f'curriculum/{name}'] = new
            trainer.logger.log_metrics(metrics, step=trainer.global_step)


if __name__ == "__main__":
    """测试Callback"""

    # 示例配置
    phase_configs = [
        {'end_epoch': 15, 'lambda_sparse': 0.001, 'name': 'warmup'},
        {'end_epoch': 45, 'lambda_sparse': 0.01, 'name': 'sparse'},
        {'end_epoch': 100, 'lambda_sparse': 0.05, 'name': 'finetune'}
    ]

    callback = CurriculumLearningCallback(phase_configs=phase_configs)

    # 验证阶段索引逻辑
    assert callback._get_phase_index(0) == 0  # Phase 1
    assert callback._get_phase_index(14) == 0  # Phase 1
    assert callback._get_phase_index(15) == 1  # Phase 2
    assert callback._get_phase_index(44) == 1  # Phase 2
    assert callback._get_phase_index(45) == 2  # Phase 3
    assert callback._get_phase_index(99) == 2  # Phase 3
    assert callback._get_phase_index(100) == 2  # 保持Phase 3

    print("✅ CurriculumLearningCallback测试通过!")
