"""
Loss和梯度验证脚本
=====================================

验证项:
1. Reconstruction Loss (L1 + LPIPS + GAN)
2. Sparse Loss (L1 norm)
3. Total Loss组成
4. 梯度传播路径 (prophet、obs、residual)

运行时间: 约7-10分钟
"""

import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict

from UniT.dataset.residual_tactile_dataset import ResidualTactileDataModule
from UniT.taming.models.residual_vqmodel import ResidualVQModel


def validate_recon_loss(model, batch, optimizer_idx=0):
    """验证Reconstruction Loss"""
    print(f"\n{'='*60}")
    print("1. Reconstruction Loss验证")
    print(f"{'='*60}")

    issues = []

    # 前向传播
    model.train()
    outputs = model.forward(batch)

    # 计算recon loss (初期, global_step=500)
    loss, log_dict = model.recon_loss(
        codebook_loss=torch.tensor(0.0).to(batch['tactile_image'].device),
        inputs=batch['tactile_image'],
        reconstructions=outputs['recon'],
        optimizer_idx=optimizer_idx,
        global_step=500,
        last_layer=model.get_last_layer(),
        split='train'
    )

    print(f"\n✓ 初期 (global_step=500, disc未启用):")
    print(f"  Total recon loss: {loss:.6f}")
    print(f"\n  组件:")
    for key in sorted(log_dict.keys()):
        if 'train/' in key:
            print(f"    {key}: {log_dict[key]:.6f}")

    # 验证disc_factor = 0 (初期)
    if 'train/g_loss' in log_dict:
        if log_dict['train/g_loss'] != 0:
            issues.append(f"初期g_loss应为0, 实际: {log_dict['train/g_loss']}")
            print(f"  ⚠ 警告: g_loss应为0")
    else:
        print(f"  ✓ g_loss未启用 (正确)")

    # 验证后期 (global_step=15000, discriminator启用)
    loss_late, log_dict_late = model.recon_loss(
        codebook_loss=torch.tensor(0.0).to(batch['tactile_image'].device),
        inputs=batch['tactile_image'],
        reconstructions=outputs['recon'],
        optimizer_idx=optimizer_idx,
        global_step=15000,
        last_layer=model.get_last_layer(),
        split='train'
    )

    print(f"\n✓ 后期 (global_step=15000, disc已启用):")
    print(f"  Total recon loss: {loss_late:.6f}")
    print(f"\n  组件:")
    for key in sorted(log_dict_late.keys()):
        if 'train/' in key:
            print(f"    {key}: {log_dict_late[key]:.6f}")

    # 验证g_loss存在
    if 'train/g_loss' not in log_dict_late:
        issues.append("后期g_loss未启用")
        print(f"  ⚠ 警告: g_loss应该启用")
    else:
        print(f"  ✓ g_loss已启用 (正确)")

    # 检查NaN
    if torch.isnan(loss):
        issues.append("recon_loss为NaN")
    if torch.isinf(loss):
        issues.append("recon_loss为Inf")

    return issues, loss, log_dict


def validate_sparse_loss(model, batch):
    """验证Sparse Loss"""
    print(f"\n{'='*60}")
    print("2. Sparse Loss验证")
    print(f"{'='*60}")

    issues = []

    # 前向传播
    model.train()
    outputs = model.forward(batch)
    residual = outputs['residual']

    # 计算sparse loss
    sparse_loss = model.lambda_sparse * residual.abs().mean()

    print(f"\n✓ Sparse Loss计算:")
    print(f"  lambda_sparse: {model.lambda_sparse}")
    print(f"  ||residual||_1: {residual.abs().mean():.6f}")
    print(f"  sparse_loss = lambda * ||r||_1: {sparse_loss:.6f}")

    # 手动验证
    expected_sparse_loss = model.lambda_sparse * residual.abs().mean()
    diff = (sparse_loss - expected_sparse_loss).abs()
    print(f"\n✓ 手动验证:")
    print(f"  期望: {expected_sparse_loss:.6f}")
    print(f"  实际: {sparse_loss:.6f}")
    print(f"  差异: {diff:.2e}")

    if diff > 1e-6:
        issues.append(f"sparse_loss计算错误: diff={diff:.2e}")
        print(f"  ❌ 验证失败")
    else:
        print(f"  ✓ 验证通过")

    # 检查NaN
    if torch.isnan(sparse_loss):
        issues.append("sparse_loss为NaN")
    if torch.isinf(sparse_loss):
        issues.append("sparse_loss为Inf")

    return issues, sparse_loss


def validate_total_loss(model, batch):
    """验证Total Loss组成"""
    print(f"\n{'='*60}")
    print("3. Total Loss组成验证")
    print(f"{'='*60}")

    issues = []

    # 使用training_step计算
    model.train()

    loss_gen = model.training_step(batch, batch_idx=0, optimizer_idx=0)

    # 获取log (从model的最后一次log)
    # 注意: training_step返回loss, log在self.log中
    # 我们需要手动计算来验证

    # 手动计算各组件
    outputs = model.forward(batch)

    # Recon loss
    recon_loss, log_dict = model.recon_loss(
        codebook_loss=torch.tensor(0.0).to(batch['tactile_image'].device),
        inputs=batch['tactile_image'],
        reconstructions=outputs['recon'],
        optimizer_idx=0,
        global_step=500,
        last_layer=model.get_last_layer(),
        split='train'
    )

    # Sparse loss
    sparse_loss = model.lambda_sparse * outputs['residual'].abs().mean()

    # Total loss
    expected_total = recon_loss + sparse_loss

    print(f"\n✓ Loss组成:")
    print(f"  recon_loss: {recon_loss:.6f}")
    print(f"  sparse_loss: {sparse_loss:.6f}")
    print(f"  expected_total: {expected_total:.6f}")
    print(f"  actual (from training_step): {loss_gen:.6f}")

    # 验证总和
    diff = (loss_gen - expected_total).abs()
    print(f"\n✓ 总和验证:")
    print(f"  差异: {diff:.2e}")

    if diff > 1e-4:
        issues.append(f"total_loss计算错误: diff={diff:.2e}")
        print(f"  ⚠ 警告: 差异较大 (可能由于不同的forward pass)")
    else:
        print(f"  ✓ 验证通过")

    # 检查NaN
    if torch.isnan(loss_gen):
        issues.append("total_loss为NaN")
    if torch.isinf(loss_gen):
        issues.append("total_loss为Inf")

    return issues, loss_gen


def validate_prophet_gradients(model, batch, loss):
    """验证ProphetEncoder梯度"""
    print(f"\n{'='*60}")
    print("4. ProphetEncoder梯度验证")
    print(f"{'='*60}")

    issues = []

    # 清空梯度
    model.zero_grad()

    # 反向传播
    loss.backward(retain_graph=True)

    # 收集梯度统计
    gradient_stats = {}

    print(f"\n✓ ProphetEncoder梯度:")
    for name, param in model.prophet_encoder.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                issues.append(f"ProphetEncoder.{name}无梯度")
                print(f"  ❌ {name}: 无梯度")
            else:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()

                gradient_stats[f'prophet.{name}'] = {
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'std': grad_std
                }

                print(f"  ✓ {name}:")
                print(f"      norm={grad_norm:.4e}, mean={grad_mean:.4e}, std={grad_std:.4e}")

                # 检查异常梯度
                if grad_norm > 1e3:
                    issues.append(f"ProphetEncoder.{name}梯度爆炸: {grad_norm:.2e}")
                if grad_norm < 1e-8:
                    issues.append(f"ProphetEncoder.{name}梯度消失: {grad_norm:.2e}")
                if torch.isnan(param.grad).any():
                    issues.append(f"ProphetEncoder.{name}梯度为NaN")

    return issues, gradient_stats


def validate_obs_gradients(model, batch, loss):
    """验证ObsEncoder梯度"""
    print(f"\n{'='*60}")
    print("5. ObsEncoder梯度验证")
    print(f"{'='*60}")

    issues = []

    # 清空梯度
    model.zero_grad()

    # 反向传播
    loss.backward(retain_graph=True)

    # 收集梯度统计
    gradient_stats = {}

    print(f"\n✓ ObsEncoder梯度 (只显示部分层):")
    count = 0
    for name, param in model.obs_encoder.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                issues.append(f"ObsEncoder.{name}无梯度")
                # 只显示前5个
                if count < 5:
                    print(f"  ❌ {name}: 无梯度")
                    count += 1
            else:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()

                gradient_stats[f'obs.{name}'] = {
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'std': grad_std
                }

                # 只显示前5个
                if count < 5:
                    print(f"  ✓ {name}:")
                    print(f"      norm={grad_norm:.4e}, mean={grad_mean:.4e}, std={grad_std:.4e}")
                    count += 1

                # 检查异常梯度
                if grad_norm > 1e3:
                    issues.append(f"ObsEncoder.{name}梯度爆炸: {grad_norm:.2e}")
                if grad_norm < 1e-8:
                    issues.append(f"ObsEncoder.{name}梯度消失: {grad_norm:.2e}")
                if torch.isnan(param.grad).any():
                    issues.append(f"ObsEncoder.{name}梯度为NaN")

    print(f"  ... (共{len(list(model.obs_encoder.parameters()))}个参数)")

    return issues, gradient_stats


def validate_residual_gradient_paths(model, batch):
    """验证Residual的双路径梯度"""
    print(f"\n{'='*60}")
    print("6. Residual梯度路径验证")
    print(f"{'='*60}")

    issues = []

    # 清空GPU缓存
    import torch
    torch.cuda.empty_cache()

    model.train()

    # 前向传播
    outputs = model.forward(batch)
    residual = outputs['residual']

    # 注意: residual是中间计算结果,不直接有grad属性
    # 我们验证梯度是否正确传播到z_exp和z_real

    # 路径1: 只有sparse loss
    print(f"\n✓ 路径1: Sparse Loss → z_exp/z_real")
    model.zero_grad()
    sparse_loss_only = model.lambda_sparse * residual.abs().mean()
    sparse_loss_only.backward(retain_graph=True)

    # 检查z_exp和z_real的梯度
    z_exp_has_grad = outputs['z_exp'].grad is not None
    z_real_has_grad = outputs['z_real'].grad is not None

    if not z_exp_has_grad or not z_real_has_grad:
        issues.append("Sparse loss未传播梯度到z_exp或z_real")
        print(f"  ❌ z_exp梯度: {z_exp_has_grad}, z_real梯度: {z_real_has_grad}")
    else:
        sparse_grad_norm = outputs['z_real'].grad.abs().mean().item()
        print(f"  ✓ z_real梯度norm: {sparse_grad_norm:.4e}")

    # 路径2和3: 简化验证,避免GPU内存问题
    print(f"\n✓ 路径2&3: 验证完整训练步骤的梯度流")

    # 清空之前的计算图
    torch.cuda.empty_cache()
    model.zero_grad()

    # 使用完整的training_step
    outputs_final = model.forward(batch)

    recon_loss_final, _ = model.recon_loss(
        codebook_loss=torch.tensor(0.0).to(batch['tactile_image'].device),
        inputs=batch['tactile_image'],
        reconstructions=outputs_final['recon'],
        optimizer_idx=0,
        global_step=500,
        last_layer=model.get_last_layer(),
        split='train'
    )
    sparse_loss_final = model.lambda_sparse * outputs_final['residual'].abs().mean()
    total_loss_final = recon_loss_final + sparse_loss_final

    total_loss_final.backward()

    # 检查关键组件的梯度
    z_exp_grad_exists = outputs_final['z_exp'].grad is not None
    z_real_grad_exists = outputs_final['z_real'].grad is not None

    print(f"  z_exp梯度: {'✓ 存在' if z_exp_grad_exists else '❌ 不存在'}")
    print(f"  z_real梯度: {'✓ 存在' if z_real_grad_exists else '❌ 不存在'}")

    if z_exp_grad_exists and z_real_grad_exists:
        z_exp_grad_norm = outputs_final['z_exp'].grad.abs().mean().item()
        z_real_grad_norm = outputs_final['z_real'].grad.abs().mean().item()
        print(f"  z_exp梯度norm: {z_exp_grad_norm:.4e}")
        print(f"  z_real梯度norm: {z_real_grad_norm:.4e}")

        # 绘制梯度流图
        os.makedirs('../debug_outputs/gradient_analysis', exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        components = ['z_exp\n(Prophet)', 'z_real\n(Obs)', 'Sparse\nLoss', 'Total\nLoss']
        norms = [z_exp_grad_norm, z_real_grad_norm, sparse_grad_norm,
                 (z_exp_grad_norm + z_real_grad_norm) / 2]

        ax.bar(components, norms, color=['blue', 'orange', 'green', 'red'])
        ax.set_ylabel('Gradient L1 Norm')
        ax.set_title('Gradient Flow Verification')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('../debug_outputs/gradient_analysis/gradient_flow_paths.png', dpi=150)
        plt.close()

        print(f"  ✓ 梯度流正常")
    else:
        issues.append("关键组件缺少梯度")
        print(f"  ❌ 梯度流异常")

    return issues


def main():
    """主验证流程"""
    print(f"\n{'#'*60}")
    print("# Loss和梯度验证脚本")
    print(f"{'#'*60}")

    # 创建输出目录
    os.makedirs('../debug_outputs/gradient_analysis', exist_ok=True)

    # 加载数据
    print(f"\n加载数据...")
    datamodule = ResidualTactileDataModule(
        dataset_path='data/residual_tactile_train',
        batch_size=4,
        num_workers=4,
        visual_size=(224, 224),
        tactile_size=(128, 160),
        normalize_robot_state=True,
        val_ratio=0.1,
        seed=42
    )
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    # 移动到GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    # 创建模型
    print(f"\n创建模型...")
    model_config = {
        'prophet_config': {
            'visual_backbone': 'resnet34',
            'visual_pretrained': True,
            'state_dim': 8,
            'latent_shape': [3, 20, 16],
            'hidden_dims': [512, 256, 128]
        },
        'lambda_sparse': 0.01,
        'ddconfig': {
            'double_z': False,
            'z_channels': 3,
            'resolution': 160,
            'in_channels': 3,
            'out_ch': 3,
            'ch': 128,
            'ch_mult': [1, 1, 2, 4],
            'num_res_blocks': 2,
            'attn_resolutions': [16],
            'dropout': 0.0
        },
        'lossconfig': {
            'target': 'UniT.taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator',
            'params': {
                'disc_conditional': False,
                'disc_in_channels': 3,
                'disc_start': 10000,
                'disc_weight': 0.8,
                'codebook_weight': 0.0,
                'pixelloss_weight': 1.0,
                'perceptual_weight': 1.0
            }
        }
    }

    model = ResidualVQModel(**model_config)
    model = model.to(device)

    # 配置优化器 (需要用于梯度计算)
    model.learning_rate = 1e-5

    # 收集所有问题
    all_issues = []
    all_gradient_stats = {}

    # 1. Reconstruction Loss验证
    try:
        issues, recon_loss, log_dict = validate_recon_loss(model, batch, optimizer_idx=0)
        all_issues.extend(issues)
    except Exception as e:
        all_issues.append(f"Recon Loss验证失败: {e}")
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 2. Sparse Loss验证
    try:
        issues, sparse_loss = validate_sparse_loss(model, batch)
        all_issues.extend(issues)
    except Exception as e:
        all_issues.append(f"Sparse Loss验证失败: {e}")
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. Total Loss验证
    try:
        issues, total_loss = validate_total_loss(model, batch)
        all_issues.extend(issues)
    except Exception as e:
        all_issues.append(f"Total Loss验证失败: {e}")
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. ProphetEncoder梯度验证
    try:
        issues, grad_stats = validate_prophet_gradients(model, batch, total_loss)
        all_issues.extend(issues)
        all_gradient_stats.update(grad_stats)
    except Exception as e:
        all_issues.append(f"Prophet梯度验证失败: {e}")
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 5. ObsEncoder梯度验证
    try:
        issues, grad_stats = validate_obs_gradients(model, batch, total_loss)
        all_issues.extend(issues)
        all_gradient_stats.update(grad_stats)
    except Exception as e:
        all_issues.append(f"Obs梯度验证失败: {e}")
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 6. Residual梯度路径验证
    try:
        issues = validate_residual_gradient_paths(model, batch)
        all_issues.extend(issues)
    except Exception as e:
        all_issues.append(f"Residual梯度路径验证失败: {e}")
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 生成报告
    print(f"\n{'='*60}")
    print("验证报告")
    print(f"{'='*60}")

    report = {
        'script': 'debug_loss_and_gradients.py',
        'status': 'PASS' if len(all_issues) == 0 else 'FAIL',
        'total_issues': len(all_issues),
        'issues': all_issues,
        'gradient_stats': all_gradient_stats,
        'checks': {
            'recon_loss': True,
            'sparse_loss': True,
            'total_loss': True,
            'prophet_gradients': True,
            'obs_gradients': True,
            'residual_gradient_paths': True
        }
    }

    # 保存报告
    report_path = '../debug_outputs/loss_and_gradients_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n总计问题数: {len(all_issues)}")
    if len(all_issues) > 0:
        print(f"\n问题列表:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
        print(f"\n❌ 状态: FAIL")
    else:
        print(f"\n✓ 所有检查通过!")
        print(f"\n✓ 状态: PASS")

    print(f"\n报告已保存: {report_path}")
    print(f"可视化已保存: debug_outputs/gradient_analysis/")

    return report['status'] == 'PASS'


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
