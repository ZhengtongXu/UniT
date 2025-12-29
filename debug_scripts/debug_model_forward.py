"""
模型前向传播验证脚本
=====================================

验证项:
1. ProphetEncoder输出 (visual + robot_state → z_exp)
2. ObsEncoder输出 (tactile → z_real)
3. Residual计算 (z_real - z_exp)
4. Decoder重建 (z_fused → recon)

运行时间: 约5-7分钟
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

from UniT.dataset.residual_tactile_dataset import ResidualTactileDataModule
from UniT.taming.models.residual_vqmodel import ResidualVQModel

def validate_prophet_encoder(model, batch):
    """验证ProphetEncoder输出"""
    print(f"\n{'='*60}")
    print("1. ProphetEncoder验证")
    print(f"{'='*60}")

    issues = []

    visual = batch['visual_image']
    robot_state = batch['robot_state']

    print(f"\n输入:")
    print(f"  visual: {visual.shape} (期望: (B, 3, 224, 224))")
    print(f"  robot_state: {robot_state.shape} (期望: (B, 8))")

    # 前向传播
    model.eval()
    with torch.no_grad():
        z_exp = model.prophet_encoder(visual, robot_state)

    print(f"\n输出:")
    print(f"  z_exp: {z_exp.shape} (期望: (B, 3, 20, 16))")
    print(f"  dtype: {z_exp.dtype}")
    print(f"  统计:")
    print(f"    Mean: {z_exp.mean():.4f}")
    print(f"    Std: {z_exp.std():.4f}")
    print(f"    Min: {z_exp.min():.4f}")
    print(f"    Max: {z_exp.max():.4f}")

    # 验证形状
    B = visual.shape[0]
    if z_exp.shape != (B, 3, 20, 16):
        issues.append(f"z_exp形状错误: {z_exp.shape}, 期望: ({B}, 3, 20, 16)")

    # 检查数值异常
    if torch.isnan(z_exp).any():
        issues.append("z_exp包含NaN")
    if torch.isinf(z_exp).any():
        issues.append("z_exp包含Inf")

    # 可视化z_exp
    os.makedirs('../debug_outputs/model_outputs', exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for c in range(3):
        im = axes[c].imshow(z_exp[0, c].cpu().numpy(), cmap='viridis')
        axes[c].set_title(f'z_exp Channel {c}')
        axes[c].axis('off')
        plt.colorbar(im, ax=axes[c])
    plt.tight_layout()
    plt.savefig('../debug_outputs/model_outputs/z_exp_heatmap.png', dpi=150)
    plt.close()

    return issues, z_exp


def validate_obs_encoder(model, batch):
    """验证ObsEncoder输出"""
    print(f"\n{'='*60}")
    print("2. ObsEncoder验证")
    print(f"{'='*60}")

    issues = []

    tactile = batch['tactile_image']

    print(f"\n输入:")
    print(f"  tactile: {tactile.shape} (期望: (B, 3, 128, 160))")

    # 前向传播
    model.eval()
    with torch.no_grad():
        h_obs = model.obs_encoder(tactile)
        z_real = model.quant_conv(h_obs)

    print(f"\n中间输出:")
    print(f"  h_obs: {h_obs.shape}")
    print(f"  h_obs统计: mean={h_obs.mean():.4f}, std={h_obs.std():.4f}")

    print(f"\n最终输出:")
    print(f"  z_real: {z_real.shape} (期望: (B, 3, 20, 16))")
    print(f"  dtype: {z_real.dtype}")
    print(f"  统计:")
    print(f"    Mean: {z_real.mean():.4f}")
    print(f"    Std: {z_real.std():.4f}")
    print(f"    Min: {z_real.min():.4f}")
    print(f"    Max: {z_real.max():.4f}")

    # 验证形状
    B = tactile.shape[0]
    if z_real.shape != (B, 3, 20, 16):
        issues.append(f"z_real形状错误: {z_real.shape}, 期望: ({B}, 3, 20, 16)")

    # 检查下采样倍数
    h_ratio = tactile.shape[2] / z_real.shape[2]  # 128 / 20 = 6.4
    w_ratio = tactile.shape[3] / z_real.shape[3]  # 160 / 16 = 10.0
    print(f"\n✓ 下采样倍数:")
    print(f"  Height: {h_ratio:.1f}x (128→20)")
    print(f"  Width: {w_ratio:.1f}x (160→16)")

    # 检查数值异常
    if torch.isnan(z_real).any():
        issues.append("z_real包含NaN")
    if torch.isinf(z_real).any():
        issues.append("z_real包含Inf")

    # 可视化z_real
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for c in range(3):
        im = axes[c].imshow(z_real[0, c].cpu().numpy(), cmap='viridis')
        axes[c].set_title(f'z_real Channel {c}')
        axes[c].axis('off')
        plt.colorbar(im, ax=axes[c])
    plt.tight_layout()
    plt.savefig('../debug_outputs/model_outputs/z_real_heatmap.png', dpi=150)
    plt.close()

    return issues, z_real


def validate_residual_calculation(model, batch, z_exp, z_real):
    """验证Residual计算"""
    print(f"\n{'='*60}")
    print("3. Residual计算验证")
    print(f"{'='*60}")

    issues = []

    # 计算residual
    residual = z_real - z_exp
    z_fused = z_exp + residual

    print(f"\n计算:")
    print(f"  residual = z_real - z_exp")
    print(f"  z_fused = z_exp + residual")

    print(f"\n形状:")
    print(f"  z_exp: {z_exp.shape}")
    print(f"  z_real: {z_real.shape}")
    print(f"  residual: {residual.shape}")
    print(f"  z_fused: {z_fused.shape}")

    # 验证恒等关系: z_fused == z_real
    max_diff = (z_fused - z_real).abs().max()
    print(f"\n✓ 恒等验证: z_fused == z_real")
    print(f"  最大差异: {max_diff:.2e}")

    if max_diff > 1e-6:
        issues.append(f"恒等关系验证失败: max_diff={max_diff:.2e}")
        print(f"  ❌ 失败 (应该 < 1e-6)")
    else:
        print(f"  ✓ 通过")

    # 统计residual
    print(f"\n✓ Residual统计:")
    print(f"  Mean: {residual.mean():.4f}")
    print(f"  Std: {residual.std():.4f}")
    print(f"  L1 norm: {residual.abs().mean():.4f}")
    print(f"  L2 norm: {(residual**2).mean().sqrt():.4f}")
    print(f"  Min: {residual.min():.4f}")
    print(f"  Max: {residual.max():.4f}")

    # 稀疏性分析
    sparsity_01 = (residual.abs() < 0.01).float().mean()
    sparsity_001 = (residual.abs() < 0.001).float().mean()
    print(f"\n✓ 稀疏性分析:")
    print(f"  |residual| < 0.01: {sparsity_01:.2%}")
    print(f"  |residual| < 0.001: {sparsity_001:.2%}")

    # 与z_exp、z_real的比较
    z_exp_norm = z_exp.abs().mean()
    z_real_norm = z_real.abs().mean()
    residual_norm = residual.abs().mean()

    print(f"\n✓ 范数比较:")
    print(f"  ||z_exp||_1: {z_exp_norm:.4f}")
    print(f"  ||z_real||_1: {z_real_norm:.4f}")
    print(f"  ||residual||_1: {residual_norm:.4f}")
    print(f"  residual/z_real: {residual_norm/z_real_norm:.2%}")

    # 检查数值异常
    if torch.isnan(residual).any():
        issues.append("residual包含NaN")
    if torch.isinf(residual).any():
        issues.append("residual包含Inf")

    # 可视化residual
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for c in range(3):
        im = axes[c].imshow(residual[0, c].cpu().numpy(), cmap='RdBu_r',
                           vmin=-residual.abs().max(), vmax=residual.abs().max())
        axes[c].set_title(f'Residual Channel {c}')
        axes[c].axis('off')
        plt.colorbar(im, ax=axes[c])
    plt.tight_layout()
    plt.savefig('../debug_outputs/model_outputs/residual_heatmap.png', dpi=150)
    plt.close()

    return issues


def validate_decoder_reconstruction(model, batch):
    """验证Decoder重建"""
    print(f"\n{'='*60}")
    print("4. Decoder重建验证")
    print(f"{'='*60}")

    issues = []

    # 完整前向传播
    model.eval()
    with torch.no_grad():
        outputs = model.forward(batch)

    recon = outputs['recon']
    tactile = batch['tactile_image']

    print(f"\n输入/输出:")
    print(f"  原始tactile: {tactile.shape} (期望: (B, 3, 128, 160))")
    print(f"  重建recon: {recon.shape} (期望: (B, 3, 128, 160))")

    # 验证形状
    if recon.shape != tactile.shape:
        issues.append(f"重建形状错误: {recon.shape}, 期望: {tactile.shape}")

    # 统计recon
    print(f"\n✓ 重建统计:")
    print(f"  Mean: {recon.mean():.4f}")
    print(f"  Std: {recon.std():.4f}")
    print(f"  Min: {recon.min():.4f}")
    print(f"  Max: {recon.max():.4f}")

    # 重建误差
    mse = ((recon - tactile) ** 2).mean()
    mae = (recon - tactile).abs().mean()
    psnr = -10 * torch.log10(mse)

    print(f"\n✓ 重建误差:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")

    # 检查数值异常
    if torch.isnan(recon).any():
        issues.append("recon包含NaN")
    if torch.isinf(recon).any():
        issues.append("recon包含Inf")

    # 可视化重建对比
    n_samples = min(4, tactile.shape[0])
    fig, axes = plt.subplots(3, n_samples, figsize=(4*n_samples, 12))

    for i in range(n_samples):
        # 原图
        tactile_img = tactile[i].permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(tactile_img)
        axes[0, i].set_title(f'Original {i}')
        axes[0, i].axis('off')

        # 重建
        recon_img = recon[i].permute(1, 2, 0).cpu().numpy()
        axes[1, i].imshow(recon_img)
        axes[1, i].set_title(f'Reconstruction {i}')
        axes[1, i].axis('off')

        # 误差
        error_img = (recon_img - tactile_img).clip(-0.5, 0.5) + 0.5
        axes[2, i].imshow(error_img)
        axes[2, i].set_title(f'Error {i}')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig('../debug_outputs/model_outputs/reconstruction_comparison.png', dpi=150)
    plt.close()

    return issues


def main():
    """主验证流程"""
    print(f"\n{'#'*60}")
    print("# 模型前向传播验证脚本")
    print(f"{'#'*60}")

    # 创建输出目录
    os.makedirs('../debug_outputs/model_outputs', exist_ok=True)

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

    # 移动到GPU (如果可用)
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
    model.eval()

    print(f"  模型参数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 收集所有问题
    all_issues = []

    # 1. ProphetEncoder验证
    try:
        issues, z_exp = validate_prophet_encoder(model, batch)
        all_issues.extend(issues)
    except Exception as e:
        all_issues.append(f"ProphetEncoder验证失败: {e}")
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 2. ObsEncoder验证
    try:
        issues, z_real = validate_obs_encoder(model, batch)
        all_issues.extend(issues)
    except Exception as e:
        all_issues.append(f"ObsEncoder验证失败: {e}")
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. Residual计算验证
    try:
        issues = validate_residual_calculation(model, batch, z_exp, z_real)
        all_issues.extend(issues)
    except Exception as e:
        all_issues.append(f"Residual计算验证失败: {e}")
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. Decoder重建验证
    try:
        issues = validate_decoder_reconstruction(model, batch)
        all_issues.extend(issues)
    except Exception as e:
        all_issues.append(f"Decoder重建验证失败: {e}")
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 生成报告
    print(f"\n{'='*60}")
    print("验证报告")
    print(f"{'='*60}")

    report = {
        'script': 'debug_model_forward.py',
        'status': 'PASS' if len(all_issues) == 0 else 'FAIL',
        'total_issues': len(all_issues),
        'issues': all_issues,
        'checks': {
            'prophet_encoder': True,
            'obs_encoder': True,
            'residual_calculation': True,
            'decoder_reconstruction': True
        }
    }

    # 保存报告
    report_path = '../debug_outputs/model_forward_report.json'
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
    print(f"可视化已保存: debug_outputs/model_outputs/")

    return report['status'] == 'PASS'


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
