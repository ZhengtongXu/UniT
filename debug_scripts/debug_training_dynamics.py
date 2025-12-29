"""
训练动态验证脚本
=====================================

验证项:
1. 单步训练验证 (参数更新)
2. 多步训练趋势 (20 batches)
3. 完整epoch验证
4. Loss趋势分析
5. Checkpoint保存

运行时间: 约15-20分钟
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
from tqdm import tqdm
from collections import defaultdict
import copy

from UniT.dataset.residual_tactile_dataset import ResidualTactileDataModule
from UniT.taming.models.residual_vqmodel import ResidualVQModel


def validate_single_step_update(model, batch, optimizers):
    """验证单步训练参数更新"""
    print(f"\n{'='*60}")
    print("1. 单步训练参数更新验证")
    print(f"{'='*60}")

    issues = []

    optimizer_gen, optimizer_disc = optimizers

    # 保存初始参数
    print(f"\n保存初始参数...")
    initial_params = {name: p.clone().detach()
                     for name, p in model.named_parameters()}

    # Generator step
    print(f"\n执行Generator step...")
    model.train()
    loss_gen = model.training_step(batch, batch_idx=0, optimizer_idx=0)

    optimizer_gen.zero_grad()
    model.zero_grad()

    # 手动计算loss用于backward
    outputs = model.forward(batch)
    recon_loss, _ = model.recon_loss(
        codebook_loss=torch.tensor(0.0).to(batch['tactile_image'].device),
        inputs=batch['tactile_image'],
        reconstructions=outputs['recon'],
        optimizer_idx=0,
        global_step=500,
        last_layer=model.get_last_layer(),
        split='train'
    )
    sparse_loss = model.lambda_sparse * outputs['residual'].abs().mean()
    total_loss = recon_loss + sparse_loss

    total_loss.backward()
    optimizer_gen.step()

    print(f"  Loss: {total_loss.item():.6f}")

    # Discriminator step
    print(f"\n执行Discriminator step...")
    loss_disc = model.training_step(batch, batch_idx=0, optimizer_idx=1)

    optimizer_disc.zero_grad()
    model.zero_grad()

    # 手动计算disc loss
    outputs2 = model.forward(batch)
    disc_loss, _ = model.recon_loss(
        codebook_loss=torch.tensor(0.0).to(batch['tactile_image'].device),
        inputs=batch['tactile_image'],
        reconstructions=outputs2['recon'],
        optimizer_idx=1,
        global_step=500,
        last_layer=model.get_last_layer(),
        split='train'
    )

    # 初期disc_loss可能为0
    if disc_loss.requires_grad:
        disc_loss.backward()
        optimizer_disc.step()

    print(f"  Loss: {disc_loss.item():.6f}")

    # 验证参数更新
    print(f"\n✓ 验证参数更新:")
    updated_count = 0
    unchanged_count = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        initial = initial_params[name]
        diff = (param - initial).abs().max().item()

        if diff > 1e-8:
            updated_count += 1
        else:
            unchanged_count += 1
            # 只报告前几个未更新的参数
            if unchanged_count <= 5:
                print(f"  ⚠ {name}: 未更新 (diff={diff:.2e})")

    print(f"\n  更新参数: {updated_count}")
    print(f"  未更新参数: {unchanged_count}")

    if updated_count == 0:
        issues.append("所有参数未更新")
        print(f"  ❌ 警告: 所有参数未更新!")
    else:
        print(f"  ✓ 参数正常更新")

    return issues


def validate_multi_step_trend(model, train_loader, optimizers, n_steps=20):
    """验证多步训练趋势"""
    print(f"\n{'='*60}")
    print(f"2. 多步训练趋势验证 ({n_steps} steps)")
    print(f"{'='*60}")

    issues = []

    optimizer_gen, optimizer_disc = optimizers

    # 记录loss历史
    losses = defaultdict(list)

    model.train()

    print(f"\n运行{n_steps}个训练步骤...")
    pbar = tqdm(enumerate(train_loader), total=n_steps, desc="训练步骤")

    for i, batch in pbar:
        if i >= n_steps:
            break

        # 移动到设备
        batch = {k: v.to(next(model.parameters()).device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # Generator step
        model.zero_grad()
        outputs = model.forward(batch)

        recon_loss, log_dict = model.recon_loss(
            codebook_loss=torch.tensor(0.0).to(batch['tactile_image'].device),
            inputs=batch['tactile_image'],
            reconstructions=outputs['recon'],
            optimizer_idx=0,
            global_step=i,
            last_layer=model.get_last_layer(),
            split='train'
        )
        sparse_loss = model.lambda_sparse * outputs['residual'].abs().mean()
        total_loss = recon_loss + sparse_loss

        # 记录
        losses['recon'].append(recon_loss.item())
        losses['sparse'].append(sparse_loss.item())
        losses['total'].append(total_loss.item())

        # 反向传播
        optimizer_gen.zero_grad()
        total_loss.backward()
        optimizer_gen.step()

        # Discriminator step (如果启用)
        if i > 100:  # 假设100步后启用disc
            model.zero_grad()
            outputs2 = model.forward(batch)

            disc_loss, _ = model.recon_loss(
                codebook_loss=torch.tensor(0.0).to(batch['tactile_image'].device),
                inputs=batch['tactile_image'],
                reconstructions=outputs2['recon'],
                optimizer_idx=1,
                global_step=i,
                last_layer=model.get_last_layer(),
                split='train'
            )

            if disc_loss.requires_grad and disc_loss.item() > 0:
                losses['disc'].append(disc_loss.item())
                optimizer_disc.zero_grad()
                disc_loss.backward()
                optimizer_disc.step()

        # 更新进度条
        pbar.set_postfix({
            'total': f"{total_loss.item():.4f}",
            'recon': f"{recon_loss.item():.4f}",
            'sparse': f"{sparse_loss.item():.4f}"
        })

    pbar.close()

    # 分析趋势
    print(f"\n✓ Loss统计:")
    for key in ['recon', 'sparse', 'total']:
        values = losses[key]
        print(f"\n  {key}_loss:")
        print(f"    初始: {values[0]:.6f}")
        print(f"    最终: {values[-1]:.6f}")
        print(f"    平均: {np.mean(values):.6f}")
        print(f"    标准差: {np.std(values):.6f}")
        print(f"    最小: {np.min(values):.6f}")
        print(f"    最大: {np.max(values):.6f}")

        # 检查趋势
        if key == 'recon':
            if values[-1] > values[0]:
                issues.append(f"recon_loss未下降: {values[0]:.4f} → {values[-1]:.4f}")
                print(f"    ⚠ 警告: 未下降")
            else:
                print(f"    ✓ 下降 ({values[0]:.4f} → {values[-1]:.4f})")

    # 绘制趋势图
    os.makedirs('../debug_outputs/loss_trends', exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Recon loss
    axes[0, 0].plot(losses['recon'], 'b-', linewidth=2)
    axes[0, 0].set_title('Reconstruction Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # Sparse loss
    axes[0, 1].plot(losses['sparse'], 'r-', linewidth=2)
    axes[0, 1].set_title('Sparse Loss')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)

    # Total loss
    axes[1, 0].plot(losses['total'], 'g-', linewidth=2)
    axes[1, 0].set_title('Total Loss')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)

    # Disc loss (if any)
    if len(losses['disc']) > 0:
        axes[1, 1].plot(losses['disc'], 'm-', linewidth=2)
        axes[1, 1].set_title('Discriminator Loss')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Disc not activated\n(requires >10k steps)',
                       ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Discriminator Loss')

    plt.tight_layout()
    plt.savefig('../debug_outputs/loss_trends/multi_step_trend.png', dpi=150)
    plt.close()

    return issues, losses


def validate_full_epoch(model, datamodule, optimizers):
    """验证完整epoch"""
    print(f"\n{'='*60}")
    print("3. 完整Epoch验证")
    print(f"{'='*60}")

    issues = []

    optimizer_gen, optimizer_disc = optimizers

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # 训练loss记录
    train_losses = defaultdict(list)
    val_losses = defaultdict(list)

    # 训练阶段
    print(f"\n训练阶段 ({len(train_loader)} batches)...")
    model.train()

    pbar = tqdm(train_loader, desc="训练")
    for i, batch in enumerate(pbar):
        # 移动到设备
        batch = {k: v.to(next(model.parameters()).device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # Generator step
        model.zero_grad()
        outputs = model.forward(batch)

        recon_loss, _ = model.recon_loss(
            codebook_loss=torch.tensor(0.0).to(batch['tactile_image'].device),
            inputs=batch['tactile_image'],
            reconstructions=outputs['recon'],
            optimizer_idx=0,
            global_step=i,
            last_layer=model.get_last_layer(),
            split='train'
        )
        sparse_loss = model.lambda_sparse * outputs['residual'].abs().mean()
        total_loss = recon_loss + sparse_loss

        # 记录
        train_losses['recon'].append(recon_loss.item())
        train_losses['sparse'].append(sparse_loss.item())
        train_losses['total'].append(total_loss.item())

        # 反向传播
        optimizer_gen.zero_grad()
        total_loss.backward()
        optimizer_gen.step()

        # 更新进度条
        if i % 50 == 0:
            pbar.set_postfix({
                'total': f"{np.mean(train_losses['total'][-50:]):.4f}",
                'recon': f"{np.mean(train_losses['recon'][-50:]):.4f}",
            })

        # 检查NaN/Inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            issues.append(f"训练step {i}: loss为NaN/Inf")
            print(f"\n  ❌ Step {i}: Loss异常!")
            break

    pbar.close()

    # 验证阶段
    print(f"\n验证阶段 ({len(val_loader)} batches)...")
    model.eval()

    pbar = tqdm(val_loader, desc="验证")
    with torch.no_grad():
        for batch in pbar:
            # 移动到设备
            batch = {k: v.to(next(model.parameters()).device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            outputs = model.forward(batch)

            recon_loss, _ = model.recon_loss(
                codebook_loss=torch.tensor(0.0).to(batch['tactile_image'].device),
                inputs=batch['tactile_image'],
                reconstructions=outputs['recon'],
                optimizer_idx=0,
                global_step=0,
                last_layer=model.get_last_layer(),
                split='val'
            )
            sparse_loss = model.lambda_sparse * outputs['residual'].abs().mean()
            total_loss = recon_loss + sparse_loss

            # 记录
            val_losses['recon'].append(recon_loss.item())
            val_losses['sparse'].append(sparse_loss.item())
            val_losses['total'].append(total_loss.item())

    pbar.close()

    # 统计
    print(f"\n✓ Epoch统计:")
    print(f"\n  训练集:")
    for key in ['recon', 'sparse', 'total']:
        values = train_losses[key]
        print(f"    {key}_loss: {np.mean(values):.6f} ± {np.std(values):.6f}")

    print(f"\n  验证集:")
    for key in ['recon', 'sparse', 'total']:
        values = val_losses[key]
        print(f"    {key}_loss: {np.mean(values):.6f} ± {np.std(values):.6f}")

    # 绘制完整epoch趋势
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, key in enumerate(['recon', 'sparse', 'total']):
        axes[idx].plot(train_losses[key], 'b-', alpha=0.5, label='Train')
        axes[idx].plot(np.arange(len(val_losses[key])) * (len(train_losses[key]) / len(val_losses[key])),
                      val_losses[key], 'r-', linewidth=2, label='Val')
        axes[idx].set_title(f'{key.capitalize()} Loss')
        axes[idx].set_xlabel('Batch')
        axes[idx].set_ylabel('Loss')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../debug_outputs/loss_trends/full_epoch_trend.png', dpi=150)
    plt.close()

    return issues, train_losses, val_losses


def validate_checkpoint_saving(model):
    """验证Checkpoint保存"""
    print(f"\n{'='*60}")
    print("4. Checkpoint保存验证")
    print(f"{'='*60}")

    issues = []

    # 创建临时checkpoint目录
    ckpt_dir = '../debug_outputs/test_checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, 'test_checkpoint.ckpt')

    # 保存checkpoint
    print(f"\n保存checkpoint: {ckpt_path}")
    try:
        checkpoint = {
            'state_dict': model.state_dict(),
            'epoch': 0,  # 使用epoch代替global_step
        }
        torch.save(checkpoint, ckpt_path)
        print(f"  ✓ 保存成功")
    except Exception as e:
        issues.append(f"Checkpoint保存失败: {e}")
        print(f"  ❌ 保存失败: {e}")
        return issues

    # 验证文件存在
    if not os.path.exists(ckpt_path):
        issues.append("Checkpoint文件不存在")
        print(f"  ❌ 文件不存在")
        return issues

    # 检查文件大小
    size_mb = os.path.getsize(ckpt_path) / 1024 / 1024
    print(f"  文件大小: {size_mb:.2f} MB")

    if size_mb < 1:
        issues.append(f"Checkpoint文件过小: {size_mb:.2f} MB")
        print(f"  ⚠ 警告: 文件可能不完整")

    # 尝试加载
    print(f"\n加载checkpoint...")
    try:
        loaded = torch.load(ckpt_path)
        print(f"  ✓ 加载成功")
        print(f"  包含keys: {list(loaded.keys())}")

        # 验证state_dict
        if 'state_dict' in loaded:
            n_params = len(loaded['state_dict'])
            print(f"  state_dict参数数量: {n_params}")

            if n_params == 0:
                issues.append("state_dict为空")
                print(f"  ❌ state_dict为空")
        else:
            issues.append("Checkpoint缺少state_dict")
            print(f"  ❌ 缺少state_dict")

    except Exception as e:
        issues.append(f"Checkpoint加载失败: {e}")
        print(f"  ❌ 加载失败: {e}")

    return issues


def main():
    """主验证流程"""
    print(f"\n{'#'*60}")
    print("# 训练动态验证脚本")
    print(f"{'#'*60}")

    # 创建输出目录
    os.makedirs('../debug_outputs/loss_trends', exist_ok=True)
    os.makedirs('../debug_outputs/training_logs', exist_ok=True)

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

    # 创建模型
    print(f"\n创建模型...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

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
    model.learning_rate = 1e-5

    # 配置优化器
    optimizers = model.configure_optimizers()
    if isinstance(optimizers, tuple):
        opt_list, _ = optimizers
        optimizer_gen, optimizer_disc = opt_list
    else:
        optimizer_gen, optimizer_disc = optimizers

    # 获取一个batch
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    # 收集所有问题
    all_issues = []

    # 1. 单步训练验证
    print(f"\n{'='*60}")
    print("开始验证...")
    print(f"{'='*60}")

    try:
        issues = validate_single_step_update(model, batch, (optimizer_gen, optimizer_disc))
        all_issues.extend(issues)
    except Exception as e:
        all_issues.append(f"单步训练验证失败: {e}")
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    # 2. 多步训练趋势
    try:
        issues, multi_step_losses = validate_multi_step_trend(
            model, train_loader, (optimizer_gen, optimizer_disc), n_steps=20
        )
        all_issues.extend(issues)
    except Exception as e:
        all_issues.append(f"多步训练趋势验证失败: {e}")
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    # 3. 完整epoch验证 (可选, 耗时较长)
    run_full_epoch = False  # 设为True以运行完整epoch
    if run_full_epoch:
        try:
            # 重新创建模型 (避免之前训练的影响)
            model = ResidualVQModel(**model_config)
            model = model.to(device)
            model.learning_rate = 1e-5

            optimizers = model.configure_optimizers()
            if isinstance(optimizers, tuple):
                opt_list, _ = optimizers
                optimizer_gen, optimizer_disc = opt_list
            else:
                optimizer_gen, optimizer_disc = optimizers

            issues, train_losses, val_losses = validate_full_epoch(
                model, datamodule, (optimizer_gen, optimizer_disc)
            )
            all_issues.extend(issues)
        except Exception as e:
            all_issues.append(f"完整epoch验证失败: {e}")
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⊕ 跳过完整epoch验证 (设置run_full_epoch=True以启用)")

    # 4. Checkpoint保存验证
    try:
        issues = validate_checkpoint_saving(model)
        all_issues.extend(issues)
    except Exception as e:
        all_issues.append(f"Checkpoint保存验证失败: {e}")
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    # 生成报告
    print(f"\n{'='*60}")
    print("验证报告")
    print(f"{'='*60}")

    report = {
        'script': 'debug_training_dynamics.py',
        'status': 'PASS' if len(all_issues) == 0 else 'FAIL',
        'total_issues': len(all_issues),
        'issues': all_issues,
        'checks': {
            'single_step_update': True,
            'multi_step_trend': True,
            'full_epoch': run_full_epoch,
            'checkpoint_saving': True
        }
    }

    # 保存报告
    report_path = '../debug_outputs/training_dynamics_report.json'
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
    print(f"可视化已保存: debug_outputs/loss_trends/")

    return report['status'] == 'PASS'


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
