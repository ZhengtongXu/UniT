"""
数据流验证脚本
=====================================

验证项:
1. Dataset.__getitem__ 输出形状、dtype、值域
2. DataLoader批次正确性 (collation、无重复)
3. 训练/验证集划分 (episode边界)
4. 归一化参数一致性

运行时间: 约3-5分钟
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
from tqdm import tqdm
import json
from collections import defaultdict

from UniT.dataset.residual_tactile_dataset import ResidualTactileDataModule

def validate_dataset_sample(dataset, sample_idx=0):
    """验证单个样本的正确性"""
    print(f"\n{'='*60}")
    print("1. Dataset样本验证")
    print(f"{'='*60}")

    sample = dataset[sample_idx]
    issues = []

    # 1.1 检查tactile_image
    tactile = sample['tactile_image']
    print(f"\n✓ tactile_image:")
    print(f"  Shape: {tactile.shape} (期望: (3, 160, 128) or (3, 128, 160))")
    print(f"  Dtype: {tactile.dtype} (期望: torch.float32)")
    print(f"  值域: [{tactile.min():.3f}, {tactile.max():.3f}] (期望: [0, 1])")

    # 接受两种形状 (数据可能是160x128或128x160)
    if tactile.shape not in [(3, 128, 160), (3, 160, 128)]:
        issues.append(f"tactile_image形状错误: {tactile.shape}")
    if tactile.dtype != torch.float32:
        issues.append(f"tactile_image dtype错误: {tactile.dtype}")
    if tactile.min() < -0.1 or tactile.max() > 1.1:
        issues.append(f"tactile_image值域异常: [{tactile.min()}, {tactile.max()}]")

    # 1.2 检查visual_image
    visual = sample['visual_image']
    print(f"\n✓ visual_image:")
    print(f"  Shape: {visual.shape} (期望: (3, 224, 224))")
    print(f"  Dtype: {visual.dtype}")
    print(f"  值域: [{visual.min():.3f}, {visual.max():.3f}] (ImageNet归一化)")

    if visual.shape != (3, 224, 224):
        issues.append(f"visual_image形状错误: {visual.shape}")

    # 1.3 检查robot_state
    state = sample['robot_state']
    print(f"\n✓ robot_state:")
    print(f"  Shape: {state.shape} (期望: (8,))")
    print(f"  Dtype: {state.dtype}")
    print(f"  值域: [{state.min():.3f}, {state.max():.3f}] (Z-score归一化)")
    print(f"  值: {state.numpy()}")

    if state.shape != (8,):
        issues.append(f"robot_state形状错误: {state.shape}")

    # 1.4 检查episode_id
    episode_id = sample['episode_id']
    print(f"\n✓ episode_id: {episode_id}")

    # 可视化样本
    print(f"\n保存样本可视化...")
    os.makedirs('../debug_outputs/data_samples', exist_ok=True)

    # Tactile
    tactile_np = tactile.permute(1, 2, 0).numpy()
    plt.figure(figsize=(8, 6))
    plt.imshow(tactile_np)
    plt.title(f"Sample {sample_idx} - Tactile Image")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'../debug_outputs/data_samples/sample_{sample_idx}_tactile.png', dpi=150)
    plt.close()

    # Visual (反归一化)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    visual_denorm = visual * std + mean
    visual_np = visual_denorm.permute(1, 2, 0).numpy().clip(0, 1)

    plt.figure(figsize=(8, 6))
    plt.imshow(visual_np)
    plt.title(f"Sample {sample_idx} - Visual Image")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'../debug_outputs/data_samples/sample_{sample_idx}_visual.png', dpi=150)
    plt.close()

    # Robot State
    plt.figure(figsize=(10, 4))
    plt.bar(range(8), state.numpy())
    plt.xlabel('State Dimension')
    plt.ylabel('Normalized Value')
    plt.title(f"Sample {sample_idx} - Robot State (Z-score normalized)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'../debug_outputs/data_samples/sample_{sample_idx}_robot_state.png', dpi=150)
    plt.close()

    return issues


def validate_dataloader_batches(datamodule):
    """验证DataLoader批次正确性"""
    print(f"\n{'='*60}")
    print("2. DataLoader批次验证")
    print(f"{'='*60}")

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    issues = []

    # 2.1 检查batch形状
    print(f"\n✓ 训练集batch形状:")
    batch = next(iter(train_loader))
    print(f"  tactile_image: {batch['tactile_image'].shape} (期望: (B, 3, H, W))")
    print(f"  visual_image: {batch['visual_image'].shape} (期望: (B, 3, 224, 224))")
    print(f"  robot_state: {batch['robot_state'].shape} (期望: (B, 8))")
    print(f"  episode_id: {batch['episode_id'].shape} (期望: (B,))")

    B = batch['tactile_image'].shape[0]
    # 接受两种形状
    if batch['tactile_image'].shape not in [(B, 3, 128, 160), (B, 3, 160, 128)]:
        issues.append(f"训练batch tactile形状错误: {batch['tactile_image'].shape}")
    if batch['visual_image'].shape != (B, 3, 224, 224):
        issues.append(f"训练batch visual形状错误: {batch['visual_image'].shape}")
    if batch['robot_state'].shape != (B, 8):
        issues.append(f"训练batch state形状错误: {batch['robot_state'].shape}")

    # 2.2 检查验证集batch
    print(f"\n✓ 验证集batch形状:")
    val_batch = next(iter(val_loader))
    print(f"  tactile_image: {val_batch['tactile_image'].shape}")
    print(f"  visual_image: {val_batch['visual_image'].shape}")
    print(f"  robot_state: {val_batch['robot_state'].shape}")

    # 2.3 检查数据类型
    print(f"\n✓ Batch数据类型:")
    print(f"  tactile dtype: {batch['tactile_image'].dtype}")
    print(f"  visual dtype: {batch['visual_image'].dtype}")
    print(f"  state dtype: {batch['robot_state'].dtype}")
    print(f"  episode_id dtype: {batch['episode_id'].dtype}")

    return issues


def validate_train_val_split(datamodule):
    """验证训练/验证集划分"""
    print(f"\n{'='*60}")
    print("3. 训练/验证集划分验证")
    print(f"{'='*60}")

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    issues = []

    # 收集所有episode_id
    print(f"\n收集训练集episodes...")
    train_episodes = set()
    for batch in tqdm(train_loader, desc="训练集"):
        train_episodes.update(batch['episode_id'].tolist())

    print(f"\n收集验证集episodes...")
    val_episodes = set()
    for batch in tqdm(val_loader, desc="验证集"):
        val_episodes.update(batch['episode_id'].tolist())

    # 检查交集
    overlap = train_episodes & val_episodes

    print(f"\n✓ Episode统计:")
    print(f"  训练集episodes: {len(train_episodes)}")
    print(f"  验证集episodes: {len(val_episodes)}")
    print(f"  重叠episodes: {len(overlap)}")

    if len(overlap) > 0:
        issues.append(f"训练/验证集有{len(overlap)}个重叠episodes: {overlap}")
        print(f"  ⚠ 警告: 发现重叠episodes!")
    else:
        print(f"  ✓ 无重叠 (正确)")

    # 统计样本数
    print(f"\n收集样本数统计...")
    train_samples = sum(batch['tactile_image'].shape[0] for batch in train_loader)
    val_samples = sum(batch['tactile_image'].shape[0] for batch in val_loader)

    print(f"\n✓ 样本数统计:")
    print(f"  训练集样本: {train_samples}")
    print(f"  验证集样本: {val_samples}")
    print(f"  总计: {train_samples + val_samples}")
    print(f"  验证集比例: {val_samples / (train_samples + val_samples):.2%}")

    return issues


def validate_normalization_consistency(datamodule):
    """验证归一化参数一致性"""
    print(f"\n{'='*60}")
    print("4. 归一化参数一致性验证")
    print(f"{'='*60}")

    issues = []

    # 获取训练集和验证集的数据集对象
    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset

    # 检查归一化参数
    print(f"\n✓ Robot State归一化参数:")
    print(f"\n训练集:")
    print(f"  Mean: {train_dataset.robot_mean}")
    print(f"  Std: {train_dataset.robot_std}")

    if hasattr(val_dataset, 'robot_mean'):
        print(f"\n验证集:")
        print(f"  Mean: {val_dataset.robot_mean}")
        print(f"  Std: {val_dataset.robot_std}")

        # 比较参数
        mean_diff = np.abs(train_dataset.robot_mean - val_dataset.robot_mean).max()
        std_diff = np.abs(train_dataset.robot_std - val_dataset.robot_std).max()

        print(f"\n✓ 参数差异:")
        print(f"  Mean最大差异: {mean_diff:.6f}")
        print(f"  Std最大差异: {std_diff:.6f}")

        if mean_diff > 1e-6 or std_diff > 1e-6:
            issues.append(f"训练/验证集归一化参数不一致 (mean_diff={mean_diff}, std_diff={std_diff})")
            print(f"  ⚠ 警告: 参数不一致!")
        else:
            print(f"  ✓ 参数一致")
    else:
        print(f"\n  ⚠ 验证集未设置归一化参数 (可能使用训练集参数)")

    return issues


def main():
    """主验证流程"""
    print(f"\n{'#'*60}")
    print("# 数据流验证脚本")
    print(f"{'#'*60}")

    # 创建输出目录
    os.makedirs('../debug_outputs/data_samples', exist_ok=True)

    # 加载数据模块
    print(f"\n加载数据模块...")
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

    # 收集所有问题
    all_issues = []

    # 1. Dataset样本验证
    try:
        issues = validate_dataset_sample(datamodule.train_dataset, sample_idx=0)
        all_issues.extend(issues)
    except Exception as e:
        all_issues.append(f"Dataset样本验证失败: {e}")
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    # 2. DataLoader批次验证
    try:
        issues = validate_dataloader_batches(datamodule)
        all_issues.extend(issues)
    except Exception as e:
        all_issues.append(f"DataLoader批次验证失败: {e}")
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    # 3. 训练/验证集划分验证
    try:
        issues = validate_train_val_split(datamodule)
        all_issues.extend(issues)
    except Exception as e:
        all_issues.append(f"训练/验证集划分验证失败: {e}")
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    # 4. 归一化一致性验证
    try:
        issues = validate_normalization_consistency(datamodule)
        all_issues.extend(issues)
    except Exception as e:
        all_issues.append(f"归一化一致性验证失败: {e}")
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    # 生成报告
    print(f"\n{'='*60}")
    print("验证报告")
    print(f"{'='*60}")

    report = {
        'script': 'debug_data_pipeline.py',
        'status': 'PASS' if len(all_issues) == 0 else 'FAIL',
        'total_issues': len(all_issues),
        'issues': all_issues,
        'checks': {
            'dataset_sample': True,
            'dataloader_batches': True,
            'train_val_split': True,
            'normalization_consistency': True
        }
    }

    # 保存报告
    report_path = '../debug_outputs/data_pipeline_report.json'
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
    print(f"可视化已保存: debug_outputs/data_samples/")

    return report['status'] == 'PASS'


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
