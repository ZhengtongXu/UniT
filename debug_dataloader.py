"""
调试DataLoader - 检查是否真的有数据重复
"""
import torch
import numpy as np
from UniT.dataset.residual_tactile_dataset import ResidualTactileDataModule
import hashlib

def hash_batch(batch):
    """计算batch的hash"""
    tactile = batch['tactile_image'].cpu().numpy()
    return hashlib.md5(tactile.tobytes()).hexdigest()

# 创建DataModule (使用相同配置)
datamodule = ResidualTactileDataModule(
    dataset_path="data/residual_tactile_train_v2",
    batch_size=16,
    num_workers=8,
    visual_size=(224, 224),
    tactile_size=(128, 160),
    normalize_robot_state=True,
    val_ratio=0.1,
    seed=42
)

datamodule.setup()
train_loader = datamodule.train_dataloader()

print(f"=== DataLoader调试 ===")
print(f"训练集大小: {len(datamodule.train_dataset)}")
print(f"验证集大小: {len(datamodule.val_dataset)}")
print(f"Batch size: {datamodule.batch_size}")
print(f"理论steps/epoch: {len(train_loader)}")
print(f"")

# 模拟多个epoch，检测数据重复
print("模拟训练3个epoch，检查是否有重复batch...\n")

batch_hashes_history = []

for epoch in range(3):
    print(f"Epoch {epoch}:")
    batch_hashes = []

    for i, batch in enumerate(train_loader):
        batch_hash = hash_batch(batch)
        batch_hashes.append(batch_hash)

        if i < 3:  # 只打印前3个batch的hash
            print(f"  Batch {i}: {batch_hash}")

    print(f"  共 {len(batch_hashes)} 个batch")

    # 检查epoch内是否有重复
    unique_hashes = set(batch_hashes)
    print(f"  唯一batch数: {len(unique_hashes)}")

    if len(unique_hashes) < len(batch_hashes):
        print(f"  ⚠️ 警告: Epoch内有重复batch!")

    batch_hashes_history.append(batch_hashes)
    print()

# 检查跨epoch的重复
print("检查跨epoch的数据重复...")
for i in range(len(batch_hashes_history) - 1):
    overlap = set(batch_hashes_history[i]) & set(batch_hashes_history[i+1])
    if len(overlap) == len(batch_hashes_history[i]):
        print(f"  ⚠️⚠️⚠️ Epoch {i} 和 Epoch {i+1} 的batch完全相同!")
    elif len(overlap) > 0:
        print(f"  Epoch {i} 和 Epoch {i+1} 有 {len(overlap)} 个重复batch (正常情况不应重复)")
    else:
        print(f"  ✓ Epoch {i} 和 Epoch {i+1} 无重复")

print("\n✓ 调试完成")
