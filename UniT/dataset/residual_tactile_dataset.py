"""
ResidualTactileDataset: 多模态触觉数据集
==========================================

加载扩展的Zarr格式数据,包含:
- tactile_image: 触觉传感器图像 (GelSight)
- visual_image: 腕部视觉图像 (RealSense RGB)
- robot_state: 机器人状态 (7-DOF joints + gripper_width)

返回batch格式:
{
    'tactile_image': (B, 3, 128, 160) float32 [0,1],
    'visual_image': (B, 3, 224, 224) float32 ImageNet归一化,
    'robot_state': (B, 8) float32 标准化,
    'episode_id': (B,) int64
}
"""

import numpy as np
import zarr
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict
import os
import pytorch_lightning as pl


class ResidualTactileDataset(Dataset):
    """
    多模态触觉数据集,用于Residual Tactile Representation Learning

    Args:
        dataset_path: Zarr数据集路径 (由convert_residual_data.py生成)
        visual_size: (width, height) 视觉图像尺寸
        tactile_size: (width, height) 触觉图像尺寸
        normalize_robot_state: 是否归一化机器人状态
        val_ratio: 验证集比例
        train: 是否为训练集
        seed: 随机种子
    """

    def __init__(self,
                 dataset_path: str,
                 visual_size: Tuple[int, int] = (224, 224),
                 tactile_size: Tuple[int, int] = (128, 160),
                 normalize_robot_state: bool = True,
                 val_ratio: float = 0.1,
                 train: bool = True,
                 seed: int = 42):
        super().__init__()

        # 参数
        self.dataset_path = dataset_path
        self.visual_size = visual_size
        self.tactile_size = tactile_size
        self.normalize_robot_state = normalize_robot_state
        self.val_ratio = val_ratio
        self.train = train
        self.seed = seed

        # 加载Zarr数据
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Zarr数据集不存在: {dataset_path}")

        self.zarr_root = zarr.open(dataset_path, mode='r')

        # 数据集大小
        self.total_frames = self.zarr_root['data/tactile_image'].shape[0]
        print(f"加载数据集: {dataset_path}")
        print(f"  总帧数: {self.total_frames}")
        print(f"  Episodes: {self.zarr_root.attrs.get('num_episodes', 'unknown')}")

        # 划分训练/验证集 (按episode边界)
        self._split_train_val()

        # 计算机器人状态归一化参数
        if self.normalize_robot_state:
            self._compute_normalization_stats()

        print(f"  {'训练' if train else '验证'}集大小: {len(self)}")

    def _split_train_val(self):
        """
        按episode边界划分训练/验证集

        重要: 不能随机打乱帧,必须保持episode的连续性
        """
        episode_ids = self.zarr_root['meta/episode_id'][:]
        unique_episodes = np.unique(episode_ids)
        num_episodes = len(unique_episodes)

        # 随机划分episodes
        rng = np.random.RandomState(self.seed)
        shuffled_episodes = rng.permutation(unique_episodes)

        num_val = int(num_episodes * self.val_ratio)
        val_episodes = set(shuffled_episodes[:num_val])

        # 获取对应的帧索引
        all_indices = np.arange(self.total_frames)

        if self.train:
            self.indices = all_indices[~np.isin(episode_ids, list(val_episodes))]
        else:
            self.indices = all_indices[np.isin(episode_ids, list(val_episodes))]

    def _compute_normalization_stats(self):
        """
        计算机器人状态的归一化参数 (仅使用训练集)

        归一化方式: z = (x - mean) / std
        """
        # 只在训练集计算统计量
        if self.train:
            # Use .oindex for fancy indexing with zarr
            robot_states = self.zarr_root['data/robot_state'].oindex[self.indices]
            self.robot_mean = robot_states.mean(axis=0).astype(np.float32)
            self.robot_std = robot_states.std(axis=0).astype(np.float32)

            # 避免除零
            self.robot_std = np.where(self.robot_std < 1e-6, 1.0, self.robot_std)

            print(f"  机器人状态归一化:")
            print(f"    Mean: {self.robot_mean}")
            print(f"    Std: {self.robot_std}")
        else:
            # 验证集使用训练集的统计量 (需要从外部传入,这里使用临时值)
            print("  警告: 验证集应该使用训练集的归一化参数")
            self.robot_mean = np.zeros(8, dtype=np.float32)
            self.robot_std = np.ones(8, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        Returns:
            dict {
                'tactile_image': (3, H, W) float32 in [0,1],
                'visual_image': (3, H, W) float32 ImageNet normalized,
                'robot_state': (8,) float32 standardized,
                'episode_id': int
            }
        """
        # 真实索引
        real_idx = self.indices[idx]

        # 加载数据
        tactile = self.zarr_root['data/tactile_image'][real_idx]  # (H, W, 3) uint8
        visual = self.zarr_root['data/visual_image'][real_idx]    # (H, W, 3) uint8
        robot_state = self.zarr_root['data/robot_state'][real_idx]  # (8,) float32
        episode_id = self.zarr_root['meta/episode_id'][real_idx]    # int32

        # 预处理
        tactile = self._process_tactile(tactile)
        visual = self._process_visual(visual)
        robot_state = self._normalize_state(robot_state)

        return {
            'tactile_image': torch.from_numpy(tactile),
            'visual_image': torch.from_numpy(visual),
            'robot_state': torch.from_numpy(robot_state),
            'episode_id': torch.tensor(episode_id, dtype=torch.long)
        }

    def _process_tactile(self, img: np.ndarray) -> np.ndarray:
        """
        触觉图像预处理

        Args:
            img: (H, W, 3) uint8
        Returns:
            img: (3, H, W) float32 in [0, 1]
        """
        # 确保尺寸正确
        if img.shape[:2] != (self.tactile_size[1], self.tactile_size[0]):
            img = cv2.resize(img, self.tactile_size, interpolation=cv2.INTER_AREA)

        # 归一化到 [0, 1]
        img = img.astype(np.float32) / 255.0

        # HWC → CHW
        img = img.transpose(2, 0, 1)

        return img

    def _process_visual(self, img: np.ndarray) -> np.ndarray:
        """
        视觉图像预处理 (ImageNet标准化)

        Args:
            img: (H, W, 3) uint8
        Returns:
            img: (3, H, W) float32 ImageNet normalized
        """
        # 确保尺寸正确
        if img.shape[:2] != (self.visual_size[1], self.visual_size[0]):
            img = cv2.resize(img, self.visual_size, interpolation=cv2.INTER_AREA)

        # 转换为float并归一化到 [0, 1]
        img = img.astype(np.float32) / 255.0

        # ImageNet归一化 (ResNet34标准)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img = (img - mean) / std

        # HWC → CHW
        img = img.transpose(2, 0, 1)

        return img

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        机器人状态归一化

        Args:
            state: (8,) float32
        Returns:
            state: (8,) float32 normalized
        """
        if self.normalize_robot_state:
            state = (state - self.robot_mean) / self.robot_std

        return state.astype(np.float32)

    def get_normalization_stats(self) -> Dict[str, np.ndarray]:
        """
        获取归一化统计量 (用于验证集)

        Returns:
            dict {'mean': (8,), 'std': (8,)}
        """
        if self.normalize_robot_state:
            return {
                'mean': self.robot_mean,
                'std': self.robot_std
            }
        else:
            return None


import pytorch_lightning as pl
from torch.utils.data import DataLoader

# 用于PyTorch Lightning DataModule的便捷类
class ResidualTactileDataModule(pl.LightningDataModule):
    """
    数据模块封装,确保训练/验证集使用相同的归一化参数

    Args:
        dataset_path: Zarr路径
        batch_size: batch大小
        num_workers: DataLoader worker数
        **kwargs: 传递给ResidualTactileDataset的参数
    """

    def __init__(self,
                 dataset_path: str,
                 batch_size: int = 16,
                 num_workers: int = 4,
                 **kwargs):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = kwargs

    def setup(self, stage=None):
        # 创建训练集
        self.train_dataset = ResidualTactileDataset(
            dataset_path=self.dataset_path,
            train=True,
            **self.dataset_kwargs
        )

        # 获取归一化参数
        norm_stats = self.train_dataset.get_normalization_stats()

        # 创建验证集 (使用训练集的归一化参数)
        self.val_dataset = ResidualTactileDataset(
            dataset_path=self.dataset_path,
            train=False,
            **self.dataset_kwargs
        )

        if norm_stats is not None:
            self.val_dataset.robot_mean = norm_stats['mean']
            self.val_dataset.robot_std = norm_stats['std']

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self):
        return self.val_dataloader()


# 测试代码
if __name__ == "__main__":
    # 测试数据集加载
    dataset = ResidualTactileDataset(
        dataset_path="data/residual_tactile_zarr",
        visual_size=(224, 224),
        tactile_size=(128, 160),
        normalize_robot_state=True,
        train=True
    )

    print(f"\n数据集大小: {len(dataset)}")

    # 测试单个样本
    sample = dataset[0]
    print("\n样本结构:")
    for k, v in sample.items():
        print(f"  {k}: {v.shape} {v.dtype}")

    # 测试DataModule
    datamodule = ResidualTactileDataModule(
        dataset_path="data/residual_tactile_zarr",
        batch_size=4,
        num_workers=0
    )

    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    print("\nBatch结构:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape} {v.dtype}")
