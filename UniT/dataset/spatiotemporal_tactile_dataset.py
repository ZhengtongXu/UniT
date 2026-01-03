"""
Spatiotemporal Tactile Dataset
================================

时空触觉数据集，返回视频片段而非单帧

与单帧数据集的主要区别:
1. 返回长度为T的连续触觉视频片段
2. 返回对应的状态序列和动作序列
3. Visual context可以是第一帧或中间帧

数据格式:
- tactile_video: (3, T, H, W) - T帧触觉图像序列
- visual_context: (3, H_v, W_v) - 单帧视觉上下文
- state_seq: (T, state_dim) - 机器人状态序列
- action_seq: (T, action_dim) - 动作序列
"""

import zarr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Dict, Optional, Tuple
import torchvision.transforms as transforms


class SpatiotemporalTactileDataset(Dataset):
    """
    时空触觉数据集

    从zarr数据中采样长度为seq_length的连续片段
    """

    def __init__(self,
                 dataset_path: str,
                 seq_length: int = 16,
                 visual_size: tuple = (224, 224),
                 tactile_size: tuple = (128, 160),
                 normalize_robot_state: bool = True,
                 train: bool = True):
        super().__init__()

        self.dataset_path = dataset_path
        self.seq_length = seq_length
        self.visual_size = visual_size
        self.tactile_size = tactile_size
        self.normalize_robot_state = normalize_robot_state
        self.train = train

        # 打开zarr数据
        self.data_root = zarr.open(dataset_path, mode='r')

        # 检查数据结构
        if 'data' not in self.data_root:
            raise ValueError(f"数据集 {dataset_path} 缺少 'data' 组")

        self.data = self.data_root['data']
        self.meta = self.data_root['meta']

        # 获取episode ID
        self.episode_ids_array = np.array(self.meta['episode_id'])
        unique_episodes = np.unique(self.episode_ids_array)

        print(f"加载数据集: {dataset_path}")
        print(f"  总帧数: {len(self.episode_ids_array)}")
        print(f"  Episodes: {len(unique_episodes)}")

        # 构建索引: 找到每个episode的连续帧
        # 然后为每个episode生成有效的起始索引
        self.valid_samples = []

        for ep_id in unique_episodes:
            # 找到属于该episode的所有帧
            ep_frame_indices = np.where(self.episode_ids_array == ep_id)[0]
            ep_len = len(ep_frame_indices)

            # 检查帧是否连续
            if not np.all(np.diff(ep_frame_indices) == 1):
                print(f"  警告: Episode {ep_id} 的帧索引不连续，跳过")
                continue

            # 为该episode生成所有有效的起始索引
            # 确保能采样完整的seq_length帧
            for i in range(ep_len - seq_length + 1):
                start_frame_idx = ep_frame_indices[i]
                self.valid_samples.append((ep_id, start_frame_idx))

        print(f"  有效样本数: {len(self.valid_samples)}")

        # 图像变换
        self.visual_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(visual_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.tactile_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(tactile_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                               std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回一个时空样本

        Returns:
            {
                'tactile_video': (3, T, H, W),
                'visual_context': (3, H_v, W_v),
                'state_seq': (T, state_dim),
                'action_seq': (T, action_dim)
            }
        """
        ep_id, start_frame_idx = self.valid_samples[idx]

        # 1. 采样触觉视频序列
        tactile_frames = []
        for t in range(self.seq_length):
            frame_idx = start_frame_idx + t
            tactile_img = np.array(self.data['tactile_image'][frame_idx])  # (H, W, 3)

            # 转换为tensor
            tactile_tensor = self.tactile_transform(tactile_img)  # (3, H, W)
            tactile_frames.append(tactile_tensor)

        tactile_video = torch.stack(tactile_frames, dim=1)  # (3, T, H, W)

        # 2. Visual context (使用第一帧的视觉图像)
        try:
            visual_img = np.array(self.data['visual_image'][start_frame_idx])  # (H, W, 3)
            visual_context = self.visual_transform(visual_img)  # (3, H, W)
        except (KeyError, ValueError):
            # 如果没有视觉图像，使用零张量
            visual_context = torch.zeros(3, *self.visual_size)

        # 3. 状态序列
        try:
            state_list = []
            for t in range(self.seq_length):
                frame_idx = start_frame_idx + t
                state = np.array(self.data['robot_state'][frame_idx], dtype=np.float32)
                state_list.append(state)
            state_seq = np.stack(state_list, axis=0)  # (T, state_dim)

            # 归一化 (可选)
            if self.normalize_robot_state:
                state_mean = state_seq.mean(axis=0, keepdims=True)
                state_std = state_seq.std(axis=0, keepdims=True) + 1e-6
                state_seq = (state_seq - state_mean) / state_std

            state_seq = torch.from_numpy(state_seq).float()  # (T, state_dim)
        except (KeyError, ValueError):
            # 默认8维零向量
            state_seq = torch.zeros(self.seq_length, 8)

        # 4. 动作序列 (如果有的话，当前数据集可能没有)
        # 使用默认零向量
        action_seq = torch.zeros(self.seq_length, 8)

        return {
            'tactile_video': tactile_video,      # (3, T, H, W)
            'visual_context': visual_context,    # (3, H_v, W_v)
            'state_seq': state_seq,              # (T, state_dim)
            'action_seq': action_seq             # (T, action_dim)
        }


class SpatiotemporalTactileDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Spatiotemporal Tactile Dataset
    """

    def __init__(self,
                 dataset_path: str,
                 batch_size: int = 8,
                 num_workers: int = 4,
                 seq_length: int = 16,
                 visual_size: tuple = (224, 224),
                 tactile_size: tuple = (128, 160),
                 normalize_robot_state: bool = True,
                 val_ratio: float = 0.1,
                 seed: int = 42):
        super().__init__()

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_length = seq_length
        self.visual_size = visual_size
        self.tactile_size = tactile_size
        self.normalize_robot_state = normalize_robot_state
        self.val_ratio = val_ratio
        self.seed = seed

    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        # 创建完整数据集
        full_dataset = SpatiotemporalTactileDataset(
            dataset_path=self.dataset_path,
            seq_length=self.seq_length,
            visual_size=self.visual_size,
            tactile_size=self.tactile_size,
            normalize_robot_state=self.normalize_robot_state,
            train=True
        )

        # 划分训练集和验证集
        total_size = len(full_dataset)
        val_size = int(total_size * self.val_ratio)
        train_size = total_size - val_size

        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

        print(f"数据集划分:")
        print(f"  训练集: {len(self.train_dataset)} 样本")
        print(f"  验证集: {len(self.val_dataset)} 样本")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )

    def test_dataloader(self):
        return self.val_dataloader()


if __name__ == "__main__":
    # 测试代码
    print("Testing Spatiotemporal Tactile Dataset...")

    dataset = SpatiotemporalTactileDataset(
        dataset_path="data/residual_tactile_train_v2",
        seq_length=16,
        visual_size=(224, 224),
        tactile_size=(128, 160)
    )

    print(f"Dataset size: {len(dataset)}")

    # 测试一个样本
    sample = dataset[0]
    print(f"✓ Sample keys: {sample.keys()}")
    print(f"✓ tactile_video shape: {sample['tactile_video'].shape}")
    print(f"✓ visual_context shape: {sample['visual_context'].shape}")
    print(f"✓ state_seq shape: {sample['state_seq'].shape}")
    print(f"✓ action_seq shape: {sample['action_seq'].shape}")
    print("✓ Dataset test passed!")
