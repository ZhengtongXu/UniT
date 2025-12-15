"""
AlignTac Dataset: 双帧触觉数据集

特点:
1. 双帧输入: (I_t, I_{t-1}) 用于捕捉动态
2. 预计算光流: 从数据集中加载,无需在线计算
3. 物理阶段标签: 可选的监督信号
4. 支持多种数据格式: zarr, hdf5, 文件夹
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union
import json
import zarr


class AlignTacDataset(Dataset):
    """
    AlignTac双帧触觉数据集
    
    数据组织 (zarr格式):
        data/
        ├── tactile_image/           # (N, H, W, 3) uint8
        ├── optical_flow/            # (N, H, W, 2) float32 (预计算)
        ├── phase_label/             # (N,) int64 (可选)
        └── meta/
            └── episode_ends        # 每个episode的结束索引
    """
    
    def __init__(
        self,
        data_path: str,
        image_size: int = 224,
        split: str = 'train',
        val_ratio: float = 0.1,
        frame_gap: int = 1,  # I_t和I_{t-1}的间隔
        use_flow: bool = True,
        use_phase_label: bool = False,
        augment: bool = True,
        normalize: bool = True,
        # 数据格式
        data_format: str = 'zarr',  # 'zarr', 'hdf5', 'folder'
    ):
        super().__init__()
        
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.split = split
        self.val_ratio = val_ratio
        self.frame_gap = frame_gap
        self.use_flow = use_flow
        self.use_phase_label = use_phase_label
        self.augment = augment and (split == 'train')
        self.normalize = normalize
        self.data_format = data_format
        
        # 加载数据
        self._load_data()
        
        # 构建有效索引 (跳过episode边界)
        self._build_valid_indices()
        
        # 按split划分
        self._split_data()
        
        print(f"[AlignTacDataset] Loaded {len(self.valid_indices)} samples for {split}")
    
    def _load_data(self):
        """加载数据"""
        if self.data_format == 'zarr':
            self._load_zarr()
        elif self.data_format == 'hdf5':
            self._load_hdf5()
        elif self.data_format == 'folder':
            self._load_folder()
        else:
            raise ValueError(f"Unknown data format: {self.data_format}")
    
    def _load_zarr(self):
        """加载zarr格式数据"""
        root = zarr.open(str(self.data_path / 'replay_buffer.zarr'), mode='r')
        
        # 触觉图像
        self.images = root['data']['tactile_image']  # (N, H, W, 3)
        self.total_frames = len(self.images)
        
        # 光流 (预计算)
        if self.use_flow and 'optical_flow' in root['data']:
            self.flows = root['data']['optical_flow']  # (N, H, W, 2)
        else:
            self.flows = None
        
        # 物理阶段标签
        if self.use_phase_label and 'phase_label' in root['data']:
            self.phase_labels = root['data']['phase_label'][:]  # (N,)
        else:
            self.phase_labels = None
        
        # Episode边界
        if 'meta' in root and 'episode_ends' in root['meta']:
            self.episode_ends = root['meta']['episode_ends'][:]
        else:
            # 假设单个episode
            self.episode_ends = np.array([self.total_frames])
    
    def _load_hdf5(self):
        """加载HDF5格式数据"""
        import h5py
        
        self.h5_file = h5py.File(self.data_path / 'data.h5', 'r')
        
        self.images = self.h5_file['tactile_image']
        self.total_frames = len(self.images)
        
        if self.use_flow and 'optical_flow' in self.h5_file:
            self.flows = self.h5_file['optical_flow']
        else:
            self.flows = None
        
        if self.use_phase_label and 'phase_label' in self.h5_file:
            self.phase_labels = self.h5_file['phase_label'][:]
        else:
            self.phase_labels = None
        
        if 'episode_ends' in self.h5_file:
            self.episode_ends = self.h5_file['episode_ends'][:]
        else:
            self.episode_ends = np.array([self.total_frames])
    
    def _load_folder(self):
        """加载文件夹格式数据"""
        # 图像文件列表
        image_dir = self.data_path / 'images'
        self.image_files = sorted(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')))
        self.total_frames = len(self.image_files)
        self.images = None  # 延迟加载
        
        # 光流
        flow_dir = self.data_path / 'flows'
        if self.use_flow and flow_dir.exists():
            self.flow_files = sorted(list(flow_dir.glob('*.npy')))
        else:
            self.flow_files = None
        self.flows = None
        
        # 阶段标签
        label_file = self.data_path / 'phase_labels.npy'
        if self.use_phase_label and label_file.exists():
            self.phase_labels = np.load(label_file)
        else:
            self.phase_labels = None
        
        # Episode边界
        meta_file = self.data_path / 'meta.json'
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
                self.episode_ends = np.array(meta.get('episode_ends', [self.total_frames]))
        else:
            self.episode_ends = np.array([self.total_frames])
    
    def _build_valid_indices(self):
        """构建有效帧索引 (跳过episode开头帧)"""
        valid = []
        episode_starts = np.concatenate([[0], self.episode_ends[:-1]])
        
        for start, end in zip(episode_starts, self.episode_ends):
            # 跳过每个episode的前frame_gap帧
            for idx in range(start + self.frame_gap, end):
                valid.append(idx)
        
        self.all_valid_indices = np.array(valid)
    
    def _split_data(self):
        """按train/val划分"""
        n_total = len(self.all_valid_indices)
        n_val = int(n_total * self.val_ratio)
        
        # 固定随机种子保证可复现
        rng = np.random.RandomState(42)
        indices = rng.permutation(n_total)
        
        if self.split == 'train':
            self.valid_indices = self.all_valid_indices[indices[n_val:]]
        elif self.split == 'val':
            self.valid_indices = self.all_valid_indices[indices[:n_val]]
        else:  # 'all'
            self.valid_indices = self.all_valid_indices
    
    def _load_image(self, idx: int) -> np.ndarray:
        """加载单帧图像"""
        if self.data_format == 'folder':
            img = Image.open(self.image_files[idx]).convert('RGB')
            return np.array(img)
        else:
            return np.array(self.images[idx])
    
    def _load_flow(self, idx: int) -> Optional[np.ndarray]:
        """加载单帧光流"""
        if self.flows is None:
            return None
        
        if self.data_format == 'folder':
            return np.load(self.flow_files[idx])
        else:
            return np.array(self.flows[idx])
    
    def _resize(self, img: np.ndarray, size: int) -> np.ndarray:
        """Resize图像"""
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((size, size), Image.BILINEAR)
        return np.array(pil_img)
    
    def _resize_flow(self, flow: np.ndarray, size: int) -> np.ndarray:
        """Resize光流 (需要调整幅值)"""
        h, w = flow.shape[:2]
        scale_x = size / w
        scale_y = size / h
        
        # Resize
        flow_u = Image.fromarray(flow[:, :, 0]).resize((size, size), Image.BILINEAR)
        flow_v = Image.fromarray(flow[:, :, 1]).resize((size, size), Image.BILINEAR)
        
        # 调整幅值
        flow_resized = np.stack([
            np.array(flow_u) * scale_x,
            np.array(flow_v) * scale_y
        ], axis=-1)
        
        return flow_resized
    
    def _augment(self, img_t: np.ndarray, img_prev: np.ndarray, 
                 flow: Optional[np.ndarray]) -> Tuple:
        """数据增强 (同时应用于两帧和光流)"""
        # 水平翻转
        if np.random.rand() > 0.5:
            img_t = np.fliplr(img_t).copy()
            img_prev = np.fliplr(img_prev).copy()
            if flow is not None:
                flow = np.fliplr(flow).copy()
                flow[:, :, 0] = -flow[:, :, 0]  # 翻转u分量
        
        # 垂直翻转
        if np.random.rand() > 0.5:
            img_t = np.flipud(img_t).copy()
            img_prev = np.flipud(img_prev).copy()
            if flow is not None:
                flow = np.flipud(flow).copy()
                flow[:, :, 1] = -flow[:, :, 1]  # 翻转v分量
        
        # 颜色抖动 (只对图像)
        if np.random.rand() > 0.5:
            brightness = 0.8 + np.random.rand() * 0.4  # [0.8, 1.2]
            img_t = np.clip(img_t * brightness, 0, 255).astype(np.uint8)
            img_prev = np.clip(img_prev * brightness, 0, 255).astype(np.uint8)
        
        return img_t, img_prev, flow
    
    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """转换为tensor并归一化"""
        # (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        
        if self.normalize:
            tensor = tensor / 127.5 - 1.0  # [-1, 1]
        else:
            tensor = tensor / 255.0  # [0, 1]
        
        return tensor
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with:
                - stacked_image: (6, H, W) concat(I_t, I_{t-1})
                - image_t: (3, H, W) 当前帧 (RGB重建目标)
                - flow_gt: (2, H, W) 光流GT (如果可用)
                - phase_label: () 物理阶段标签 (如果可用)
        """
        frame_idx = self.valid_indices[idx]
        prev_idx = frame_idx - self.frame_gap
        
        # 加载双帧
        img_t = self._load_image(frame_idx)
        img_prev = self._load_image(prev_idx)
        
        # 加载光流
        flow = self._load_flow(frame_idx) if self.use_flow else None
        
        # Resize
        img_t = self._resize(img_t, self.image_size)
        img_prev = self._resize(img_prev, self.image_size)
        if flow is not None:
            flow = self._resize_flow(flow, self.image_size)
        
        # 数据增强
        if self.augment:
            img_t, img_prev, flow = self._augment(img_t, img_prev, flow)
        
        # 转换为tensor
        img_t_tensor = self._to_tensor(img_t)
        img_prev_tensor = self._to_tensor(img_prev)
        
        # 堆叠双帧: (6, H, W)
        stacked = torch.cat([img_t_tensor, img_prev_tensor], dim=0)
        
        output = {
            'stacked_image': stacked,
            'image_t': img_t_tensor,
        }
        
        # 光流
        if flow is not None:
            # (H, W, 2) -> (2, H, W)
            flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).float()
            output['flow_gt'] = flow_tensor
        
        # 物理阶段标签
        if self.phase_labels is not None:
            output['phase_label'] = torch.tensor(self.phase_labels[frame_idx], dtype=torch.long)
        
        return output


class AlignTacDataModule:
    """
    AlignTac数据模块 (用于训练脚本)
    """
    
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 8,
        image_size: int = 224,
        val_ratio: float = 0.1,
        frame_gap: int = 1,
        use_flow: bool = True,
        use_phase_label: bool = False,
        augment: bool = True,
        data_format: str = 'zarr',
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.val_ratio = val_ratio
        self.frame_gap = frame_gap
        self.use_flow = use_flow
        self.use_phase_label = use_phase_label
        self.augment = augment
        self.data_format = data_format
    
    def setup(self):
        """创建数据集"""
        self.train_dataset = AlignTacDataset(
            data_path=self.data_path,
            image_size=self.image_size,
            split='train',
            val_ratio=self.val_ratio,
            frame_gap=self.frame_gap,
            use_flow=self.use_flow,
            use_phase_label=self.use_phase_label,
            augment=self.augment,
            data_format=self.data_format,
        )
        
        self.val_dataset = AlignTacDataset(
            data_path=self.data_path,
            image_size=self.image_size,
            split='val',
            val_ratio=self.val_ratio,
            frame_gap=self.frame_gap,
            use_flow=self.use_flow,
            use_phase_label=self.use_phase_label,
            augment=False,
            data_format=self.data_format,
        )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )


def prepare_flow_offline(
    data_path: str,
    output_path: str = None,
    method: str = 'farneback',
    device: str = 'cuda',
):
    """
    预计算光流的工具函数
    
    Args:
        data_path: 输入数据路径 (zarr格式)
        output_path: 输出路径,None则原地添加
        method: 光流算法 ('farneback', 'raft')
        device: 计算设备
    
    Usage:
        python -c "from dataset.aligntac_dataset import prepare_flow_offline; prepare_flow_offline('/path/to/data')"
    """
    import cv2
    from tqdm import tqdm
    
    root = zarr.open(data_path, mode='r+' if output_path is None else 'r')
    images = root['data']['tactile_image']
    n_frames = len(images)
    
    # 准备输出
    if output_path is None:
        # 原地添加
        if 'optical_flow' not in root['data']:
            h, w = images[0].shape[:2]
            root['data'].create_dataset(
                'optical_flow',
                shape=(n_frames, h, w, 2),
                dtype=np.float32,
                chunks=(1, h, w, 2)
            )
        flows = root['data']['optical_flow']
    else:
        out_root = zarr.open(output_path, mode='w')
        h, w = images[0].shape[:2]
        flows = out_root.create_dataset(
            'optical_flow',
            shape=(n_frames, h, w, 2),
            dtype=np.float32,
            chunks=(1, h, w, 2)
        )
    
    print(f"Computing optical flow with method: {method}")
    
    if method == 'farneback':
        # OpenCV Farneback光流
        for i in tqdm(range(1, n_frames)):
            prev_gray = cv2.cvtColor(images[i-1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flows[i] = flow
        flows[0] = np.zeros_like(flows[1])  # 第一帧无光流
    
    elif method == 'raft':
        # RAFT光流 (需要安装RAFT)
        try:
            from torchvision.models.optical_flow import raft_large
            import torchvision.transforms.functional as TF
        except ImportError:
            raise ImportError("Please install torchvision >= 0.13 for RAFT")
        
        model = raft_large(pretrained=True).to(device).eval()
        
        with torch.no_grad():
            for i in tqdm(range(1, n_frames)):
                prev = TF.to_tensor(images[i-1]).unsqueeze(0).to(device)
                curr = TF.to_tensor(images[i]).unsqueeze(0).to(device)
                
                flow = model(prev, curr)[-1]  # 最后一个迭代
                flows[i] = flow[0].permute(1, 2, 0).cpu().numpy()
        
        flows[0] = np.zeros_like(flows[1])
    
    print(f"Optical flow computed and saved to {output_path or data_path}")
