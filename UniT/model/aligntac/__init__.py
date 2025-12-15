"""
AlignTac: 基于VQ-VAE的几何-动力学解耦触觉表征

模块:
- AlignTacVQVAE: 双流解耦VQ-VAE主模型
- AlignTacLoss: 多任务损失函数
- OrthogonalityLoss: 正交约束损失
- FlowLoss: 光流预测损失
"""

from .aligntac_vqvae import AlignTacVQVAE
from .losses import (
    AlignTacLoss,
    OrthogonalityLoss,
    FlowLoss,
    PhaseClassificationLoss,
    EPE,
    compute_codebook_usage,
)

__all__ = [
    'AlignTacVQVAE',
    'AlignTacLoss',
    'OrthogonalityLoss',
    'FlowLoss',
    'PhaseClassificationLoss',
    'EPE',
    'compute_codebook_usage',
]
