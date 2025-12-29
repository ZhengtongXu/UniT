"""
Memory Module for AlignTac

Implements FIFO queue + Cross-Attention mechanism for temporal feature aggregation.
The memory acts on dynamic features (Z_dyn) to enable cumulative energy computation
and distinguish between static success vs oscillating jamming states.

Author: Claude Code
Date: 2025-12-17
"""

import torch
import torch.nn as nn
from collections import deque


class MemoryModule(nn.Module):
    """
    FIFO Queue + Cross-Attention Memory Module

    This module maintains a sliding window of past K frames' features using a FIFO queue,
    and aggregates temporal information via cross-attention mechanism.

    Architecture:
        Input: z_dyn (current frame's dynamic features)
        ↓
        [FIFO Queue] - Store past K frames (detached)
        ↓
        [Cross-Attention] - Query: current, Key/Value: history
        ↓
        [Residual + LayerNorm]
        ↓
        Output: z_dyn_mem (memory-augmented dynamic features)
    """

    def __init__(self, embed_dim=3, spatial_size=(8, 10), K=5,
                 num_heads=1, dropout=0.1):
        """
        Initialize Memory Module

        Args:
            embed_dim (int): Feature dimension (channel count)
            spatial_size (tuple): Spatial dimensions (H, W) of the feature map
            K (int): History window size (number of frames to remember)
            num_heads (int): Number of attention heads for multi-head attention
            dropout (float): Dropout rate for attention
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.spatial_size = spatial_size
        self.K = K
        
        # FIFO queue for storing history (initialized dynamically per batch)
        self.memory_queue = None
        
        # Cross-Attention layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Use batch_first format: (B, L, D)
        )
        
        # LayerNorm for residual connection
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, z_dyn):
        """
        Forward pass: Aggregate temporal information via cross-attention

        Args:
            z_dyn (torch.Tensor): Current frame's dynamic features
                Shape: (B, C, H, W)

        Returns:
            z_dyn_mem (torch.Tensor): Memory-augmented dynamic features
                Shape: (B, C, H, W)
        """
        B, C, H, W = z_dyn.shape
        assert C == self.embed_dim, f"Channel mismatch: {C} vs {self.embed_dim}"
        assert (H, W) == self.spatial_size, f"Spatial size mismatch: {(H, W)} vs {self.spatial_size}"

        # Flatten spatial dimensions: (B, C, H, W) → (B, H*W, C)
        z_flat = z_dyn.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        # Check if initialization is needed
        need_init = False
        if self.memory_queue is None:
            need_init = True
        elif len(self.memory_queue) > 0 and self.memory_queue[0].shape[0] != B:
            # Batch size changed - need to reinitialize
            need_init = True
            self.reset_memory()

        # 1. 实现"首帧复制"初始化 (Replication Padding)
        if need_init:
            self._init_memory_queue(B, device=z_dyn.device)
            # 将当前帧复制 K 次填满队列
            z_flat_detached = z_flat.detach()
            for _ in range(self.K):
                self.memory_queue.append(z_flat_detached)

        # 2. 统一的更新逻辑
        # 入队：将当前帧加入队列尾部
        self.memory_queue.append(z_flat.detach())
        # 出队：如果队列长度 > K，弹出最旧的一帧
        if len(self.memory_queue) > self.K:
            self.memory_queue.popleft()

        # 3. 修正 Attention 数据流
        # Key/Value 构建：使用队列中所有 K 帧（包含当前帧）
        # Stack: (K, B, HW, C) → (B, K*HW, C)
        all_frames = torch.stack(list(self.memory_queue), dim=0)  # (K, B, HW, C)
        K_mem = all_frames.shape[0]
        memory_flat = all_frames.permute(1, 0, 2, 3).reshape(B, K_mem * H * W, C)

        # Cross-Attention (对于第一帧，这是 Self-Attention，因为 K=V=Q 都来自同一帧的副本)
        # Query: current frame (B, HW, C)
        # Key/Value: all K frames including current (B, K*HW, C)
        attn_output, attn_weights = self.cross_attn(
            query=z_flat,
            key=memory_flat,
            value=memory_flat
        )  # attn_output: (B, HW, C)

        # Residual connection + LayerNorm
        # 确保每一帧（包括第一帧）都完整经过 Cross-Attention -> Residual Add -> LayerNorm
        z_mem_flat = self.norm(z_flat + attn_output)

        # Reshape back to spatial format: (B, HW, C) → (B, C, H, W)
        z_dyn_mem = z_mem_flat.permute(0, 2, 1).reshape(B, C, H, W)

        return z_dyn_mem
    
    def _init_memory_queue(self, batch_size, device):
        """
        Initialize FIFO memory queue

        Args:
            batch_size (int): Batch size (not used in deque, but for future extensions)
            device (torch.device): Device to place tensors on
        """
        self.memory_queue = deque(maxlen=self.K)
    
    def reset_memory(self):
        """
        Reset memory queue (call this at the start of a new episode)

        This ensures that features from previous episodes don't contaminate
        the current episode's memory.
        """
        self.memory_queue = None


# Example usage and testing
if __name__ == "__main__":
    # Test the MemoryModule
    print("Testing MemoryModule...")
    
    # Create module
    memory = MemoryModule(embed_dim=3, spatial_size=(8, 10), K=5)
    
    # Simulate processing a sequence of frames
    B, C, H, W = 2, 3, 8, 10
    num_frames = 10
    
    print(f"\nProcessing {num_frames} frames with shape ({B}, {C}, {H}, {W})")
    
    for t in range(num_frames):
        # Random input features
        z_dyn = torch.randn(B, C, H, W)
        
        # Forward pass
        z_dyn_mem = memory(z_dyn)
        
        print(f"Frame {t}: Input shape {z_dyn.shape} → Output shape {z_dyn_mem.shape}")
        
        # Check that output preserves shape
        assert z_dyn_mem.shape == z_dyn.shape
    
    print("\n✓ MemoryModule test passed!")
    
    # Test reset
    print("\nTesting reset_memory()...")
    memory.reset_memory()
    assert memory.memory_queue is None
    print("✓ Reset successful!")
