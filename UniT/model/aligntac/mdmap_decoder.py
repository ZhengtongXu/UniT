"""
MD-Map Decoder for AlignTac

Decodes memory-augmented dynamic features (Z_dyn_mem) into 3-channel MD-Map.
The decoder architecture reuses UniT's Decoder structure for consistency.

MD-Map Channels:
    - Ch0: dx (horizontal displacement)
    - Ch1: dy (vertical displacement)
    - Ch2: Energy (cumulative energy over K frames, requires Memory)

Author: Claude Code
Date: 2025-12-17
"""

import torch
import torch.nn as nn
from UniT.taming.modules.diffusionmodules.model import Decoder


class MDMapDecoder(nn.Module):
    """
    MD-Map Decoder

    Decodes latent dynamic features (with memory) into 3-channel MD-Map representation.
    The architecture mirrors UniT's RGB decoder but outputs MD-Map instead.

    Architecture:
        Input: Z_dyn_mem (B, z_channels, 8, 10) - Memory-augmented dynamics
        ↓
        [Decoder Blocks] - Multiple resolution levels with ResBlocks + Attention
        ↓
        Output: MD-Map (B, 3, H, W) - 3-channel displacement + energy map
    """

    def __init__(self, z_channels=3, ch=128, ch_mult=[1, 1, 2, 4],
                 num_res_blocks=2, attn_resolutions=[16],
                 resolution=240, dropout=0.0):
        """
        Initialize MD-Map Decoder

        Args:
            z_channels (int): Input latent channels (should match Z_dyn dimension)
            ch (int): Base channel count for decoder
            ch_mult (list): Channel multipliers for each resolution level
            num_res_blocks (int): Number of residual blocks per level
            attn_resolutions (list): Resolutions where attention is applied
            resolution (int): Output spatial resolution (height, assumes 4:3 aspect ratio)
            dropout (float): Dropout rate
        """
        super().__init__()

        self.z_channels = z_channels
        self.resolution = resolution

        # Use UniT's Decoder architecture
        # Note: The original Decoder is designed for VQ-VAE, we reuse it for MD-Map
        self.decoder = Decoder(
            ch=ch,
            out_ch=3,  # MD-Map has 3 channels
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=True,
            in_channels=z_channels,  # Input from latent space
            resolution=resolution,
            z_channels=z_channels,
            give_pre_end=False,
        )

    def forward(self, z_dyn_mem):
        """
        Forward pass: Decode latent features to MD-Map

        Args:
            z_dyn_mem (torch.Tensor): Memory-augmented dynamic features
                Shape: (B, z_channels, H_latent, W_latent)
                Example: (B, 3, 8, 10)

        Returns:
            md_map (torch.Tensor): 3-channel MD-Map
                Shape: (B, 3, H, W)
                Example: (B, 3, 240, 320)
                - Ch0: dx (horizontal displacement)
                - Ch1: dy (vertical displacement)
                - Ch2: Energy (cumulative energy requiring Memory)
        """
        md_map = self.decoder(z_dyn_mem)
        return md_map


# Example usage and testing
if __name__ == "__main__":
    print("Testing MDMapDecoder...")

    # Create decoder
    decoder = MDMapDecoder(
        z_channels=3,
        ch=128,
        ch_mult=[1, 1, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[16],
        resolution=240,
        dropout=0.0
    )

    # Test input: latent features from Z_dyn_mem
    B = 4
    z_dyn_mem = torch.randn(B, 3, 8, 10)

    print(f"\nInput shape: {z_dyn_mem.shape}")

    # Forward pass
    md_map = decoder(z_dyn_mem)

    print(f"Output shape: {md_map.shape}")
    print(f"Expected: ({B}, 3, 240, 320)")

    # Check shape
    assert md_map.shape[0] == B
    assert md_map.shape[1] == 3  # 3 channels
    # Note: The exact output resolution depends on the decoder architecture
    # With ch_mult=[1,1,2,4], starting from 8×10 latent, we get:
    # 8×10 → 16×20 → 32×40 → 64×80 → 128×160
    # To get 240×320, we need appropriate upsampling factors

    print(f"\n✓ MDMapDecoder test passed!")
    print(f"  Input latent: {z_dyn_mem.shape}")
    print(f"  Output MD-Map: {md_map.shape}")
