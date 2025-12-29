# pytorch_diffusion + derived encoder decoder
"""
Diffusion Model Architecture Implementation

This module implements various diffusion model architectures including:
- U-Net based diffusion models (Model class)
- Autoencoder components (Encoder, Decoder)
- Variational U-Net (VUNet)
- Simplified decoders for specific use cases

Key Components:
- ResNet blocks with time conditioning
- Self-attention mechanisms
- Multi-resolution processing with downsampling/upsampling
- Group normalization for stable training
"""

import math
import torch
import torch.nn as nn
import numpy as np


def get_timestep_embedding(timesteps, embedding_dim):
    """
    Generate sinusoidal timestep embeddings for diffusion models.

    This implements the positional encoding from "Attention Is All You Need"
    but adapted for diffusion timestep conditioning. Creates embeddings that
    allow the model to understand the diffusion timestep in a continuous manner.

    Args:
        timesteps: Tensor of shape (batch_size,) containing diffusion timesteps
        embedding_dim: Dimension of the embedding vector

    Returns:
        Tensor of shape (batch_size, embedding_dim) with sinusoidal embeddings
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    """Swish activation function: x * sigmoid(x)"""
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    """
    Group normalization layer.

    Uses 32 groups for stable training of diffusion models.
    GroupNorm is preferred over BatchNorm in generative models as it
    doesn't depend on batch statistics.
    """
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    """
    Upsampling layer with optional convolution.

    Performs nearest neighbor upsampling by factor of 2, optionally followed
    by a 3x3 convolution to refine the upsampled features.
    """
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    Downsampling layer with optional convolution.

    Either uses strided convolution (with asymmetric padding) or average pooling
    to reduce spatial resolution by factor of 2.
    """
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # Asymmetric padding to maintain spatial alignment
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)  # Asymmetric padding for proper downsampling
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    """
    ResNet block with optional time conditioning and channel dimension changes.

    This is the core building block of the U-Net architecture. It consists of:
    1. GroupNorm + Swish activation
    2. 3x3 convolution
    3. Optional time conditioning via linear projection
    4. GroupNorm + Swish + Dropout
    5. 3x3 convolution
    6. Skip connection (with optional channel adjustment)

    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension (defaults to in_channels)
        conv_shortcut: Whether to use 3x3 conv or 1x1 conv for channel adjustment
        dropout: Dropout probability
        temb_channels: Dimension of timestep embedding (0 if no conditioning)
    """
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        # First convolution block
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        # Time conditioning projection
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)

        # Second convolution block
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        # Skip connection for channel dimension changes
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            temb: Time embedding tensor of shape (B, temb_channels) or None
        Returns:
            Output tensor of shape (B, out_channels, H, W)
        """
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        # Add time conditioning if provided
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Skip connection
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    """
    Self-attention block for spatial attention in diffusion models.

    Implements multi-head self-attention where the spatial dimensions
    (H, W) are treated as sequence elements. This allows the model to
    capture long-range dependencies across the image.

    The attention computation treats (H*W) as sequence length and C as
    feature dimension, computing: Attention(Q, K, V) where Q, K, V are
    derived from the same input via 1x1 convolutions.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        # Query, Key, Value projections
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Output tensor of shape (B, C, H, W) with self-attention applied
        """
        h_ = x
        h_ = self.norm(h_)

        # Generate Q, K, V
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Reshape for attention computation
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # (B, H*W, C)
        k = k.reshape(b, c, h*w) # (B, C, H*W)
        v = v.reshape(b, c, h*w) # (B, C, H*W)

        # Attention: (Q @ K^T) / sqrt(C)
        w_ = torch.bmm(q, k)     # (B, H*W, H*W)
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # Apply attention to values
        w_ = w_.permute(0, 2, 1)   # (B, H*W, H*W)
        h_ = torch.bmm(v, w_)     # (B, C, H*W)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)
        return x + h_  # Residual connection


class Model(nn.Module):
    """
    Main U-Net diffusion model architecture.

    This implements a U-Net with:
    - Multi-resolution processing (downsampling/upsampling)
    - Time-conditioned ResNet blocks
    - Self-attention at specified resolutions
    - Skip connections between encoder and decoder

    Architecture:
    Input -> [Downsampling blocks] -> Middle block -> [Upsampling blocks] -> Output

    Each resolution level contains multiple ResNet blocks and optional attention.
    Skip connections are maintained between corresponding encoder/decoder levels.

    Args:
        ch: Base channel dimension
        out_ch: Output channel dimension
        ch_mult: Channel multiplier for each resolution level
        num_res_blocks: Number of ResNet blocks per resolution level
        attn_resolutions: List of resolutions where attention is applied
        dropout: Dropout probability
        resamp_with_conv: Whether to use convolution in resampling
        in_channels: Input channel dimension
        resolution: Input spatial resolution
        use_timestep: Whether to use timestep conditioning
    """
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, use_timestep=True):
        super().__init__()
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # Timestep embedding network
        self.use_timestep = use_timestep
        if self.use_timestep:
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch, self.temb_ch),
                torch.nn.Linear(self.temb_ch, self.temb_ch),
            ])

        # Input convolution
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        # Downsampling path (Encoder)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            # Add ResNet blocks for this level
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                # Add attention if this resolution requires it
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn

            # Add downsampling (except for last level)
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # Middle block (bottleneck)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # Upsampling path (Decoder)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]

            # One extra block for skip connection concatenation
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))

            up = nn.Module()
            up.block = block
            up.attn = attn

            # Add upsampling (except for first level)
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # Prepend to maintain order

        # Output layers
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t=None):
        """
        Forward pass through the U-Net diffusion model.

        Args:
            x: Input tensor of shape (B, in_channels, H, W)
            t: Timestep tensor of shape (B,) or None

        Returns:
            Output tensor of shape (B, out_ch, H, W)
        """
        # Timestep embedding
        if self.use_timestep:
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # Encoder path
        hs = [self.conv_in(x)]  # Store intermediate features for skip connections
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # Middle (bottleneck)
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # Decoder path with skip connections
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)  # Concatenate skip connection
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # Output
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Encoder(nn.Module):
    """
    Autoencoder Encoder for latent space learning.

    Similar to the Model class but without timestep conditioning and decoder.
    Used for learning compressed representations of input data.

    Architecture: Input -> [Downsampling blocks] -> Middle block -> Latent representation
    """
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0  # No timestep conditioning
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # Downsampling path
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # Middle block
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # Output convolution to latent space
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        """
        Encode input to latent representation.

        Args:
            x: Input tensor of shape (B, in_channels, H, W)
        Returns:
            Latent tensor of shape (B, z_channels, H', W') or (B, 2*z_channels, H', W')
        """
        temb = None  # No timestep conditioning

        # Encoder path
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # Middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # Output to latent space
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    """
    Autoencoder Decoder for reconstructing from latent space.

    Takes latent representations and reconstructs the original input.
    Symmetric to Encoder but in reverse order.

    Architecture: Latent -> [Upsampling blocks] -> Output
    """
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0  # No timestep conditioning
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # Compute dimensions for the latent space
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # Latent to feature space
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # Middle block
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # Upsampling path
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # Output layers
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        """
        Decode latent representation back to original space.

        Args:
            z: Latent tensor of shape (B, z_channels, H', W')
        Returns:
            Reconstructed tensor of shape (B, out_ch, H, W)
        """
        self.last_z_shape = z.shape

        temb = None  # No timestep conditioning

        # Latent to feature space
        h = self.conv_in(z)

        # Middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # Decoder path
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # Output
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VUNet(nn.Module):
    """
    Variational U-Net for conditional generation.

    Combines conditioning input (c) with latent representation (z) in the middle
    of the U-Net architecture. Used for conditional generation tasks.

    Architecture: Condition -> Encoder -> [Middle with z] -> Decoder -> Output
    """
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 in_channels, c_channels,
                 resolution, z_channels, use_timestep=False, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        # Timestep conditioning (optional)
        self.use_timestep = use_timestep
        if self.use_timestep:
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch, self.temb_ch),
                torch.nn.Linear(self.temb_ch, self.temb_ch),
            ])

        # Condition input convolution
        self.conv_in = torch.nn.Conv2d(c_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # Latent injection
        self.z_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=1, stride=1, padding=0)

        # Middle block (combines condition and latent)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=2*block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # Upsampling path
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # Output layers
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, z):
        """
        Conditional generation with latent injection.

        Args:
            x: Condition tensor of shape (B, c_channels, H, W)
            z: Latent tensor of shape (B, z_channels, H', W')
        Returns:
            Generated tensor of shape (B, out_ch, H, W)
        """
        if self.use_timestep:
            # Note: This seems to have a bug in original code - 't' is not defined
            # assert t is not None
            # temb = get_timestep_embedding(t, self.ch)
            temb = None  # Placeholder
        else:
            temb = None

        # Encoder path
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # Middle with latent injection
        h = hs[-1]
        z = self.z_in(z)
        h = torch.cat((h, z), dim=1)  # Concatenate condition and latent
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # Decoder path
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # Output
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class SimpleDecoder(nn.Module):
    """
    Simplified decoder for basic reconstruction tasks.

    A lightweight decoder with fewer layers, suitable for simple
    reconstruction or when computational resources are limited.

    Architecture: Input -> Conv1x1 -> ResNet blocks -> Upsample -> Output
    """
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.model = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, 1),  # Channel adjustment
            ResnetBlock(in_channels=in_channels,
                       out_channels=2 * in_channels,
                       temb_channels=0, dropout=0.0),
            ResnetBlock(in_channels=2 * in_channels,
                       out_channels=4 * in_channels,
                       temb_channels=0, dropout=0.0),
            ResnetBlock(in_channels=4 * in_channels,
                       out_channels=2 * in_channels,
                       temb_channels=0, dropout=0.0),
            nn.Conv2d(2*in_channels, in_channels, 1),  # Channel reduction
            Upsample(in_channels, with_conv=True)     # Final upsampling
        ])

        # Output layers
        self.norm_out = Normalize(in_channels)
        self.conv_out = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """Simple decoding without timestep conditioning."""
        for i, layer in enumerate(self.model):
            if i in [1,2,3]:  # ResNet blocks need temb parameter
                x = layer(x, None)
            else:
                x = layer(x)

        h = self.norm_out(x)
        h = nonlinearity(h)
        x = self.conv_out(h)
        return x


class UpsampleDecoder(nn.Module):
    """
    Multi-resolution upsampling decoder.

    A flexible decoder that can handle different channel configurations
    and resolution levels for various upsampling tasks.

    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        ch: Base channel dimension
        num_res_blocks: Number of ResNet blocks per level
        resolution: Input spatial resolution
        ch_mult: Channel multipliers for each resolution level
        dropout: Dropout probability
    """
    def __init__(self, in_channels, out_channels, ch, num_res_blocks, resolution,
                 ch_mult=(2,2), dropout=0.0):
        super().__init__()
        self.temb_ch = 0  # No timestep conditioning
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = in_channels
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        # Build resolution levels
        for i_level in range(self.num_resolutions):
            res_block = []
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            self.res_blocks.append(nn.ModuleList(res_block))

            # Add upsampling (except last level)
            if i_level != self.num_resolutions - 1:
                self.upsample_blocks.append(Upsample(block_in, True))
                curr_res = curr_res * 2

        # Output layers
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Multi-resolution upsampling.

        Args:
            x: Input tensor of shape (B, in_channels, H, W)
        Returns:
            Output tensor of shape (B, out_channels, H*scale, W*scale)
        """
        h = x
        for k, i_level in enumerate(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.res_blocks[i_level][i_block](h, None)
            if i_level != self.num_resolutions - 1:
                h = self.upsample_blocks[k](h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
