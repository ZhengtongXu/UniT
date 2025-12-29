"""
Memory-Augmented VQ-VAE for AlignTac

Implements dual-stream architecture with geometry-dynamics disentanglement
and memory-augmented temporal modeling.

Architecture:
    Input (Tactile Image)
    ↓
    [Shared Encoder]
    ↓
    ├─[Proj_Geo]──→ Z_geo ──→ [Quantize_Geo] ──→ [Decoder_RGB] ──→ RGB Reconstruction
    │
    └─[Proj_Dyn]──→ Z_dyn ──→ [Memory Module] ──→ Z_dyn_mem ──→ [Quantize_Dyn] ──→ [Decoder_MDMap] ──→ MD-Map Prediction

Key Components:
- Geometry Branch: Learns static contact geometry (RGB reconstruction)
- Dynamics Branch: Learns marker dynamics with memory (MD-Map prediction)
- Memory Module: FIFO + Cross-Attention for temporal aggregation

Author: Claude Code
Date: 2025-12-17
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from UniT.taming.modules.diffusionmodules.model import Encoder, Decoder
from UniT.taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from UniT.taming.logging_util import instantiate_from_config

from .memory_module import MemoryModule
from .mdmap_decoder import MDMapDecoder


class MemoryVQVAE(pl.LightningModule):
    """
    Memory-Augmented Dual-Stream VQ-VAE

    This model extends the standard VQ-VAE with:
    1. Dual-stream architecture (geometry + dynamics)
    2. Memory module for temporal modeling
    3. Dual reconstruction targets (RGB + MD-Map)
    """

    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed=1024,
                 geo_embed_dim=3,
                 dyn_embed_dim=3,
                 memory_K=5,
                 memory_num_heads=1,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 mdmap_key="md_map",
                 monitor=None,
                 lambda_rgb=1.0,
                 lambda_mdmap=1.0,
                 ):
        """
        Initialize Memory-Augmented VQ-VAE

        Args:
            ddconfig (dict): Encoder/Decoder configuration
            lossconfig (dict): Loss function configuration
            n_embed (int): Codebook size (shared for both branches)
            geo_embed_dim (int): Embedding dimension for geometry branch
            dyn_embed_dim (int): Embedding dimension for dynamics branch
            memory_K (int): Memory window size (number of frames)
            memory_num_heads (int): Number of attention heads in Memory Module
            ckpt_path (str): Path to pretrained checkpoint
            ignore_keys (list): Keys to ignore when loading checkpoint
            image_key (str): Key for input image in batch
            mdmap_key (str): Key for MD-Map GT in batch
            monitor (str): Metric to monitor for checkpointing
            lambda_rgb (float): Weight for RGB reconstruction loss
            lambda_mdmap (float): Weight for MD-Map prediction loss
        """
        super().__init__()

        self.image_key = image_key
        self.mdmap_key = mdmap_key
        self.lambda_rgb = lambda_rgb
        self.lambda_mdmap = lambda_mdmap

        # === Shared Encoder ===
        # Encodes input image to shared latent representation
        # Example: (B, 3, 128, 160) → (B, z_channels, 8, 10)
        self.encoder = Encoder(**ddconfig)

        z_channels = ddconfig["z_channels"]

        # === Dual Projection Heads ===
        # Project shared features to geometry and dynamics branches
        self.proj_geo = nn.Sequential(
            nn.Conv2d(z_channels, z_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(z_channels, geo_embed_dim, 1),
        )

        self.proj_dyn = nn.Sequential(
            nn.Conv2d(z_channels, z_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(z_channels, dyn_embed_dim, 1),
        )

        # === Memory Module ===
        # Acts on dynamic features to enable temporal aggregation
        # Note: Encoder downsamples (len(ch_mult)-1) times, not len(ch_mult) times
        # See UniT/taming/modules/diffusionmodules/model.py line 500-502
        # Input resolution: 128x160 (GelSight standard) → After 3 downsamples: 16x20
        num_downsample = len(ddconfig["ch_mult"]) - 1
        spatial_size = (ddconfig["resolution"] // (2 ** num_downsample),
                       160 // (2 ** num_downsample))  # Width is 160 (GelSight standard)
        self.memory = MemoryModule(
            embed_dim=dyn_embed_dim,
            spatial_size=spatial_size,
            K=memory_K,
            num_heads=memory_num_heads,
            dropout=0.1,
        )

        # === Quantization Conv Layers ===
        # Channel adjustment before quantization (standard VQ-VAE design)
        self.quant_conv_geo = nn.Conv2d(geo_embed_dim, geo_embed_dim, 1)
        self.quant_conv_dyn = nn.Conv2d(dyn_embed_dim, dyn_embed_dim, 1)

        # === Dual Quantizers ===
        # Separate codebooks for geometry and dynamics
        self.quantize_geo = VectorQuantizer(n_embed, geo_embed_dim, beta=0.25)
        self.quantize_dyn = VectorQuantizer(n_embed, dyn_embed_dim, beta=0.25)

        # Post-quantization projection (for decoder input)
        self.post_quant_conv_geo = nn.Conv2d(geo_embed_dim, ddconfig["z_channels"], 1)
        self.post_quant_conv_dyn = nn.Conv2d(dyn_embed_dim, ddconfig["z_channels"], 1)

        # === Dual Decoders ===
        # RGB Decoder: Reconstructs original tactile image from Z_geo
        self.decoder_rgb = Decoder(**ddconfig)

        # MD-Map Decoder: Predicts 3-channel MD-Map from Z_dyn_mem
        # Note: output_ch=3 for MD-Map (dx, dy, energy)
        mdmap_ddconfig = ddconfig.copy()
        mdmap_ddconfig["out_ch"] = 3  # 3 channels for MD-Map
        self.decoder_mdmap = Decoder(**mdmap_ddconfig)

        # === Loss Function ===
        self.loss = instantiate_from_config(lossconfig)

        # === Training Configuration ===
        # Learning rate - will be set by workspace or config
        self.learning_rate = 4.5e-6

        # Load pretrained weights if provided
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        """Load checkpoint with optional key filtering"""
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        """
        Encode input to dual-stream latent representations

        Args:
            x (torch.Tensor): Input tactile image (B, 3, H, W)

        Returns:
            h (torch.Tensor): Shared encoder output
            z_geo (torch.Tensor): Geometry features (B, geo_embed_dim, h, w)
            z_dyn (torch.Tensor): Dynamics features (B, dyn_embed_dim, h, w)
        """
        # Shared encoding
        h = self.encoder(x)

        # Dual projection
        z_geo = self.proj_geo(h)
        z_dyn = self.proj_dyn(h)

        return h, z_geo, z_dyn

    def forward(self, x, md_map_gt=None, global_step=0):
        """
        Full forward pass: encode → memory → quantize → decode

        Args:
            x (torch.Tensor): Input tactile image (B, 3, H, W)
            md_map_gt (torch.Tensor, optional): MD-Map ground truth (B, 3, H, W)
            global_step (int): Current training step (for logging)

        Returns:
            dict with keys:
                - rgb_recon: RGB reconstruction (B, 3, H, W)
                - mdmap_pred: MD-Map prediction (B, 3, H, W)
                - quant_loss_geo: Geometry quantization loss
                - quant_loss_dyn: Dynamics quantization loss
        """
        # === Encoding ===
        h, z_geo, z_dyn = self.encode(x)

        # === Channel Adjustment (Standard VQ-VAE) ===
        z_geo = self.quant_conv_geo(z_geo)
        z_dyn = self.quant_conv_dyn(z_dyn)

        # === Memory Augmentation (on dynamics branch) ===
        z_dyn_mem = self.memory(z_dyn)

        # === Quantization ===
        z_geo_q, loss_geo, info_geo = self.quantize_geo(z_geo)
        z_dyn_q, loss_dyn, info_dyn = self.quantize_dyn(z_dyn_mem)

        # === Decoding ===
        # RGB reconstruction from geometry
        z_geo_dec = self.post_quant_conv_geo(z_geo_q)
        rgb_recon = self.decoder_rgb(z_geo_dec)

        # MD-Map prediction from dynamics
        z_dyn_dec = self.post_quant_conv_dyn(z_dyn_q)
        mdmap_pred = self.decoder_mdmap(z_dyn_dec)

        return {
            'rgb_recon': rgb_recon,
            'mdmap_pred': mdmap_pred,
            'quant_loss_geo': loss_geo,
            'quant_loss_dyn': loss_dyn,
            'info_geo': info_geo,
            'info_dyn': info_dyn,
        }

    def to_latent(self, x):
        """
        Extract continuous latent features (for policy learning)

        Returns both geometry and dynamics features for downstream tasks.
        This is the interface used by DiffusionUnitPolicy.

        Args:
            x (torch.Tensor): Input tactile image (B, 3, H, W)

        Returns:
            dict with keys:
                - z_geo: Geometry features (B, geo_embed_dim, h, w)
                - z_dyn: Dynamics features (B, dyn_embed_dim, h, w)
                - z_dyn_mem: Memory-augmented dynamics (B, dyn_embed_dim, h, w)
        """
        h, z_geo, z_dyn = self.encode(x)

        # === Channel Adjustment (Standard VQ-VAE) ===
        z_geo = self.quant_conv_geo(z_geo)
        z_dyn = self.quant_conv_dyn(z_dyn)

        z_dyn_mem = self.memory(z_dyn)

        return {
            'z_geo': z_geo,
            'z_dyn': z_dyn,
            'z_dyn_mem': z_dyn_mem,
        }

    def get_input(self, batch, k):
        """Extract and preprocess input from batch"""
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        Training step for dual-objective learning

        Optimizes:
        1. RGB reconstruction (geometry branch)
        2. MD-Map prediction (dynamics branch)
        3. Dual quantization losses
        """
        # === Episode Boundary Detection & Memory Reset ===
        episode_ids = batch.get('episode_id', None)
        if episode_ids is not None:
            # Initialize last episode tracking
            if not hasattr(self, '_last_episode_id'):
                self._last_episode_id = episode_ids[0].item() if torch.is_tensor(episode_ids[0]) else episode_ids[0]

            # Get current episode ID (all frames in batch should have same episode_id)
            current_episode_id = episode_ids[0].item() if torch.is_tensor(episode_ids[0]) else episode_ids[0]

            # If episode changed, reset memory queues
            if current_episode_id != self._last_episode_id:
                self.memory.reset_memory()
                if self.global_step % 100 == 0:  # Only log occasionally
                    print(f"[Step {self.global_step}] Memory reset: Episode {self._last_episode_id} → {current_episode_id}")
                self._last_episode_id = current_episode_id

        # Get inputs
        x = self.get_input(batch, self.image_key)
        md_map_gt = self.get_input(batch, self.mdmap_key) if self.mdmap_key in batch else None

        # Forward pass
        outputs = self(x, md_map_gt, global_step=self.global_step)

        # === Optimizer 0: Generator (Encoder + Decoders) ===
        if optimizer_idx == 0:
            # Quantization losses (needed as first parameter for loss function)
            quant_loss = outputs['quant_loss_geo'] + outputs['quant_loss_dyn']

            # RGB reconstruction loss (with correct VQ-VAE signature)
            # Set loss module to training mode explicitly
            self.loss.train()
            rgb_loss, rgb_log_dict = self.loss(
                quant_loss,              # First param: quantization loss
                x,                       # Second param: original input
                outputs['rgb_recon'],    # Third param: reconstruction
                optimizer_idx,
                self.global_step,
                last_layer=self.decoder_rgb.conv_out.weight,
                split="train"
            )

            # MD-Map prediction loss (if GT available)
            if md_map_gt is not None:
                # Resize prediction to match GT if needed
                mdmap_pred_resized = F.interpolate(
                    outputs['mdmap_pred'],
                    size=md_map_gt.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

                # CRITICAL FIX: Clip MD-Map GT to prevent outlier explosion
                # MD-Map GT has extreme outliers (max=6057 in episode_049, episode_044)
                # Normal range: [-100, 100], clip to prevent gradient explosion
                md_map_gt_clipped = torch.clamp(md_map_gt, min=-100.0, max=100.0)

                # Use Huber loss (robust to outliers) instead of MSE
                mdmap_loss = F.smooth_l1_loss(mdmap_pred_resized, md_map_gt_clipped, beta=1.0)
            else:
                mdmap_loss = torch.tensor(0.0, device=x.device)

            # Total loss
            loss = (self.lambda_rgb * rgb_loss +
                   self.lambda_mdmap * mdmap_loss +
                   quant_loss)

            # Logging
            self.log("train/rgb_loss", rgb_loss, prog_bar=True, logger=True)
            self.log("train/mdmap_loss", mdmap_loss, prog_bar=True, logger=True)
            self.log("train/quant_loss_geo", outputs['quant_loss_geo'], prog_bar=False, logger=True)
            self.log("train/quant_loss_dyn", outputs['quant_loss_dyn'], prog_bar=False, logger=True)
            self.log("train/total_loss", loss, prog_bar=True, logger=True)

            for k, v in rgb_log_dict.items():
                self.log(k, v, prog_bar=False, logger=True)

            return loss

        # === Optimizer 1: Discriminator ===
        if optimizer_idx == 1:
            # Compute quantization loss for discriminator
            quant_loss = outputs['quant_loss_geo'] + outputs['quant_loss_dyn']

            # Discriminator loss (with correct VQ-VAE signature)
            disc_loss, disc_log_dict = self.loss(
                quant_loss,              # First param: quantization loss
                x,                       # Second param: original input
                outputs['rgb_recon'],    # Third param: reconstruction
                optimizer_idx,
                self.global_step,
                last_layer=self.decoder_rgb.conv_out.weight,
                split="train"
            )

            self.log("train/disc_loss", disc_loss, prog_bar=True, logger=True)
            for k, v in disc_log_dict.items():
                self.log(k, v, prog_bar=False, logger=True)

            return disc_loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        # === Episode Boundary Detection & Memory Reset ===
        episode_ids = batch.get('episode_id', None)
        if episode_ids is not None:
            # Initialize tracking on first validation step
            if not hasattr(self, '_last_val_episode_id'):
                self._last_val_episode_id = episode_ids[0].item() if torch.is_tensor(episode_ids[0]) else episode_ids[0]

            # Extract current episode ID
            current_episode_id = episode_ids[0].item() if torch.is_tensor(episode_ids[0]) else episode_ids[0]

            # Reset memory if episode changed
            if current_episode_id != self._last_val_episode_id:
                self.memory.reset_memory()
                # Log validation memory resets less frequently
                if batch_idx % 50 == 0:
                    print(f"[Val Step {batch_idx}] Memory reset: Episode {self._last_val_episode_id} → {current_episode_id}")
                self._last_val_episode_id = current_episode_id

        x = self.get_input(batch, self.image_key)
        md_map_gt = self.get_input(batch, self.mdmap_key) if self.mdmap_key in batch else None

        outputs = self(x, md_map_gt)

        # Quantization losses (needed as first parameter for loss function)
        quant_loss = outputs['quant_loss_geo'] + outputs['quant_loss_dyn']

        # RGB reconstruction loss (with correct VQ-VAE signature)
        # Set loss module to eval mode to avoid assertion error
        self.loss.eval()
        rgb_loss, rgb_log_dict = self.loss(
            quant_loss,              # First param: quantization loss
            x,                       # Second param: original input
            outputs['rgb_recon'],    # Third param: reconstruction
            0,                       # optimizer_idx for generator
            self.global_step,
            last_layer=self.decoder_rgb.conv_out.weight,
            split="val"
        )

        # MD-Map loss
        if md_map_gt is not None:
            mdmap_pred_resized = F.interpolate(
                outputs['mdmap_pred'],
                size=md_map_gt.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            # Apply same clipping and robust loss as in training
            md_map_gt_clipped = torch.clamp(md_map_gt, min=-100.0, max=100.0)
            mdmap_loss = F.smooth_l1_loss(mdmap_pred_resized, md_map_gt_clipped, beta=1.0)
        else:
            mdmap_loss = torch.tensor(0.0, device=x.device)
        total_loss = self.lambda_rgb * rgb_loss + self.lambda_mdmap * mdmap_loss + quant_loss

        self.log("val/rgb_loss", rgb_loss, prog_bar=True, logger=True)
        self.log("val/mdmap_loss", mdmap_loss, prog_bar=True, logger=True)
        self.log("val/total_loss", total_loss, prog_bar=True, logger=True)

        return total_loss

    def configure_optimizers(self):
        """Configure optimizers for generator and discriminator"""
        lr = self.learning_rate

        # Optimizer for generator (encoder + decoders + quantizers + memory)
        opt_gen = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.proj_geo.parameters()) +
            list(self.proj_dyn.parameters()) +
            list(self.memory.parameters()) +
            list(self.quantize_geo.parameters()) +
            list(self.quantize_dyn.parameters()) +
            list(self.post_quant_conv_geo.parameters()) +
            list(self.post_quant_conv_dyn.parameters()) +
            list(self.decoder_rgb.parameters()) +
            list(self.decoder_mdmap.parameters()),
            lr=lr, betas=(0.5, 0.9)
        )

        # Optimizer for discriminator
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(),
            lr=lr, betas=(0.5, 0.9)
        )

        return [opt_gen, opt_disc], []

    def reset_memory(self):
        """Reset memory queue (call at episode boundaries)"""
        self.memory.reset_memory()

    def _mdmap_to_rgb(self, mdmap: torch.Tensor) -> torch.Tensor:
        """
        Convert 3-channel MD-Map to RGB visualization for wandb logging

        Args:
            mdmap: (N, 3, H, W) tensor containing:
                - Ch 0-1: Displacement vectors (dx, dy)
                - Ch 2: Cumulative energy

        Returns:
            rgb: (N, 3, H, W) tensor in [0, 1] range for visualization
        """
        N, C, H, W = mdmap.shape
        assert C == 3, f"Expected 3 channels, got {C}"

        # Extract channels (N, H, W)
        dx = mdmap[:, 0, :, :]
        dy = mdmap[:, 1, :, :]
        energy = mdmap[:, 2, :, :]

        # Convert displacement vectors to HSV color wheel
        # Hue: direction (arctan2), Value: magnitude
        magnitude = torch.sqrt(dx**2 + dy**2)  # (N, H, W)
        angle = torch.atan2(dy, dx)  # (N, H, W), range [-π, π]

        # Normalize to [0, 1]
        hue = (angle + torch.pi) / (2 * torch.pi)  # [0, 1]
        saturation = torch.ones_like(magnitude)  # Full saturation
        value = magnitude / (magnitude.max() + 1e-6)  # Normalize by max

        # HSV to RGB conversion (vectorized)
        # Based on standard HSV->RGB algorithm
        h = hue * 6.0  # [0, 6]
        c = value * saturation
        x = c * (1 - torch.abs(h % 2 - 1))
        m = value - c

        # Initialize RGB
        rgb = torch.zeros(N, 3, H, W, device=mdmap.device)

        # Segment-wise assignment (avoiding explicit loops)
        mask0 = (h >= 0) & (h < 1)
        mask1 = (h >= 1) & (h < 2)
        mask2 = (h >= 2) & (h < 3)
        mask3 = (h >= 3) & (h < 4)
        mask4 = (h >= 4) & (h < 5)
        mask5 = (h >= 5) & (h < 6)

        rgb[:, 0, :, :] = (c * mask0 + x * mask1 + m * mask2 + m * mask3 + x * mask4 + c * mask5 + m)
        rgb[:, 1, :, :] = (x * mask0 + c * mask1 + c * mask2 + x * mask3 + m * mask4 + m * mask5 + m)
        rgb[:, 2, :, :] = (m * mask0 + m * mask1 + x * mask2 + c * mask3 + c * mask4 + x * mask5 + m)

        # Overlay energy as brightness modulation on channel 2 (blue)
        energy_norm = energy / (energy.abs().max() + 1e-6)  # [-1, 1]
        energy_norm = torch.clamp((energy_norm + 1) / 2, 0, 1)  # [0, 1]
        rgb[:, 2, :, :] = rgb[:, 2, :, :] * 0.5 + energy_norm * 0.5

        return torch.clamp(rgb, 0, 1)

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        """
        Generate images for wandb logging during training

        Args:
            batch: Dict with 'image' and 'md_map' keys

        Returns:
            log: Dict of tensors for wandb (all in [0, 1] range, (N, 3, H, W) format)
        """
        log = {}

        # Get inputs
        x = self.get_input(batch, self.image_key)  # (N, 3, H, W)
        md_map_gt = self.get_input(batch, self.mdmap_key)  # (N, 3, H, W)

        x = x.to(self.device)
        md_map_gt = md_map_gt.to(self.device)

        # Limit to 4 images for logging
        N = min(x.shape[0], 4)
        x = x[:N]
        md_map_gt = md_map_gt[:N]

        # Run forward pass
        outputs = self(x, md_map_gt)

        # Convert from [-1, 1] to [0, 1] for wandb
        log['tactile_input'] = (x + 1.0) / 2.0
        log['rgb_reconstruction'] = torch.clamp((outputs['rgb_recon'] + 1.0) / 2.0, 0, 1)

        # Convert MD-Maps to RGB visualization
        log['mdmap_gt_vis'] = self._mdmap_to_rgb(md_map_gt)
        log['mdmap_pred_vis'] = self._mdmap_to_rgb(outputs['mdmap_pred'])

        # Also log raw MD-Map channels separately for detailed inspection
        # Ch 2 (cumulative energy) is the most interesting
        energy_gt = md_map_gt[:, 2:3, :, :]  # (N, 1, H, W)
        energy_pred = outputs['mdmap_pred'][:, 2:3, :, :]

        # Normalize energy to [0, 1] for visualization
        energy_gt_norm = (energy_gt - energy_gt.min()) / (energy_gt.max() - energy_gt.min() + 1e-6)
        energy_pred_norm = (energy_pred - energy_pred.min()) / (energy_pred.max() - energy_pred.min() + 1e-6)

        # Expand to 3 channels for grayscale visualization
        log['energy_gt'] = energy_gt_norm.repeat(1, 3, 1, 1)
        log['energy_pred'] = energy_pred_norm.repeat(1, 3, 1, 1)

        return log


# Example usage
if __name__ == "__main__":
    print("Testing MemoryVQVAE...")

    # Minimal config for testing
    ddconfig = {
        "double_z": False,
        "z_channels": 3,
        "resolution": 128,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 1, 2, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [16],
        "dropout": 0.0,
    }

    lossconfig = {
        "target": "UniT.taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
        "params": {
            "disc_start": 10000,
            "disc_weight": 0.8,
            "codebook_weight": 1.0,
        }
    }

    # Create model
    model = MemoryVQVAE(
        ddconfig=ddconfig,
        lossconfig=lossconfig,
        n_embed=512,
        geo_embed_dim=3,
        dyn_embed_dim=3,
        memory_K=5,
        memory_num_heads=1,
    )

    print(f"\nModel created successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Test forward pass
    B = 2
    x = torch.randn(B, 3, 128, 160)
    md_map_gt = torch.randn(B, 3, 240, 320)

    print(f"\nInput shape: {x.shape}")
    print(f"MD-Map GT shape: {md_map_gt.shape}")

    outputs = model(x, md_map_gt)

    print(f"\nOutputs:")
    print(f"  RGB reconstruction: {outputs['rgb_recon'].shape}")
    print(f"  MD-Map prediction: {outputs['mdmap_pred'].shape}")
    print(f"  Geo quant loss: {outputs['quant_loss_geo'].item():.4f}")
    print(f"  Dyn quant loss: {outputs['quant_loss_dyn'].item():.4f}")

    print("\n✓ MemoryVQVAE test passed!")
