"""
Residual VQ-VAE Visualization Video Generator
==============================================

Creates comprehensive multi-panel visualization videos demonstrating
the semantic value of VQ residual signals in the v4 two-stage training.

Usage:
    python visualize_residual_vqvae.py --checkpoint_path <path> --episode_idx <idx>

Example:
    python visualize_residual_vqvae.py \
        --checkpoint_path representation_models/2025-12-26T11-51-18residual_vqvae_v4_twostage/checkpoints/checkpoint-epoch149.ckpt \
        --episode_idx 0 \
        --max_frames 150
"""

import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple
from tqdm import tqdm
import zarr

from UniT.taming.models.residual_vqmodel import ResidualVQModel
from UniT.dataset.residual_tactile_dataset import ResidualTactileDataset
from UniT.utils.video_utils import (
    tensor_to_cv2,
    upsample_latent,
    apply_gaussian_blur,
    create_bar_chart,
    compose_visualization_frame,
    write_video
)


def load_model_from_checkpoint(checkpoint_path: str, device: str = 'cuda') -> ResidualVQModel:
    """
    Load ResidualVQModel from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        model: Loaded model in eval mode
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract hyperparameters from checkpoint if available
    # Otherwise use default values
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        print(f"Using hyperparameters from checkpoint")
    else:
        # Default hyperparameters (should match training config)
        hparams = {
            'ddconfig': {
                'double_z': False,
                'z_channels': 3,
                'resolution': 128,
                'in_channels': 3,
                'out_ch': 3,
                'ch': 128,
                'ch_mult': [1, 1, 2, 4],
                'num_res_blocks': 2,
                'attn_resolutions': [16],
                'dropout': 0.0
            },
            'lossconfig': {
                'target': 'UniT.taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator',
                'params': {
                    'disc_start': 0,
                    'disc_weight': 0.8,
                    'disc_in_channels': 3,
                }
            },
            'prophet_config': {
                'visual_backbone': 'resnet18',
                'visual_pretrained': True,
                'state_dim': 8,
                'use_state': False,
                'latent_shape': [3, 20, 16],
                'hidden_dims': [256, 128, 64]
            },
            'residual_vq_config': {
                'n_e': 512,
                'e_dim': 3,
                'beta': 0.25,
                'legacy': False
            },
            'lambda_sparse': 0.01,
            'z_exp_dropout': 0.0
        }
        print("Using default hyperparameters (checkpoint doesn't contain hparams)")

    # Filter out keys that are not in __init__ parameters
    # aux_loss_weight and hybrid_loss_weight are set as instance variables
    init_params = {k: v for k, v in hparams.items()
                  if k not in ['aux_loss_weight', 'hybrid_loss_weight']}

    # Create model
    model = ResidualVQModel(**init_params)

    # Load state dict
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # Move to device and set to eval mode
    model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f"  Device: {device}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

    return model


def load_dataset(dataset_path: str, use_train: bool = False, val_ratio: float = 0.1, seed: int = 42) -> ResidualTactileDataset:
    """
    Load dataset (training or validation).

    Args:
        dataset_path: Path to zarr dataset
        use_train: If True, use training set; if False, use validation set
        val_ratio: Validation split ratio
        seed: Random seed

    Returns:
        dataset: ResidualTactileDataset instance
    """
    print(f"\nLoading dataset from: {dataset_path}")
    print(f"  Split: {'Training' if use_train else 'Validation'}")

    dataset = ResidualTactileDataset(
        dataset_path=dataset_path,
        visual_size=(224, 224),
        tactile_size=(128, 160),
        normalize_robot_state=True,
        train=use_train,  # Use train or val based on parameter
        val_ratio=val_ratio,
        seed=seed
    )

    print(f"✓ Dataset loaded")
    print(f"  Total samples: {len(dataset)}")

    return dataset


def get_episode_indices(dataset: ResidualTactileDataset) -> Dict[int, List[int]]:
    """
    Group dataset indices by episode ID.

    Args:
        dataset: ResidualTactileDataset instance

    Returns:
        episode_map: Dict mapping episode_id -> list of LOCAL dataset indices (0 to len(dataset)-1)
    """
    # Load episode IDs from zarr
    zarr_path = dataset.dataset_path
    zarr_root = zarr.open(zarr_path, mode='r')

    if 'episode_id' in zarr_root['meta']:
        episode_ids = zarr_root['meta']['episode_id'][:]
    else:
        # If no episode_id, treat all as single episode
        print("Warning: No episode_id found, treating all samples as single episode")
        return {0: list(range(len(dataset)))}

    # Group by episode using LOCAL indices
    # dataset.indices contains the raw zarr indices
    # but dataset[i] expects i in range(0, len(dataset))
    episode_map = {}
    for local_idx, zarr_idx in enumerate(dataset.indices):
        ep_id = int(episode_ids[zarr_idx])
        if ep_id not in episode_map:
            episode_map[ep_id] = []
        episode_map[ep_id].append(local_idx)  # Use LOCAL index, not zarr index

    # Sort indices within each episode
    for ep_id in episode_map:
        episode_map[ep_id].sort()

    print(f"\n✓ Found {len(episode_map)} episodes in {'validation' if not dataset.train else 'training'} set")
    for ep_id, indices in list(episode_map.items())[:5]:
        print(f"  Episode {ep_id}: {len(indices)} frames")
    if len(episode_map) > 5:
        print(f"  ... and {len(episode_map)-5} more episodes")

    return episode_map


def visualize_episode(
    model: ResidualVQModel,
    dataset: ResidualTactileDataset,
    episode_indices: List[int],
    episode_id: int,
    output_path: str,
    max_frames: int = 200,
    fps: int = 15,
    device: str = 'cuda'
) -> None:
    """
    Generate visualization video for a single episode.

    Args:
        model: Trained ResidualVQModel
        dataset: ResidualTactileDataset instance
        episode_indices: List of sample indices for this episode
        episode_id: Episode ID for naming
        output_path: Output video file path
        max_frames: Maximum number of frames to visualize
        fps: Video frame rate
        device: Device for inference
    """
    print(f"\nGenerating visualization for Episode {episode_id}")
    print(f"  Total frames: {len(episode_indices)}")
    print(f"  Max frames: {max_frames}")

    # Limit frames
    episode_indices = episode_indices[:max_frames]
    total_frames = len(episode_indices)

    # Storage for metrics
    raw_residual_norms = []
    vq_residual_norms = []
    frames = []

    # Helper functions (following existing log_images logic)
    def fixed_norm(z: torch.Tensor) -> torch.Tensor:
        """Fixed range [-3, 3] normalization"""
        return torch.clamp((z + 3.0) / 6.0, 0., 1.)

    def auto_norm(z: torch.Tensor) -> torch.Tensor:
        """Per-sample min-max normalization"""
        B = z.shape[0]
        z_flat = z.view(B, -1)
        z_min = z_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        z_max = z_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        denom = z_max - z_min
        denom[denom < 1e-6] = 1.0
        return (z - z_min) / denom

    # Process each frame
    for frame_idx, sample_idx in enumerate(tqdm(episode_indices, desc="Processing frames")):
        # Get sample from dataset
        sample = dataset[sample_idx]

        # Convert to batch and move to device
        batch = {
            'tactile_image': sample['tactile_image'].unsqueeze(0).to(device),
            'visual_image': sample['visual_image'].unsqueeze(0).to(device),
            'robot_state': sample['robot_state'].unsqueeze(0).to(device)
        }

        # Model forward pass
        with torch.no_grad():
            outputs = model(batch)

            # Get baseline reconstruction (z_exp only)
            recon_z_exp_only = model.decode_z_exp_only(outputs['z_exp'])

        # Compute metrics
        raw_res_norm = outputs['raw_residual'].abs().mean().item()
        vq_res_norm = outputs['residual'].abs().mean().item()
        raw_residual_norms.append(raw_res_norm)
        vq_residual_norms.append(vq_res_norm)

        # Compute reconstruction MSE
        tactile_gt = batch['tactile_image']
        hybrid_mse = F.mse_loss(outputs['recon'], tactile_gt).item()
        zexp_mse = F.mse_loss(recon_z_exp_only, tactile_gt).item()

        # === Panel 1: Wrist Camera ===
        wrist_img = tensor_to_cv2(batch['visual_image'], denormalize_imagenet=True)

        # === Panel 2: Original Tactile ===
        tactile_img = tensor_to_cv2(batch['tactile_image'], denormalize_imagenet=False)

        # === Use model.log_images() to get correct visualization ===
        log_dict = model.log_images(batch)

        # === Panel 3: VQ Residual (auto_norm) ===
        # Upsample with nearest-neighbor to preserve sharp boundaries
        # Maintain native aspect ratio: (H=20, W=16) → aspect = W/H = 0.8
        # Scale up by 8x: 16×8=128, 20×8=160 → (H=160, W=128)
        vq_res_vis = log_dict['residual_vq_rgb']  # (1, 3, 20, 16) already auto_normed
        # Use nearest-neighbor upsampling (mode='nearest') to keep sharp, no blur
        vq_res_upsampled = F.interpolate(vq_res_vis, scale_factor=8, mode='nearest')
        vq_residual_img = tensor_to_cv2(vq_res_upsampled, denormalize_imagenet=False)
        # Result: (H=160, W=128) with aspect ratio 0.8 preserved, sharp boundaries

        # === Panel 4: z_real_fixed ===
        # Same upsampling strategy: nearest-neighbor 8x to preserve aspect ratio
        z_real_vis = log_dict['z_real_fixed']  # (1, 3, 20, 16) already fixed_normed
        z_real_upsampled = F.interpolate(z_real_vis, scale_factor=8, mode='nearest')
        z_real_img = tensor_to_cv2(z_real_upsampled, denormalize_imagenet=False)
        # Result: (H=160, W=128) with aspect ratio 0.8 preserved, sharp boundaries

        # === Panel 5: Residual Norm Bar Chart ===
        chart_img = create_bar_chart(
            raw_norms=raw_residual_norms,
            vq_norms=vq_residual_norms,
            current_idx=frame_idx,
            window_size=50,
            chart_size=(1920, 180),
            title="Residual Norm Over Time"
        )

        # === Panel 6: Hybrid Reconstruction ===
        hybrid_recon_img = tensor_to_cv2(outputs['recon'], denormalize_imagenet=False)

        # === Panel 7: z_exp Only Reconstruction ===
        zexp_only_img = tensor_to_cv2(recon_z_exp_only, denormalize_imagenet=False)

        # === Compose Final Frame ===
        composed_frame = compose_visualization_frame(
            wrist_img=wrist_img,
            tactile_img=tactile_img,
            vq_residual_img=vq_residual_img,
            z_real_fixed_img=z_real_img,
            chart_img=chart_img,
            hybrid_recon_img=hybrid_recon_img,
            zexp_only_img=zexp_only_img,
            frame_idx=frame_idx,
            total_frames=total_frames,
            residual_norm=vq_res_norm,
            hybrid_mse=hybrid_mse,
            zexp_mse=zexp_mse
        )

        frames.append(composed_frame)

    # Write video
    print(f"\nWriting video to: {output_path}")
    write_video(frames, output_path, fps=fps, codec='mp4v')

    # Print summary statistics
    print(f"\n{'='*60}")
    print("Visualization Summary:")
    print(f"  Episode ID: {episode_id}")
    print(f"  Total frames: {total_frames}")
    print(f"  Average raw residual norm: {np.mean(raw_residual_norms):.4f}")
    print(f"  Average VQ residual norm: {np.mean(vq_residual_norms):.4f}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate Residual VQ-VAE Visualization Video")

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='representation_models/2025-12-26T11-51-18residual_vqvae_v4_twostage/checkpoints/checkpoint-epoch149.ckpt',
        help='Path to trained checkpoint'
    )

    parser.add_argument(
        '--dataset_path',
        type=str,
        default='data/residual_tactile_train_v2',
        help='Path to dataset'
    )

    parser.add_argument(
        '--use_train',
        action='store_true',
        help='Use training set instead of validation set (training set has longer episodes)'
    )

    parser.add_argument(
        '--episode_idx',
        type=int,
        default=0,
        help='Episode index to visualize (0-based, sorted by episode length descending)'
    )

    parser.add_argument(
        '--max_frames',
        type=int,
        default=200,
        help='Maximum number of frames to visualize'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=15,
        help='Video frame rate'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='visualization_outputs',
        help='Output directory for videos'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for inference'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = load_model_from_checkpoint(args.checkpoint_path, device=args.device)

    # Load dataset
    dataset = load_dataset(args.dataset_path, use_train=args.use_train)

    # Get episode mapping
    episode_map = get_episode_indices(dataset)

    # Sort episodes by length (descending) so episode_idx=0 is the longest
    episode_lengths = [(ep_id, len(indices)) for ep_id, indices in episode_map.items()]
    episode_lengths.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop 10 longest episodes:")
    for i, (ep_id, length) in enumerate(episode_lengths[:10]):
        duration = length / args.fps
        marker = " <- Selected" if i == args.episode_idx else ""
        print(f"  [{i}] Episode {ep_id}: {length} frames ({duration:.1f}s @ {args.fps}fps){marker}")

    # Get requested episode
    if args.episode_idx >= len(episode_lengths):
        print(f"\nError: Episode index {args.episode_idx} out of range (0-{len(episode_lengths)-1})")
        return

    episode_id, _ = episode_lengths[args.episode_idx]
    episode_indices = episode_map[episode_id]

    # Output path
    checkpoint_name = pathlib.Path(args.checkpoint_path).stem
    output_path = os.path.join(
        args.output_dir,
        f"residual_vqvae_ep{episode_id}_{checkpoint_name}.mp4"
    )

    # Generate visualization
    visualize_episode(
        model=model,
        dataset=dataset,
        episode_indices=episode_indices,
        episode_id=episode_id,
        output_path=output_path,
        max_frames=args.max_frames,
        fps=args.fps,
        device=args.device
    )

    print("✓ Visualization complete!")


if __name__ == "__main__":
    main()
