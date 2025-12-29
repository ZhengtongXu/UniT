"""
Video Visualization Utilities for Residual VQ-VAE

Provides helper functions for composing multi-panel visualization videos.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def tensor_to_cv2(tensor: torch.Tensor, denormalize_imagenet: bool = False) -> np.ndarray:
    """
    Convert PyTorch tensor to OpenCV BGR format.

    Args:
        tensor: (C, H, W) or (B, C, H, W) tensor in [0, 1] or ImageNet-normalized
        denormalize_imagenet: If True, reverse ImageNet normalization

    Returns:
        img: (H, W, 3) uint8 BGR image for OpenCV
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first batch

    # Move to CPU and convert to numpy
    img = tensor.cpu().numpy()  # (C, H, W)

    # Denormalize ImageNet if needed
    if denormalize_imagenet:
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img = img * std + mean

    # Clip to [0, 1]
    img = np.clip(img, 0, 1)

    # Convert to uint8 [0, 255]
    img = (img * 255).astype(np.uint8)

    # Convert from (C, H, W) to (H, W, C)
    img = img.transpose(1, 2, 0)

    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def upsample_latent(latent: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    """
    Upsample latent visualization to target size with bilinear interpolation.

    Args:
        latent: (B, C, H, W) latent tensor
        target_size: (height, width) target size

    Returns:
        upsampled: (B, C, target_h, target_w)
    """
    return F.interpolate(
        latent,
        size=target_size,
        mode='bilinear',
        align_corners=False
    )


def apply_gaussian_blur(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply Gaussian blur for smoother visualization.

    Args:
        img: (H, W, C) uint8 image
        kernel_size: Gaussian kernel size (must be odd)

    Returns:
        blurred: (H, W, C) uint8 image
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def add_text_with_background(
    img: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.7,
    thickness: int = 2,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    padding: int = 5
) -> np.ndarray:
    """
    Add text with background rectangle for better readability.

    Args:
        img: (H, W, 3) uint8 image
        text: Text string to render
        position: (x, y) top-left position
        font_scale: Font size
        thickness: Text thickness
        text_color: RGB color for text
        bg_color: RGB color for background
        padding: Padding around text

    Returns:
        img: Modified image
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Draw background rectangle
    x, y = position
    cv2.rectangle(
        img,
        (x - padding, y - padding),
        (x + text_w + padding, y + text_h + baseline + padding),
        bg_color,
        -1  # Filled
    )

    # Draw text
    cv2.putText(
        img,
        text,
        (x, y + text_h),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )

    return img


def create_bar_chart(
    raw_norms: List[float],
    vq_norms: List[float],
    current_idx: int,
    window_size: int = 50,
    chart_size: Tuple[int, int] = (1920, 180),
    title: str = "Residual Norm Over Time"
) -> np.ndarray:
    """
    Create a bar chart showing residual norms with rolling window.

    Args:
        raw_norms: List of raw residual norms
        vq_norms: List of quantized residual norms
        current_idx: Current frame index
        window_size: Number of frames to show
        chart_size: (width, height) of output chart
        title: Chart title

    Returns:
        chart_img: (H, W, 3) uint8 BGR image
    """
    # Create figure
    dpi = 100
    fig_width = chart_size[0] / dpi
    fig_height = chart_size[1] / dpi
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    # Determine window range
    start_idx = max(0, current_idx - window_size + 1)
    end_idx = current_idx + 1

    # Extract window data
    window_raw = raw_norms[start_idx:end_idx]
    window_vq = vq_norms[start_idx:end_idx]
    x_indices = list(range(start_idx, end_idx))

    # Plot bars
    bar_width = 0.4
    x_pos = np.arange(len(x_indices))

    ax.bar(x_pos - bar_width/2, window_raw, bar_width,
           label='Raw Residual', color='#4285f4', alpha=0.8)
    ax.bar(x_pos + bar_width/2, window_vq, bar_width,
           label='VQ Residual', color='#ea4335', alpha=0.8)

    # Add mean reference lines
    if window_raw:
        mean_raw = np.mean(window_raw)
        mean_vq = np.mean(window_vq)
        ax.axhline(mean_raw, color='#4285f4', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(mean_vq, color='#ea4335', linestyle='--', linewidth=1, alpha=0.5)

    # Styling
    ax.set_xlabel('Frame Index', fontsize=12, color='white')
    ax.set_ylabel('L2 Norm', fontsize=12, color='white')
    ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=10)
    ax.legend(loc='upper right', fontsize=10, facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
    ax.grid(True, alpha=0.3, color='#cccccc')

    # Set background
    fig.patch.set_facecolor('#2b2b2b')
    ax.set_facecolor('#2b2b2b')
    ax.tick_params(colors='white', labelsize=9)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')

    # Set x-axis ticks (show every 10 frames)
    if len(x_indices) > 10:
        tick_step = max(1, len(x_indices) // 10)
        tick_positions = x_pos[::tick_step]
        tick_labels = [x_indices[i] for i in range(0, len(x_indices), tick_step)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
    else:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_indices)

    # Convert to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    chart_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    chart_img = chart_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Convert RGB to BGR for OpenCV
    chart_img = cv2.cvtColor(chart_img, cv2.COLOR_RGB2BGR)

    plt.close(fig)

    return chart_img


def pad_to_size(img: np.ndarray, target_h: int, target_w: int, center: bool = True) -> np.ndarray:
    """
    Pad image to target size with black borders, maintaining aspect ratio.

    Args:
        img: (H, W, 3) input image
        target_h: Target height
        target_w: Target width
        center: If True, center the image; otherwise top-left align

    Returns:
        padded: (target_h, target_w, 3) padded image
    """
    h, w = img.shape[:2]

    # Create black canvas
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Calculate padding
    if center:
        y_offset = (target_h - h) // 2
        x_offset = (target_w - w) // 2
    else:
        y_offset = 0
        x_offset = 0

    # Ensure within bounds
    y_offset = max(0, y_offset)
    x_offset = max(0, x_offset)
    y_end = min(target_h, y_offset + h)
    x_end = min(target_w, x_offset + w)

    # Place image on canvas
    canvas[y_offset:y_end, x_offset:x_end] = img[:y_end-y_offset, :x_end-x_offset]

    return canvas


def compose_visualization_frame(
    wrist_img: np.ndarray,
    tactile_img: np.ndarray,
    vq_residual_img: np.ndarray,
    z_real_fixed_img: np.ndarray,
    chart_img: np.ndarray,
    hybrid_recon_img: np.ndarray,
    zexp_only_img: np.ndarray,
    frame_idx: int,
    total_frames: int,
    residual_norm: float,
    hybrid_mse: float,
    zexp_mse: float,
    border_thickness: int = 2,
    border_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Compose all panels into final 1920x1080 frame with horizontal layout.

    NEW Layout (Left-to-Right, no vertical stacking):
    ┌─────────────┬─────────────┬─────────────┐
    │             │             │             │
    │   Wrist     │  Original   │ VQ Residual │
    │  Camera     │  Tactile    │ (auto_norm) │
    │  224×224    │  160×128    │  160×128    │
    │             │             │             │
    ├─────────────┼─────────────┼─────────────┤
    │             │             │             │
    │ z_real_     │   Hybrid    │  Baseline   │
    │ fixed       │   Recon     │   (z_exp)   │
    │ 160×128     │  160×128    │  160×128    │
    │             │             │             │
    ├─────────────┴─────────────┴─────────────┤
    │      Residual Norm Chart (Bottom)       │
    └─────────────────────────────────────────┘

    Args:
        All images in original resolution (BGR uint8):
        wrist_img: (224, 224, 3)
        tactile_img: (128, 160, 3) - note: H×W
        vq_residual_img: (128, 160, 3)
        z_real_fixed_img: (128, 160, 3)
        chart_img: (H, W, 3) - will be resized to fit
        hybrid_recon_img: (128, 160, 3)
        zexp_only_img: (128, 160, 3)

    Returns:
        frame: (1080, 1920, 3) uint8 BGR
    """
    # Canvas size
    CANVAS_H = 1080
    CANVAS_W = 1920

    # Create blank canvas
    frame = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    # Layout parameters
    # 3 columns, 2 rows for images + 1 row for chart
    col_w = CANVAS_W // 3  # 640 per column
    row1_h = 400  # Top row height
    row2_h = 400  # Middle row height
    chart_h = CANVAS_H - row1_h - row2_h  # Bottom row (chart)

    # Row 1: Wrist | Original Tactile | VQ Residual
    # Column 1: Wrist Camera (224×224)
    wrist_padded = pad_to_size(wrist_img, row1_h, col_w, center=True)
    frame[0:row1_h, 0:col_w] = wrist_padded
    add_text_with_background(frame, "Wrist Camera View", (10, 10), font_scale=0.7, thickness=2)

    # Column 2: Original Tactile (128×160)
    tactile_padded = pad_to_size(tactile_img, row1_h, col_w, center=True)
    frame[0:row1_h, col_w:2*col_w] = tactile_padded
    add_text_with_background(frame, "Original Tactile", (col_w+10, 10), font_scale=0.7, thickness=2)

    # Column 3: VQ Residual (128×160)
    vq_residual_padded = pad_to_size(vq_residual_img, row1_h, col_w, center=True)
    frame[0:row1_h, 2*col_w:3*col_w] = vq_residual_padded
    add_text_with_background(frame, "VQ Residual (auto_norm)", (2*col_w+10, 10), font_scale=0.7, thickness=2)

    # Draw vertical borders for row 1
    cv2.line(frame, (col_w, 0), (col_w, row1_h), border_color, border_thickness)
    cv2.line(frame, (2*col_w, 0), (2*col_w, row1_h), border_color, border_thickness)

    # Draw horizontal border after row 1
    cv2.line(frame, (0, row1_h), (CANVAS_W, row1_h), border_color, border_thickness)

    # Row 2: z_real_fixed | Hybrid Recon | Baseline Recon
    row2_y = row1_h

    # Column 1: z_real_fixed (128×160)
    z_real_padded = pad_to_size(z_real_fixed_img, row2_h, col_w, center=True)
    frame[row2_y:row2_y+row2_h, 0:col_w] = z_real_padded
    add_text_with_background(frame, "z_real (fixed_norm)", (10, row2_y+10), font_scale=0.7, thickness=2)

    # Column 2: Hybrid Reconstruction (128×160)
    hybrid_padded = pad_to_size(hybrid_recon_img, row2_h, col_w, center=True)
    frame[row2_y:row2_y+row2_h, col_w:2*col_w] = hybrid_padded
    hybrid_title = f"Hybrid (z_exp+VQ) MSE:{hybrid_mse:.5f}"
    add_text_with_background(frame, hybrid_title, (col_w+10, row2_y+10), font_scale=0.6, thickness=2)

    # Column 3: Baseline Reconstruction (128×160)
    zexp_padded = pad_to_size(zexp_only_img, row2_h, col_w, center=True)
    frame[row2_y:row2_y+row2_h, 2*col_w:3*col_w] = zexp_padded
    zexp_title = f"Baseline (z_exp) MSE:{zexp_mse:.5f}"
    add_text_with_background(frame, zexp_title, (2*col_w+10, row2_y+10), font_scale=0.6, thickness=2)

    # Draw vertical borders for row 2
    cv2.line(frame, (col_w, row2_y), (col_w, row2_y+row2_h), border_color, border_thickness)
    cv2.line(frame, (2*col_w, row2_y), (2*col_w, row2_y+row2_h), border_color, border_thickness)

    # Draw horizontal border after row 2
    chart_y = row2_y + row2_h
    cv2.line(frame, (0, chart_y), (CANVAS_W, chart_y), border_color, border_thickness)

    # Row 3: Chart (full width)
    chart_resized = cv2.resize(chart_img, (CANVAS_W, chart_h))
    frame[chart_y:chart_y+chart_h, :] = chart_resized

    # Add global info overlay
    info_text = f"Frame: {frame_idx+1:03d}/{total_frames:03d} | Residual Norm: {residual_norm:.4f}"
    add_text_with_background(frame, info_text, (CANVAS_W-550, CANVAS_H-40),
                            font_scale=0.7, thickness=2)

    # Add improvement percentage
    if zexp_mse > 0:
        improvement = ((zexp_mse - hybrid_mse) / zexp_mse) * 100
        improvement_text = f"Improvement: {improvement:.2f}%"
        color = (0, 255, 0) if improvement > 0 else (0, 0, 255)
        add_text_with_background(frame, improvement_text, (CANVAS_W-300, CANVAS_H-70),
                                font_scale=0.7, thickness=2, text_color=color)

    return frame


def write_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 15,
    codec: str = 'mp4v'
) -> None:
    """
    Write frames to video file.

    Args:
        frames: List of (H, W, 3) uint8 BGR images
        output_path: Output video file path
        fps: Frames per second
        codec: Video codec ('mp4v', 'avc1', 'H264')
    """
    if not frames:
        raise ValueError("No frames to write")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        writer.write(frame)

    writer.release()
    print(f"Video saved to: {output_path}")
    print(f"  Resolution: {width}×{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {len(frames)}")
    print(f"  Duration: {len(frames)/fps:.2f} seconds")
