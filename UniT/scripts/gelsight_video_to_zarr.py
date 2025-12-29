"""
Convert GelSight MP4 videos to Zarr ReplayBuffer format

This script extracts frames from GelSight tactile sensor videos and converts them
to the zarr format expected by MDMapDataset.

Input: tac_data/test/peg_in_hole1/*/v1/gelsight/gelsight_left_rgb.mp4
Output: replay_buffer.zarr with structure:
    data/
    ├── tactile_image/     # (N, H, W, 3) uint8
    └── meta/
        └── episode_ends   # [ep1_end, ep2_end, ...] cumulative frame counts

Author: Claude Code
Date: 2025-12-18
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from diffusion_policy.common.replay_buffer import ReplayBuffer
    USE_REPLAY_BUFFER = True
except ImportError:
    print("Warning: diffusion_policy not found, using direct zarr implementation")
    import zarr
    USE_REPLAY_BUFFER = False


def extract_frames_from_video(video_path):
    """
    Extract all frames from an MP4 video file

    Args:
        video_path: Path to MP4 file

    Returns:
        frames: numpy array of shape (T, H, W, 3) uint8
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames extracted from {video_path}")

    return np.array(frames, dtype=np.uint8)


def convert_videos_to_zarr(
    input_dir,
    output_zarr,
    max_episodes=None,
    target_shape=None,
):
    """
    Convert GelSight videos to ReplayBuffer zarr format

    Args:
        input_dir: Root directory containing episode folders
        output_zarr: Output path for replay_buffer.zarr
        max_episodes: Maximum number of episodes to convert (None = all)
        target_shape: Target shape (H, W) for resizing (None = keep original)
    """
    input_path = Path(input_dir)

    # Find all GelSight video files
    video_pattern = "**/gelsight/gelsight_left_rgb.mp4"
    video_files = sorted(input_path.rglob(video_pattern))

    if len(video_files) == 0:
        raise FileNotFoundError(f"No GelSight videos found in {input_dir} with pattern {video_pattern}")

    print(f"Found {len(video_files)} GelSight video files")

    # Limit number of episodes if requested
    if max_episodes is not None:
        video_files = video_files[:max_episodes]
        print(f"Processing first {max_episodes} episodes")

    # Create output directory
    output_path = Path(output_zarr)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Creating zarr at: {output_zarr}")

    # Collect all episodes first
    all_frames = []
    episode_ends = []

    # Process each video
    total_frames = 0
    for i, video_path in enumerate(tqdm(video_files, desc="Converting episodes")):
        try:
            # Extract frames from video
            frames = extract_frames_from_video(video_path)

            # Resize if target shape is specified
            if target_shape is not None:
                H_target, W_target = target_shape
                H_orig, W_orig = frames.shape[1:3]

                if (H_orig, W_orig) != (H_target, W_target):
                    resized_frames = []
                    for frame in frames:
                        # Note: cv2.resize expects (W, H)
                        resized = cv2.resize(frame, (W_target, H_target), interpolation=cv2.INTER_LINEAR)
                        resized_frames.append(resized)
                    frames = np.array(resized_frames, dtype=np.uint8)

            # Collect frames
            all_frames.append(frames)
            total_frames += len(frames)
            episode_ends.append(total_frames)

            # Extract episode name from path (timestamp folder)
            episode_name = video_path.parent.parent.parent.name
            print(f"  Episode {i} ({episode_name}): {len(frames)} frames, shape {frames.shape}")

        except Exception as e:
            print(f"  ❌ Failed to process {video_path}: {e}")
            continue

    # Concatenate all frames
    print(f"\nConcatenating {len(all_frames)} episodes...")
    all_frames_concat = np.concatenate(all_frames, axis=0)
    episode_ends = np.array(episode_ends, dtype=np.int64)

    print(f"Total shape: {all_frames_concat.shape}")

    # Save to zarr
    if USE_REPLAY_BUFFER:
        buffer = ReplayBuffer.create_from_path(zarr_path=str(output_zarr), mode='a')
        for frames in all_frames:
            episode_data = {'tactile_image': frames}
            buffer.add_episode(episode_data, compressors='disk')
        print(f"\nUsing ReplayBuffer API:")
        print(f"  - data/tactile_image: {buffer.root['data']['tactile_image'].shape}")
        print(f"  - meta/episode_ends: {buffer.root['meta']['episode_ends'][:].tolist()}")
    else:
        # Direct zarr implementation
        import zarr
        root = zarr.open(str(output_zarr), mode='w')

        # Create data group
        data_group = root.create_group('data')
        data_group.create_dataset(
            'tactile_image',
            data=all_frames_concat,
            chunks=(1, all_frames_concat.shape[1], all_frames_concat.shape[2], all_frames_concat.shape[3]),
            dtype=np.uint8,
        )

        # Create meta group
        meta_group = root.create_group('meta')
        meta_group.create_dataset('episode_ends', data=episode_ends, dtype=np.int64)

        print(f"\nDirect zarr implementation:")
        print(f"  - data/tactile_image: {root['data']['tactile_image'].shape}")
        print(f"  - meta/episode_ends: {root['meta']['episode_ends'][:].tolist()}")

    print(f"\n✅ Conversion complete!")
    print(f"  Total episodes: {len(all_frames)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Output: {output_zarr}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert GelSight MP4 videos to Zarr ReplayBuffer format"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Root directory containing episode folders (e.g., tac_data/test/peg_in_hole1)'
    )
    parser.add_argument(
        '--output-zarr',
        type=str,
        required=True,
        help='Output path for replay_buffer.zarr'
    )
    parser.add_argument(
        '--max-episodes',
        type=int,
        default=None,
        help='Maximum number of episodes to convert (default: all)'
    )
    parser.add_argument(
        '--resize',
        nargs=2,
        type=int,
        default=None,
        metavar=('H', 'W'),
        help='Resize frames to (H, W). Example: --resize 240 320'
    )

    args = parser.parse_args()

    # Convert resize argument to tuple
    target_shape = tuple(args.resize) if args.resize is not None else None

    # Run conversion
    convert_videos_to_zarr(
        input_dir=args.input_dir,
        output_zarr=args.output_zarr,
        max_episodes=args.max_episodes,
        target_shape=target_shape,
    )


if __name__ == '__main__':
    main()
