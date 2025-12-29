"""
数据转换脚本: 多模态对齐 + Zarr格式转换
========================================

功能:
1. 从 tac_data/test/peg_in_hole1/{timestamp}/v1/ 加载:
   - GelSight触觉视频 (gelsight_left_rgb.mp4)
   - RealSense视觉视频 (main_realsense_rgb.mp4)
   - Franka机器人状态 (Franka_4_arms_arm.jsonl)

2. 时间戳对齐:
   - 以触觉帧时间戳为锚点
   - 最近邻匹配视觉帧和机器人状态

3. 输出扩展的Zarr格式:
   - data/tactile_image: (N, 128, 160, 3) uint8
   - data/visual_image: (N, 224, 224, 3) uint8
   - data/robot_state: (N, 8) float32
   - meta/episode_id: (N,) int32
   - meta/timestamp: (N,) float64

使用:
    python UniT/dataset/convert_residual_data.py \
        --input_dir /path/to/tac_data/test/peg_in_hole1 \
        --output_zarr data/residual_tactile_zarr \
        --visual_size 224 224 \
        --tactile_size 128 160
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import cv2
import zarr
from tqdm import tqdm


def load_timestamps_from_txt(txt_path: str) -> np.ndarray:
    """
    从.txt文件加载时间戳 (每行一个浮点数)

    Args:
        txt_path: 时间戳文件路径
    Returns:
        timestamps: (N,) 数组
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"时间戳文件不存在: {txt_path}")

    timestamps = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                timestamps.append(float(line))

    return np.array(timestamps)


def load_timestamps_from_jsonl(jsonl_path: str) -> np.ndarray:
    """
    从.jsonl文件加载时间戳 (每行一个JSON对象,包含timestamp字段)

    Args:
        jsonl_path: JSONL时间戳文件路径
    Returns:
        timestamps: (N,) 数组 (单位: 秒)
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"时间戳文件不存在: {jsonl_path}")

    timestamps = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                ts = data['timestamp']
                # RealSense timestamps are in milliseconds, convert to seconds
                if ts > 1e12:  # If timestamp > 1 trillion, it's likely in milliseconds
                    ts = ts / 1000.0
                timestamps.append(ts)

    return np.array(timestamps)


def load_robot_states_from_jsonl(jsonl_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从JSONL加载机器人状态

    Args:
        jsonl_path: Franka_4_arms_arm.jsonl 路径
    Returns:
        timestamps: (N,) 数组
        states: (N, 8) 数组 [joint_positions(7D), gripper_width(1D)]
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"机器人状态文件不存在: {jsonl_path}")

    timestamps = []
    states = []

    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                timestamps.append(data['timestamp'])

                # 提取8D状态向量
                joints = np.array(data['joint_positions'])  # 7D
                # gripper_width在JSONL中已是list,取第一个元素
                gripper_val = data['gripper_width'][0] if isinstance(data['gripper_width'], list) else data['gripper_width']
                gripper = np.array([gripper_val])  # 1D
                state_8d = np.concatenate([joints, gripper])
                states.append(state_8d)

    return np.array(timestamps), np.array(states)


def decode_video_frames(video_path: str, target_size: Tuple[int, int] = None) -> np.ndarray:
    """
    解码视频并调整尺寸

    Args:
        video_path: 视频文件路径
        target_size: (width, height) 目标尺寸
    Returns:
        frames: (N, H, W, 3) uint8数组
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=frame_count, desc=f"解码 {Path(video_path).name}", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # OpenCV读取为BGR,转为RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 调整尺寸
            if target_size is not None:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

            frames.append(frame)
            pbar.update(1)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"视频无帧: {video_path}")

    return np.array(frames, dtype=np.uint8)


def find_nearest_indices(query_timestamps: np.ndarray,
                         reference_timestamps: np.ndarray) -> np.ndarray:
    """
    为每个查询时间戳找到最近的参考时间戳索引

    Args:
        query_timestamps: (N,) 查询时间戳
        reference_timestamps: (M,) 参考时间戳
    Returns:
        indices: (N,) 最近邻索引
    """
    indices = np.searchsorted(reference_timestamps, query_timestamps)

    # 处理边界情况
    indices = np.clip(indices, 0, len(reference_timestamps) - 1)

    # 对于非边界情况,检查左右两侧哪个更近
    for i in range(len(indices)):
        idx = indices[i]
        if idx > 0:
            left_dist = abs(query_timestamps[i] - reference_timestamps[idx - 1])
            right_dist = abs(query_timestamps[i] - reference_timestamps[idx])
            if left_dist < right_dist:
                indices[i] = idx - 1

    return indices


def process_episode(episode_dir: str,
                   tactile_size: Tuple[int, int],
                   visual_size: Tuple[int, int],
                   episode_id: int) -> Dict:
    """
    处理单个episode的多模态数据对齐

    Args:
        episode_dir: episode目录 (如 .../peg_in_hole1/155859/v1)
        tactile_size: (width, height) 触觉图像尺寸
        visual_size: (width, height) 视觉图像尺寸
        episode_id: episode编号
    Returns:
        aligned_data: dict包含对齐后的数据
    """
    print(f"\n处理 Episode {episode_id}: {episode_dir}")

    # 1. 加载触觉时间戳 (作为锚点)
    tactile_ts_path = os.path.join(episode_dir, 'origin_data', 'gelsight_left_timestamps.txt')
    tactile_ts = load_timestamps_from_txt(tactile_ts_path)
    print(f"  触觉时间戳: {len(tactile_ts)} 帧")

    # 2. 解码触觉视频
    tactile_video_path = os.path.join(episode_dir, 'gelsight', 'gelsight_left_rgb.mp4')
    tactile_frames = decode_video_frames(tactile_video_path, target_size=tactile_size)
    print(f"  触觉视频: {tactile_frames.shape}")

    # 验证时间戳和帧数匹配
    if len(tactile_ts) != len(tactile_frames):
        print(f"  警告: 触觉时间戳({len(tactile_ts)})与帧数({len(tactile_frames)})不匹配")
        # 取最小长度
        min_len = min(len(tactile_ts), len(tactile_frames))
        tactile_ts = tactile_ts[:min_len]
        tactile_frames = tactile_frames[:min_len]

    # 3. 加载视觉时间戳和视频 (使用handeye摄像头)
    visual_ts_path = os.path.join(episode_dir, 'origin_data', 'handeye_realsense_timestamps.jsonl')
    visual_ts = load_timestamps_from_jsonl(visual_ts_path)
    print(f"  视觉时间戳: {len(visual_ts)} 帧")

    visual_video_path = os.path.join(episode_dir, 'videos', 'handeye_realsense_rgb.mp4')
    visual_frames = decode_video_frames(visual_video_path, target_size=visual_size)
    print(f"  视觉视频: {visual_frames.shape}")

    # 验证视觉时间戳和帧数
    if len(visual_ts) != len(visual_frames):
        print(f"  警告: 视觉时间戳({len(visual_ts)})与帧数({len(visual_frames)})不匹配")
        min_len = min(len(visual_ts), len(visual_frames))
        visual_ts = visual_ts[:min_len]
        visual_frames = visual_frames[:min_len]

    # 4. 加载机器人状态
    robot_jsonl_path = os.path.join(episode_dir, 'data', 'Franka_4_arms_arm.jsonl')
    robot_ts, robot_states = load_robot_states_from_jsonl(robot_jsonl_path)
    print(f"  机器人状态: {robot_states.shape}")

    # 5. 时间戳对齐 (以触觉为锚点)
    print("  对齐时间戳...")
    visual_indices = find_nearest_indices(tactile_ts, visual_ts)
    robot_indices = find_nearest_indices(tactile_ts, robot_ts)

    # 6. 构建对齐后的数据
    aligned_visual = visual_frames[visual_indices]
    aligned_robot = robot_states[robot_indices]

    # 计算对齐误差统计
    visual_time_errors = np.abs(tactile_ts - visual_ts[visual_indices])
    robot_time_errors = np.abs(tactile_ts - robot_ts[robot_indices])

    print(f"  视觉时间对齐误差: mean={visual_time_errors.mean():.4f}s, max={visual_time_errors.max():.4f}s")
    print(f"  机器人时间对齐误差: mean={robot_time_errors.mean():.4f}s, max={robot_time_errors.max():.4f}s")

    return {
        'tactile_image': tactile_frames,
        'visual_image': aligned_visual,
        'robot_state': aligned_robot,
        'episode_id': np.full(len(tactile_frames), episode_id, dtype=np.int32),
        'timestamp': tactile_ts,
        'alignment_errors': {
            'visual_mean': visual_time_errors.mean(),
            'visual_max': visual_time_errors.max(),
            'robot_mean': robot_time_errors.mean(),
            'robot_max': robot_time_errors.max(),
        }
    }


def create_zarr_structure(output_path: str, tactile_size: Tuple[int, int],
                         visual_size: Tuple[int, int]):
    """
    创建Zarr数据结构

    Args:
        output_path: 输出Zarr路径
        tactile_size: (width, height)
        visual_size: (width, height)
    """
    root = zarr.open(output_path, mode='w')

    # 数据字段
    root.create_dataset('data/tactile_image',
                       shape=(0, tactile_size[1], tactile_size[0], 3),
                       chunks=(1, tactile_size[1], tactile_size[0], 3),
                       dtype=np.uint8)

    root.create_dataset('data/visual_image',
                       shape=(0, visual_size[1], visual_size[0], 3),
                       chunks=(1, visual_size[1], visual_size[0], 3),
                       dtype=np.uint8)

    root.create_dataset('data/robot_state',
                       shape=(0, 8),
                       chunks=(128, 8),
                       dtype=np.float32)

    # 元数据字段
    root.create_dataset('meta/episode_id',
                       shape=(0,),
                       chunks=(128,),
                       dtype=np.int32)

    root.create_dataset('meta/timestamp',
                       shape=(0,),
                       chunks=(128,),
                       dtype=np.float64)

    print(f"✓ 已创建Zarr结构: {output_path}")
    return root


def append_to_zarr(root: zarr.Group, data: Dict):
    """
    将数据追加到Zarr

    Args:
        root: Zarr根节点
        data: 对齐后的数据字典
    """
    n = len(data['tactile_image'])

    # 追加数据
    root['data/tactile_image'].append(data['tactile_image'], axis=0)
    root['data/visual_image'].append(data['visual_image'], axis=0)
    root['data/robot_state'].append(data['robot_state'], axis=0)
    root['meta/episode_id'].append(data['episode_id'], axis=0)
    root['meta/timestamp'].append(data['timestamp'], axis=0)

    print(f"  追加 {n} 帧到Zarr")


def main():
    parser = argparse.ArgumentParser(description="多模态触觉数据转换为Zarr格式")
    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入目录 (如 tac_data/test/peg_in_hole1)')
    parser.add_argument('--output_zarr', type=str, required=True,
                       help='输出Zarr路径 (如 data/residual_tactile_zarr)')
    parser.add_argument('--tactile_size', type=int, nargs=2, default=[128, 160],
                       help='触觉图像尺寸 [width height]')
    parser.add_argument('--visual_size', type=int, nargs=2, default=[224, 224],
                       help='视觉图像尺寸 [width height]')
    parser.add_argument('--episodes', type=str, default=None,
                       help='指定episode列表 (如 "155859,160306") 或留空处理全部')

    args = parser.parse_args()

    # 验证输入目录
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"输入目录不存在: {args.input_dir}")

    # 获取所有episode目录
    if args.episodes:
        episode_names = args.episodes.split(',')
    else:
        # 自动发现所有episode
        episode_names = [d for d in os.listdir(args.input_dir)
                        if os.path.isdir(os.path.join(args.input_dir, d, 'v1'))]
        episode_names.sort()

    print(f"找到 {len(episode_names)} 个episodes: {episode_names}")

    # 创建Zarr结构
    tactile_size = tuple(args.tactile_size)
    visual_size = tuple(args.visual_size)
    root = create_zarr_structure(args.output_zarr, tactile_size, visual_size)

    # 处理每个episode
    total_frames = 0
    alignment_stats = []

    for episode_id, episode_name in enumerate(episode_names):
        episode_dir = os.path.join(args.input_dir, episode_name, 'v1')

        try:
            aligned_data = process_episode(
                episode_dir=episode_dir,
                tactile_size=tactile_size,
                visual_size=visual_size,
                episode_id=episode_id
            )

            append_to_zarr(root, aligned_data)

            total_frames += len(aligned_data['tactile_image'])
            alignment_stats.append(aligned_data['alignment_errors'])

        except Exception as e:
            print(f"  错误: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 保存元数据
    metadata = {
        'num_episodes': len(episode_names),
        'total_frames': total_frames,
        'tactile_size': tactile_size,
        'visual_size': visual_size,
    }

    # 只有成功处理的episode才有alignment_stats
    if alignment_stats:
        metadata['alignment_stats'] = {
            'visual_mean_error': np.mean([s['visual_mean'] for s in alignment_stats]),
            'visual_max_error': np.max([s['visual_max'] for s in alignment_stats]),
            'robot_mean_error': np.mean([s['robot_mean'] for s in alignment_stats]),
            'robot_max_error': np.max([s['robot_max'] for s in alignment_stats]),
        }
    else:
        print("\n警告: 没有成功处理的episode,无对齐统计信息")
        metadata['alignment_stats'] = None

    root.attrs.update(metadata)

    print(f"\n{'='*60}")
    print(f"✓ 转换完成!")
    print(f"  总Episodes: {metadata['num_episodes']}")
    print(f"  总帧数: {metadata['total_frames']}")
    print(f"  输出路径: {args.output_zarr}")
    if metadata['alignment_stats']:
        print(f"  平均对齐误差: 视觉={metadata['alignment_stats']['visual_mean_error']:.4f}s, "
              f"机器人={metadata['alignment_stats']['robot_mean_error']:.4f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
