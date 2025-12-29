#!/usr/bin/env python3
"""
批量生成MD-Map GT (Ground Truth)

遍历data/目录下所有GelSight视频，调用md_map_visualizer.py生成MD-Map，
保存为.npz格式供训练使用。

用法:
    python generate_mdmap_gt.py \\
        --data-root /mlp_vepfs/share/zpw/tac/UniT-Align/data \\
        --output-root /mlp_vepfs/share/zpw/tac/UniT-Align/data_mdmap_gt \\
        --K 5 \\
        --norm-coeff 0.8

作者: Claude Code
日期: 2025-12-17
"""

import argparse
import subprocess
import os
from pathlib import Path
import sys

# 尝试导入tqdm，如果没有则使用简单的替代
try:
    from tqdm import tqdm
except ImportError:
    # 简单的tqdm替代类
    class tqdm:
        def __init__(self, iterable, desc=None, unit=None):
            self.iterable = iterable
            if desc:
                print(f"{desc}...")

        def __iter__(self):
            return iter(self.iterable)

        @staticmethod
        def write(msg):
            print(msg)


def find_gelsight_videos(data_root):
    """
    查找data_root下所有GelSight视频

    Args:
        data_root: 数据根目录

    Returns:
        video_files: 找到的视频文件路径列表 (Path objects)
    """
    data_root = Path(data_root)

    # 查找所有gelsight_left_rgb.mp4文件
    video_files = list(data_root.rglob("**/gelsight_left_rgb.mp4"))

    print(f"在 {data_root} 中找到 {len(video_files)} 个GelSight视频")

    return video_files


def get_episode_id(video_path):
    """
    从视频路径提取episode ID

    假设路径格式: .../task_name/episode_id/v1/gelsight/gelsight_left_rgb.mp4
    例如: tac_data/test/peg_in_hole1/164327/v1/gelsight/gelsight_left_rgb.mp4
          → episode_id = "164327"

    Args:
        video_path: 视频文件路径

    Returns:
        episode_id: episode的唯一标识符
    """
    # 向上4级: gelsight → v1 → episode_id → task_name
    parts = video_path.parts
    if len(parts) >= 4:
        episode_id = parts[-4]  # v1的上一级
        return episode_id
    else:
        # 如果路径不标准，使用完整相对路径作为ID
        return str(video_path).replace('/', '_').replace('.mp4', '')


def generate_mdmap_for_video(video_path, output_dir, K=5, norm_coeff=0.8,
                             visualizer_path=None, skip_existing=True):
    """
    为单个视频生成MD-Map GT

    Args:
        video_path: GelSight视频路径
        output_dir: 输出目录
        K: 历史窗口大小
        norm_coeff: 归一化系数
        visualizer_path: md_map_visualizer.py的路径（如果不在PATH中）
        skip_existing: 是否跳过已存在的GT文件

    Returns:
        success: 是否成功生成
        npz_path: 生成的npz文件路径（如果成功）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 输出文件路径
    output_video = output_dir / "temp_mdmap.mp4"
    npz_path = output_dir / "mdmap_gt.npz"

    # 如果已存在且skip_existing=True，跳过
    if skip_existing and npz_path.exists():
        print(f"  跳过（已存在）: {npz_path}")
        return True, npz_path

    # 默认使用相对路径（假设在UniT-Align目录下运行）
    if visualizer_path is None:
        # 默认路径：/mlp_vepfs/share/zpw/tac/script/md_map_visualizer.py
        visualizer_path = "/mlp_vepfs/share/zpw/tac/script/md_map_visualizer.py"

    if not Path(visualizer_path).exists():
        print(f"  错误: 找不到md_map_visualizer.py: {visualizer_path}")
        return False, None

    # 构造命令
    cmd = [
        "python", visualizer_path,
        "--input", str(video_path),
        "--output", str(output_video),
        "--window-size", str(K),
        "--norm-coeff", str(norm_coeff),
        "--save-all-frames",  # 保存所有帧为npz
        "--no-concat",  # 不生成拼接视频（节省时间）
    ]

    try:
        # 执行命令（捕获输出以避免打印太多信息）
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # 检查npz文件是否生成
        # md_map_visualizer.py会保存为 {output_video}_mdmap_gt.npz
        generated_npz = Path(str(output_video).replace('.mp4', '_mdmap_gt.npz'))

        if not generated_npz.exists():
            print(f"  错误: npz文件未生成: {generated_npz}")
            return False, None

        # 移动npz文件到目标位置
        if generated_npz != npz_path:
            generated_npz.rename(npz_path)

        # 删除临时视频
        if output_video.exists():
            output_video.unlink()

        print(f"  成功: {npz_path}")
        return True, npz_path

    except subprocess.CalledProcessError as e:
        print(f"  失败: {video_path}")
        print(f"  错误信息: {e.stderr[:200]}")  # 只打印前200字符
        return False, None


def main():
    parser = argparse.ArgumentParser(
        description='批量生成MD-Map GT数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基本用法
    python generate_mdmap_gt.py \\
        --data-root /mlp_vepfs/share/zpw/tac/UniT-Align/data \\
        --output-root /mlp_vepfs/share/zpw/tac/UniT-Align/data_mdmap_gt

    # 自定义参数
    python generate_mdmap_gt.py \\
        --data-root /mlp_vepfs/share/zpw/tac/UniT-Align/data \\
        --output-root /mlp_vepfs/share/zpw/tac/UniT-Align/data_mdmap_gt \\
        --K 7 \\
        --norm-coeff 0.9 \\
        --overwrite

输出结构:
    output-root/
    ├── task1/
    │   ├── episode_001/
    │   │   └── mdmap_gt.npz  # (T, 240, 320, 3)
    │   ├── episode_002/
    │   │   └── mdmap_gt.npz
    │   └── ...
    └── task2/
        └── ...
        """
    )

    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='数据根目录（包含GelSight视频的目录）'
    )

    parser.add_argument(
        '--output-root',
        type=str,
        required=True,
        help='输出根目录（保存MD-Map GT的目录）'
    )

    parser.add_argument(
        '--K', '--window-size',
        type=int,
        default=5,
        help='历史窗口大小（默认: 5）'
    )

    parser.add_argument(
        '--norm-coeff',
        type=float,
        default=0.8,
        help='归一化系数（默认: 0.8）'
    )

    parser.add_argument(
        '--visualizer-path',
        type=str,
        default=None,
        help='md_map_visualizer.py的路径（默认: /mlp_vepfs/share/zpw/tac/script/md_map_visualizer.py）'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='覆盖已存在的GT文件（默认: 跳过已存在的）'
    )

    parser.add_argument(
        '--max-videos',
        type=int,
        default=None,
        help='最多处理的视频数量（用于测试，默认: 全部）'
    )

    args = parser.parse_args()

    # 验证data-root存在
    if not os.path.exists(args.data_root):
        print(f"错误: 数据根目录不存在: {args.data_root}", file=sys.stderr)
        return 1

    # 创建output-root
    os.makedirs(args.output_root, exist_ok=True)

    print("=" * 70)
    print(" " * 20 + "批量生成MD-Map GT")
    print("=" * 70)
    print(f"\n数据根目录: {args.data_root}")
    print(f"输出根目录: {args.output_root}")
    print(f"历史窗口K: {args.K}")
    print(f"归一化系数: {args.norm_coeff}")
    print(f"覆盖已存在: {'是' if args.overwrite else '否'}")
    print("=" * 70)
    print()

    # 查找所有GelSight视频
    video_files = find_gelsight_videos(args.data_root)

    if not video_files:
        print("未找到任何GelSight视频文件，退出")
        return 1

    # 限制处理数量（用于测试）
    if args.max_videos is not None:
        video_files = video_files[:args.max_videos]
        print(f"（测试模式：仅处理前 {args.max_videos} 个视频）\n")

    # 统计信息
    success_count = 0
    fail_count = 0

    # 逐个处理
    for video_path in tqdm(video_files, desc="处理视频", unit="video"):
        # 提取episode信息
        episode_id = get_episode_id(video_path)

        # 构造输出目录：保持相对路径结构
        # 例如: data/task1/ep001/v1/gelsight/video.mp4
        #    → output/task1/ep001/
        rel_path = video_path.relative_to(args.data_root)
        # 向上2级（去掉 v1/gelsight）
        episode_rel_dir = Path(*rel_path.parts[:-3])
        output_dir = Path(args.output_root) / episode_rel_dir

        # 生成MD-Map GT
        tqdm.write(f"处理: {episode_id}")
        success, npz_path = generate_mdmap_for_video(
            video_path,
            output_dir,
            K=args.K,
            norm_coeff=args.norm_coeff,
            visualizer_path=args.visualizer_path,
            skip_existing=not args.overwrite
        )

        # 注意：如果文件已存在且skip_existing=True，generate_mdmap_for_video返回success=True
        # 但实际是跳过了，需要检查消息
        if success:
            # 检查是否是新生成的（通过检查消息或文件时间戳）
            # 简单方法：如果返回True但文件已存在很久，说明是跳过的
            # 这里简化处理：success_count统计所有成功的（包括跳过）
            success_count += 1
        else:
            fail_count += 1

    # 打印统计
    print("\n" + "=" * 70)
    print("处理完成")
    print("=" * 70)
    print(f"成功: {success_count} 个")
    print(f"失败: {fail_count} 个")
    print(f"总计: {len(video_files)} 个")
    print("=" * 70)

    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
