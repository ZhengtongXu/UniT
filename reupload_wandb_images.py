#!/usr/bin/env python3
"""
重新上传wandb图像 - 从本地文件恢复到wandb run

适用场景: 训练过程中网络中断导致图像没上传，但metrics数值已上传
"""
import os
import sys
import wandb
from PIL import Image
import glob
from tqdm import tqdm
import re

# 配置信息
RUN_ID = "fw4l4z5l"  # v4的run ID
PROJECT = "residual_tactile"
ENTITY = "pweiii_github"  # 你的wandb用户名

# 本地图像路径
LOCAL_IMAGE_DIR = "representation_models/2025-12-25T15-32-46residual_vqvae_v4_twostage/wandb/latest-run/files/media/images"

def parse_step_from_filename(filename):
    """从文件名中解析step number"""
    # 文件名格式: inputs_tactile_37944_5e75d6af8c220e4c3376.png
    match = re.search(r'_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def upload_images_to_run(run_id, project, entity, local_dir):
    """
    重新上传图像到指定的wandb run

    Args:
        run_id: wandb run ID
        project: wandb project名称
        entity: wandb用户名/团队名
        local_dir: 本地图像目录
    """
    # 恢复已有的run
    print(f"连接到wandb run: {entity}/{project}/{run_id}")
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    print(f"Run信息:")
    print(f"  Name: {run.name}")
    print(f"  State: {run.state}")
    print(f"  URL: {run.url}")
    print()

    # Resume这个run
    print("Resume run以上传图像...")
    wandb.init(
        project=project,
        entity=entity,
        id=run_id,
        resume="must"  # 必须恢复这个run
    )

    # 定义metric允许乱序写入
    print("配置wandb允许乱序上传图像...")
    wandb.define_metric("*", step_metric="wandb_step", summary="none")

    # 遍历本地图像目录
    for split in ['train', 'val']:
        split_dir = os.path.join(local_dir, split)
        if not os.path.exists(split_dir):
            print(f"⚠️ 目录不存在: {split_dir}")
            continue

        print(f"\n处理 {split} 图像...")

        # 获取所有图像文件
        image_files = sorted(glob.glob(os.path.join(split_dir, "*.png")))
        print(f"找到 {len(image_files)} 个图像文件")

        # 按image_type分组
        image_groups = {}
        for img_path in image_files:
            filename = os.path.basename(img_path)

            # 提取image type (inputs_tactile, recon_hybrid等)
            # 文件名格式: {image_type}_{step}_{hash}.png
            parts = filename.split('_')
            if len(parts) >= 3:
                # image_type可能包含多个下划线，比如inputs_tactile
                image_type = '_'.join(parts[:-2])
                step = parse_step_from_filename(filename)

                if step is not None:
                    key = f"{split}/{image_type}"
                    if key not in image_groups:
                        image_groups[key] = []
                    image_groups[key].append((step, img_path))

        # 按image type上传
        for key, images in image_groups.items():
            # 按step排序
            images.sort(key=lambda x: x[0])

            print(f"\n  上传 {key}: {len(images)} 张图像")

            # 批量上传 (每次上传一定数量避免内存问题)
            batch_size = 50
            for i in tqdm(range(0, len(images), batch_size), desc=f"  {key}"):
                batch = images[i:i+batch_size]

                for step, img_path in batch:
                    try:
                        # 读取图像
                        img = Image.open(img_path)

                        # 上传到wandb，使用wandb_step来允许乱序写入
                        wandb.log({
                            key: wandb.Image(img),
                            "wandb_step": step  # 作为自定义step
                        })

                    except Exception as e:
                        print(f"    ⚠️ 上传失败 {img_path}: {e}")

    # 完成
    wandb.finish()
    print(f"\n✓ 图像重新上传完成!")
    print(f"查看结果: {run.url}")


if __name__ == "__main__":
    print("=" * 60)
    print("重新上传wandb图像")
    print("=" * 60)
    print()

    # 检查本地目录是否存在
    if not os.path.exists(LOCAL_IMAGE_DIR):
        print(f"❌ 错误: 本地图像目录不存在: {LOCAL_IMAGE_DIR}")
        sys.exit(1)

    print(f"本地图像目录: {LOCAL_IMAGE_DIR}")
    print()

    # 确认
    response = input("确认要重新上传图像到wandb吗? (yes/no): ")
    if response.lower() != 'yes':
        print("取消上传")
        sys.exit(0)

    # 执行上传
    try:
        upload_images_to_run(RUN_ID, PROJECT, ENTITY, LOCAL_IMAGE_DIR)
    except Exception as e:
        print(f"\n❌ 上传失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
