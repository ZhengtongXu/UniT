#!/bin/bash
# ==========================================
# 重新生成数据集 v2 (使用handeye摄像头)
# ==========================================
#
# 说明:
#   - 使用handeye_realsense_rgb替代main_realsense_rgb
#   - 生成到新目录 data/residual_tactile_train_v2
#   - 保留旧数据 data/residual_tactile_train 供对比
#
# 使用:
#   bash regenerate_data_v2.sh
# ==========================================

set -e

cd /root/code/zpw/tac/UniT-Align

echo "=========================================="
echo "数据集重新生成 (v2 - handeye摄像头)"
echo "=========================================="
echo ""

# 检查环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "❌ 错误: 请先激活conda环境"
    echo "运行: conda activate unitenv"
    exit 1
fi

echo "环境: $CONDA_DEFAULT_ENV"
echo ""

# 检查输入数据
INPUT_DIR="/mlp_vepfs/share/zpw/tac/script/tac_data/test/peg_in_hole1"
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ 错误: 输入数据目录不存在: $INPUT_DIR"
    exit 1
fi

# 统计episode数量
EPISODE_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -type d -name "[0-9]*" | wc -l)
echo "输入数据: $INPUT_DIR"
echo "Episodes: $EPISODE_COUNT"
echo ""

# 检查旧数据
OLD_DATA="data/residual_tactile_train"
if [ -d "$OLD_DATA" ]; then
    echo "✓ 旧数据存在: $OLD_DATA (将保留)"
else
    echo "⚠️  警告: 旧数据不存在: $OLD_DATA"
fi
echo ""

# 运行数据转换
echo "=========================================="
echo "开始数据转换..."
echo "=========================================="
echo ""

python UniT/dataset/convert_residual_data.py \
    --input_dir "$INPUT_DIR" \
    --output_zarr data/residual_tactile_train_v2 \
    --visual_size 224 224 \
    --tactile_size 128 160

echo ""
echo "=========================================="
echo "数据转换完成"
echo "=========================================="
echo ""

# 验证数据
echo "验证新数据集..."
python -c "
import zarr
import numpy as np

z = zarr.open('data/residual_tactile_train_v2', 'r')

print('\n数据集结构:')
print(f'  tactile_image: {z[\"data/tactile_image\"].shape} {z[\"data/tactile_image\"].dtype}')
print(f'  visual_image:  {z[\"data/visual_image\"].shape} {z[\"data/visual_image\"].dtype}')
print(f'  robot_state:   {z[\"data/robot_state\"].shape} {z[\"data/robot_state\"].dtype}')
print(f'  episode_id:    {z[\"meta/episode_id\"].shape} {z[\"meta/episode_id\"].dtype}')
print(f'  timestamp:     {z[\"meta/timestamp\"].shape} {z[\"meta/timestamp\"].dtype}')

print('\n数据统计:')
print(f'  总帧数: {len(z[\"data/tactile_image\"])}')
print(f'  独立episodes: {len(np.unique(z[\"meta/episode_id\"]))}')

print('\n视觉图像样本 (前3帧):')
for i in range(min(3, len(z['data/visual_image']))):
    img = z['data/visual_image'][i]
    print(f'  帧{i}: min={img.min()}, max={img.max()}, mean={img.mean():.2f}')
"

echo ""
echo "=========================================="
echo "✓ 数据集生成完成"
echo "=========================================="
echo ""
echo "新数据路径: data/residual_tactile_train_v2"
echo "旧数据路径: data/residual_tactile_train (已保留)"
echo ""
echo "下一步:"
echo "  1. 比对v1和v2的visual_image差异"
echo "  2. 运行验证脚本确认数据质量"
echo "  3. 使用v2数据启动训练"
echo ""
