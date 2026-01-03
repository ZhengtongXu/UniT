#!/bin/bash
# ===================================================================
# Spatiotemporal Residual VQ-VAE 训练启动脚本
# ===================================================================
# 功能: 启动"Feeling the Unexpected"时空触觉表征学习训练
#
# 核心理念:
# - Prophet预测预期的平滑物理趋势
# - Residual捕捉意外事件 (碰撞、打滑等)
# - VQ将连续的意外信号压缩成离散的Event Code
# - Dynamic FiLM Decoder使用Event Code调制解码
#
# 使用方法:
#   bash launch_spatiotemporal_training.sh
# ===================================================================

set -e  # Exit on error

echo "================================================================"
echo "Spatiotemporal Residual VQ-VAE 训练启动"
echo "================================================================"
echo ""

# ===================================================================
# 1. 环境检查
# ===================================================================
echo "[1/5] 检查环境..."

# 检查conda环境
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "❌ 错误: Conda环境未激活"
    echo "   请运行: conda activate unitenv"
    exit 1
fi

if [[ "${CONDA_DEFAULT_ENV}" != "unitenv" ]]; then
    echo "⚠️  警告: 当前环境为 ${CONDA_DEFAULT_ENV}, 建议使用 unitenv"
fi

echo "✓ Conda环境: ${CONDA_DEFAULT_ENV}"

# 检查GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ 错误: nvidia-smi未找到，无法检查GPU"
    exit 1
fi

GPU_ID=${GPU_ID:-0}
echo "✓ 使用GPU: ${GPU_ID}"
nvidia-smi --id=${GPU_ID} --query-gpu=name,memory.total,memory.free --format=csv,noheader

# ===================================================================
# 2. 数据集检查
# ===================================================================
echo ""
echo "[2/5] 检查数据集..."

DATASET_PATH="data/residual_tactile_train_v2"

if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "❌ 错误: 数据集未找到: ${DATASET_PATH}"
    echo "   请确保数据集已生成"
    exit 1
fi

if [[ ! -f "${DATASET_PATH}/.zarray" ]] && [[ ! -d "${DATASET_PATH}/data" ]]; then
    echo "❌ 错误: ${DATASET_PATH} 不是有效的zarr数据集"
    exit 1
fi

echo "✓ 数据集路径: ${DATASET_PATH}"

# ===================================================================
# 3. 配置文件检查
# ===================================================================
echo ""
echo "[3/5] 检查配置文件..."

CONFIG_PATH="UniT/config/spatiotemporal_residual_vqvae.yaml"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "❌ 错误: 配置文件未找到: ${CONFIG_PATH}"
    exit 1
fi

echo "✓ 配置文件: ${CONFIG_PATH}"

# ===================================================================
# 4. 显示训练配置
# ===================================================================
echo ""
echo "[4/5] 训练配置:"
echo "────────────────────────────────────────────────────────────────"
echo "  模型: Spatiotemporal Residual VQ-VAE"
echo "  理念: \"Feeling the Unexpected\" (感受那份意外)"
echo "  序列长度: 16帧"
echo "  Batch Size: 4 (视频数据显存占用大)"
echo "  最大Epochs: 150"
echo "  GPU: ${GPU_ID}"
echo ""
echo "  课程学习策略:"
echo "    - Stage 1 (0-40):   触觉重建 (force_identity=True)"
echo "    - Stage 2 (40-90):  VQ训练 (force_identity=False)"
echo "    - Stage 3 (90-150): FiLM Fine-tune (端到端优化)"
echo "────────────────────────────────────────────────────────────────"

# ===================================================================
# 5. 启动训练
# ===================================================================
echo ""
echo "[5/5] 启动训练..."
echo ""

# 设置GPU
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# 启动训练
python train.py \
    --config-dir=./UniT/config \
    --config-name=spatiotemporal_residual_vqvae \
    gpus="${GPU_ID}"

echo ""
echo "================================================================"
echo "✓ 训练完成!"
echo "================================================================"
