#!/bin/bash
# ==========================================
# Residual VQ-VAE 课程学习训练脚本
# ==========================================
#
# 使用Callback实现三阶段课程学习 (单次训练运行)
#
# 使用方法:
#   bash train_residual_curriculum.sh
#
# 自定义配置 (可选):
#   VAL_INTERVAL=5 BATCH_SIZE=32 GPU_ID=1 MAX_EPOCHS=100 bash train_residual_curriculum.sh
#
# ==========================================

set -e

echo "=========================================="
echo "Residual Tactile VQ-VAE 课程学习训练"
echo "=========================================="
echo ""

# 检查环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "错误: 请先激活conda环境"
    echo "运行: conda activate unitenv"
    exit 1
fi

echo "环境: $CONDA_DEFAULT_ENV"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo ""

# 训练配置参数 (可通过环境变量覆盖)
VAL_INTERVAL=${VAL_INTERVAL:-3}     # 验证频率 (默认每3个epoch)
BATCH_SIZE=${BATCH_SIZE:-16}        # Batch size
GPU_ID=${GPU_ID:-0}                 # GPU ID
MAX_EPOCHS=${MAX_EPOCHS:-100}       # 总训练epoch数

echo "训练配置:"
echo "  Validation interval: $VAL_INTERVAL epochs"
echo "  Batch size: $BATCH_SIZE"
echo "  GPU: $GPU_ID"
echo "  Max epochs: $MAX_EPOCHS"
echo "  Dataset: data/residual_tactile_train_v2"
echo ""

echo "课程学习策略:"
echo "  Phase 1 (Epochs 0-14):  λ=0.001 (预热)"
echo "  Phase 2 (Epochs 15-44): λ=0.01  (稀疏化)"
echo "  Phase 3 (Epochs 45-99): λ=0.05  (精细调优)"
echo ""

echo "=========================================="
echo "开始训练..."
echo "=========================================="

# 启动训练 (启用课程学习)
python UniT/workspace/train_residual_vqvae_workspace.py \
    model.lambda_sparse=0.001 \
    batch_size=$BATCH_SIZE \
    max_epochs=$MAX_EPOCHS \
    validation.val_check_interval=$VAL_INTERVAL \
    validation.image_log_interval=$VAL_INTERVAL \
    curriculum_learning.enabled=true \
    gpus=[${GPU_ID}]

echo ""
echo "=========================================="
echo "✓ 训练完成!"
echo "=========================================="
echo ""

# 显示最新的checkpoints
echo "最新的Checkpoints:"
ls -lht representation_models/*/checkpoints/*.ckpt 2>/dev/null | head -5 || echo "未找到checkpoint文件"
