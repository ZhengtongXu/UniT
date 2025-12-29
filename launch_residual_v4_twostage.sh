#!/bin/bash
# 启动Residual VQ-VAE v4两阶段训练
# ============================================
# Stage 1 (0-50): 训练Obs Encoder (Aux Loss + Sparse Loss)
# Stage 2 (50-150): 冻结Obs Encoder, 训练VQ Decoder (Hybrid Loss)

set -e

echo "=========================================="
echo "Residual Tactile VQ-VAE v4 Two-Stage Training"
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
VAL_INTERVAL=${VAL_INTERVAL:-3}  # 验证频率 (默认每3个epoch)
BATCH_SIZE=${BATCH_SIZE:-16}     # Batch size
GPU_ID=${GPU_ID:-1}              # GPU ID (默认GPU 1)

echo "训练配置:"
echo "  配置文件: residual_vqvae_representation_v4_twostage.yaml"
echo "  训练策略: Two-Stage (Stage1: Obs Encoder, Stage2: VQ Decoder)"
echo "  Prophet backbone: resnet18"
echo "  use_state: false"
echo "  Validation interval: $VAL_INTERVAL epochs"
echo "  Batch size: $BATCH_SIZE"
echo "  GPU: $GPU_ID"
echo "  Max Epochs: 150"
echo "  Dataset: data/residual_tactile_train_v2"
echo ""

echo "=========================================="
echo "训练阶段说明:"
echo "=========================================="
echo ""
echo "【Stage 1: 训练Observation Encoder】 (Epoch 0-50)"
echo "  目标: 让 z_real 达到最佳重建质量"
echo "  Loss: Aux Loss (1.0) + Sparse Loss (0.01→0.005)"
echo "  不使用: Hybrid Loss (0.0)"
echo "  状态: Obs Encoder可训练"
echo ""
echo "【Stage 2: 训练VQ Decoder重建】 (Epoch 50-150)"
echo "  目标: 让 z_exp+VQ 重建达到 Stage 1 的 Aux 水平"
echo "  Loss: Hybrid Loss (1.0) + 弱Sparse Loss (0.002→0.0005)"
echo "  不使用: Aux Loss (0.0)"
echo "  状态: 🔒 Obs Encoder冻结"
echo ""
echo "=========================================="
echo "启动训练..."
echo "=========================================="

python train.py \
    --config-dir=./UniT/config \
    --config-name=residual_vqvae_representation_v4_twostage \
    batch_size=$BATCH_SIZE \
    validation.val_check_interval=$VAL_INTERVAL \
    validation.image_log_interval=$VAL_INTERVAL \
    gpus="${GPU_ID}"

echo ""
echo "=========================================="
echo "✓ 训练完成!"
echo "=========================================="
echo ""
echo "Checkpoints:"
ls -lh representation_models/*/checkpoints/*.ckpt 2>/dev/null | tail -5
