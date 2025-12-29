#!/bin/bash
# 启动Residual VQ-VAE v3延长训练 (200 epochs)
# ============================================
# 完整训练200个epoch,观察后期VQ loss收敛情况

set -e

echo "=========================================="
echo "Residual Tactile VQ-VAE v3 Extended Training (200 epochs)"
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
BATCH_SIZE=${BATCH_SIZE:-16}     # Batch size (必须与v3一致)
GPU_ID=${GPU_ID:-1}              # GPU ID (必须与v3一致: GPU 1)

echo "训练配置:"
echo "  配置文件: residual_vqvae_representation_v3_extended.yaml"
echo "  Prophet backbone: resnet18"
echo "  use_state: false"
echo "  Validation interval: $VAL_INTERVAL epochs"
echo "  Batch size: $BATCH_SIZE"
echo "  GPU: $GPU_ID"
echo "  Max Epochs: 200"
echo "  Dataset: data/residual_tactile_train_v2"
echo ""

echo "=========================================="
echo "课程学习计划 (6阶段):"
echo "=========================================="
echo ""
echo "  Phase 1 (0-30):     aux=1.0,  dropout=0.5"
echo "  Phase 2 (30-50):    aux=0.5,  dropout=0.3"
echo "  Phase 3 (50-70):    aux=0.2,  dropout=0.1"
echo "  Phase 4 (70-120):   aux=0.1,  dropout=0.0"
echo "  Phase 5 (120-160):  aux=0.05, dropout=0.0"
echo "  Phase 6 (160-200):  aux=0.02, dropout=0.0"
echo ""
echo "💡 根据之前的训练观察:"
echo "  - Epoch 70+: VQ loss开始收敛"
echo "  - Epoch 90+: rec loss接近aux rec loss (0.01)"
echo "  - 延长训练可能让两者完全持平"
echo ""
echo "=========================================="
echo "启动训练..."
echo "=========================================="

python train.py \
    --config-dir=./UniT/config \
    --config-name=residual_vqvae_representation_v3_extended \
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
