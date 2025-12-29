#!/bin/bash
# 启动Residual VQ-VAE v3训练 (Anti-Overfitting版本)
# 使用课程学习策略: z_exp_dropout从高到低, aux_loss_weight从高到低

set -e

echo "=========================================="
echo "Residual Tactile VQ-VAE v3 Training"
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
BATCH_SIZE=${BATCH_SIZE:-16}     # Batch size (v3: 降低Prophet容量后可用16)
GPU_ID=${GPU_ID:-1}              # GPU ID (默认GPU 1)

echo "训练配置:"
echo "  配置文件: residual_vqvae_representation_v3.yaml"
echo "  Prophet backbone: resnet18 (降低表达能力)"
echo "  use_state: false (仅使用视觉)"
echo "  Validation interval: $VAL_INTERVAL epochs"
echo "  Batch size: $BATCH_SIZE"
echo "  GPU: $GPU_ID"
echo "  Dataset: data/residual_tactile_train_v2"
echo ""

# 启动训练: 使用v3配置文件
# 核心改进: Prophet降低容量 + z_exp_dropout强制VQ学习
echo "=========================================="
echo "启动v3残差训练 (Anti-Overfitting Strategy)"
echo "=========================================="
python train.py \
    --config-dir=./UniT/config \
    --config-name=residual_vqvae_representation_v3 \
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

