#!/bin/bash
# 快速启动Residual VQ-VAE训练 (单阶段版本)

cd /root/code/zpw/tac/UniT-Align

# 检查环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "请先激活环境: conda activate unitenv"
    exit 1
fi

echo "=========================================="
echo "启动Residual VQ-VAE训练"
echo "=========================================="
echo "环境: $CONDA_DEFAULT_ENV"
echo "数据: data/residual_tactile_train (3564 frames, 50 episodes)"
echo "配置: lambda_sparse=0.01, batch_size=16, max_epochs=100"
echo ""

# 启动训练
python UniT/workspace/train_residual_vqvae_workspace.py

