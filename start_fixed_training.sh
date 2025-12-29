#!/bin/bash
# 启动修复后的Residual VQ-VAE训练
# 修复内容:
#   1. VQ Codebook: 256 → 512
#   2. Sparse Loss: 0.001 → 0.01
#   3. 课程学习: 2阶段 → 4阶段平滑过渡

set -e

echo "=========================================="
echo "启动Residual VQ-VAE v2 训练 (修复版)"
echo "=========================================="
echo ""
echo "修复内容:"
echo "  ✓ VQ Codebook: 512 (原256)"
echo "  ✓ Sparse Loss: 0.01 (原0.001)"
echo "  ✓ 课程学习: 4阶段平滑 (原2阶段)"
echo ""
echo "预期改善:"
echo "  → VQ Loss (验证): 3232 → <200"
echo "  → Epoch 30不再爆炸"
echo "  → 训练/验证比: 0.17 → >0.5"
echo ""
echo "训练配置:"
echo "  - Epochs: 100"
echo "  - Batch Size: 16"
echo "  - GPU: 0"
echo "  - 预计时长: 3-4小时"
echo ""
echo "=========================================="

# 切换到项目根目录
cd /root/code/zpw/tac/UniT-Align

# 检查数据集是否存在
if [ ! -d "data/residual_tactile_train_v2" ]; then
    echo "❌ 错误: 数据集不存在 data/residual_tactile_train_v2"
    exit 1
fi

echo "✓ 数据集检查通过"
echo ""

# 激活conda环境 (如果需要)
# source ~/miniforge3/bin/activate unitenv

# 开始训练
echo "开始训练..."
echo "日志将保存到: representation_models/*/wandb/latest-run/files/output.log"
echo ""

python train.py \
    --config-dir=./UniT/config \
    --config-name=residual_vqvae_representation

echo ""
echo "=========================================="
echo "训练完成!"
echo "=========================================="
echo ""
echo "检查结果:"
echo "  1. 查看最新checkpoint: ls -lht representation_models/*/checkpoints/"
echo "  2. 分析训练日志: python analyze_training.py"
echo "  3. 查看WandB: 见日志中的链接"
echo ""
