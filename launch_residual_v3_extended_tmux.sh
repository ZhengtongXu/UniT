#!/bin/bash
# ==========================================
# Residual VQ-VAE v3 延长训练启动脚本 (Tmux版本)
# ==========================================
#
# 完整训练200个epoch,观察后期VQ loss收敛
#
# 使用方法:
#   bash launch_residual_v3_extended_tmux.sh
#
# 查看训练状态:
#   tmux attach -t residual_v3_extended
#
# 退出tmux (不停止训练):
#   按 Ctrl+B 然后按 D
#
# 停止训练:
#   tmux kill-session -t residual_v3_extended
# ==========================================

SESSION_NAME="residual_v3_extended"

# 检查tmux是否已安装
if ! command -v tmux &> /dev/null; then
    echo "错误: tmux未安装"
    exit 1
fi

# 检查会话是否已存在
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "⚠️  会话 '$SESSION_NAME' 已存在"
    echo ""
    echo "选项:"
    echo "  1. 连接到现有会话:  tmux attach -t $SESSION_NAME"
    echo "  2. 删除现有会话:    tmux kill-session -t $SESSION_NAME"
    echo "  3. 查看所有会话:    tmux ls"
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 从环境变量读取训练配置
VAL_INTERVAL=${VAL_INTERVAL:-3}
BATCH_SIZE=${BATCH_SIZE:-16}  # 必须与v3一致
GPU_ID=${GPU_ID:-1}  # 必须与v3一致: GPU 1

# 创建新的tmux会话
echo "======================================"
echo "启动 Residual VQ-VAE v3 延长训练"
echo "======================================"
echo ""
echo "训练配置:"
echo "  配置: residual_vqvae_representation_v3_extended.yaml"
echo "  Max Epochs: 200 (完整训练)"
echo "  Validation interval: $VAL_INTERVAL epochs"
echo "  Batch size: $BATCH_SIZE"
echo "  GPU: $GPU_ID"
echo "  Dataset: data/residual_tactile_train_v2"
echo ""
echo "课程学习 (6阶段):"
echo "  Phase 1-3 (0-70):    建立基础 (aux: 1.0→0.2)"
echo "  Phase 4 (70-120):    开始收敛 (aux: 0.1)"
echo "  Phase 5 (120-160):   VQ主导 (aux: 0.05)"
echo "  Phase 6 (160-200):   完全收敛 (aux: 0.02)"
echo ""
echo "💡 期望:"
echo "  - Epoch 100+: VQ loss持续收敛"
echo "  - Epoch 150+: rec loss ≈ aux rec loss (0.01)"
echo "  - Epoch 200:  完全收敛状态"
echo ""
echo "会话创建后将自动启动训练"
echo ""
echo "查看训练进度:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "退出tmux (不停止训练):"
echo "  按 Ctrl+B 然后按 D"
echo ""
echo "停止训练:"
echo "  tmux kill-session -t $SESSION_NAME"
echo ""
echo "======================================"
sleep 2

# 创建tmux会话并运行训练
tmux new-session -d -s $SESSION_NAME -c "$SCRIPT_DIR" "VAL_INTERVAL=$VAL_INTERVAL BATCH_SIZE=$BATCH_SIZE GPU_ID=$GPU_ID bash $SCRIPT_DIR/launch_residual_v3_extended.sh; echo ''; echo '训练结束，按Enter键退出'; read"

echo "✓ Tmux会话已创建: $SESSION_NAME"
echo ""
echo "连接到会话查看训练:"
echo "  tmux attach -t $SESSION_NAME"
