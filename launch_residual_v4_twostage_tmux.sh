#!/bin/bash
# ==========================================
# Residual VQ-VAE v4 两阶段训练启动脚本 (Tmux版本)
# ==========================================
#
# 在tmux会话中启动训练，确保退出终端后训练继续进行
#
# 使用方法:
#   bash launch_residual_v4_twostage_tmux.sh
#
# 自定义配置 (可选):
#   VAL_INTERVAL=1 BATCH_SIZE=32 GPU_ID=0 bash launch_residual_v4_twostage_tmux.sh
#
# 查看训练状态:
#   tmux attach -t residual_v4_twostage
#
# 退出tmux (不停止训练):
#   按 Ctrl+B 然后按 D
#
# 停止训练:
#   tmux kill-session -t residual_v4_twostage
# ==========================================

SESSION_NAME="residual_v4_twostage"

# 检查tmux是否已安装
if ! command -v tmux &> /dev/null; then
    echo "错误: tmux未安装"
    echo "请先安装tmux: apt-get install tmux 或 yum install tmux"
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
BATCH_SIZE=${BATCH_SIZE:-16}
GPU_ID=${GPU_ID:-1}

# 创建新的tmux会话并在后台运行
echo "======================================"
echo "创建tmux会话: $SESSION_NAME"
echo "======================================"
echo ""
echo "训练配置:"
echo "  配置: residual_vqvae_representation_v4_twostage.yaml"
echo "  策略: Two-Stage Training"
echo "  Validation interval: $VAL_INTERVAL epochs"
echo "  Batch size: $BATCH_SIZE"
echo "  GPU: $GPU_ID"
echo "  Max Epochs: 150"
echo "  Dataset: data/residual_tactile_train_v2"
echo ""
echo "训练阶段:"
echo "  Stage 1 (0-50):   训练Obs Encoder (Aux Loss)"
echo "  Stage 2 (50-150): 冻结Obs Encoder, 训练VQ Decoder (Hybrid Loss)"
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

# 创建tmux会话并运行训练 (传递环境变量)
tmux new-session -d -s $SESSION_NAME -c "$SCRIPT_DIR" "VAL_INTERVAL=$VAL_INTERVAL BATCH_SIZE=$BATCH_SIZE GPU_ID=$GPU_ID bash $SCRIPT_DIR/launch_residual_v4_twostage.sh; echo ''; echo '训练结束，按Enter键退出'; read"

echo "✓ Tmux会话已创建: $SESSION_NAME"
echo ""
echo "连接到会话查看训练:"
echo "  tmux attach -t $SESSION_NAME"
