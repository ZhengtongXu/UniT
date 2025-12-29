#!/bin/bash
# ==========================================
# 恢复 Residual VQ-VAE v3 训练 (Tmux版本)
# ==========================================
#
# 从 Epoch 100 checkpoint 恢复，继续训练到 Epoch 200
# 保持在同一个 wandb run (3ijlxpfh) 中记录
#
# 使用方法:
#   bash resume_v3_training_tmux.sh
#
# 查看训练状态:
#   tmux attach -t residual_v3_resume
#
# 退出tmux (不停止训练):
#   按 Ctrl+B 然后按 D
#
# 停止训练:
#   tmux kill-session -t residual_v3_resume
# ==========================================

SESSION_NAME="residual_v3_resume"

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

# 创建新的tmux会话
echo "======================================"
echo "恢复 Residual VQ-VAE v3 训练"
echo "======================================"
echo ""
echo "训练配置:"
echo "  Checkpoint: checkpoint-epoch99.ckpt"
echo "  WandB Run ID: 3ijlxpfh"
echo "  继续训练: Epoch 100 → 200"
echo "  GPU: 1"
echo "  Batch size: 16"
echo ""
echo "课程学习 (延长版):"
echo "  Phase 4 (70-120):   aux_weight=0.1"
echo "  Phase 5 (120-160):  aux_weight=0.05"
echo "  Phase 6 (160-200):  aux_weight=0.02"
echo ""
echo "💡 说明:"
echo "  - 训练会自动从 Epoch 100 开始"
echo "  - 所有日志会继续记录到同一个 wandb run"
echo "  - Checkpoint会保存到原目录"
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
tmux new-session -d -s $SESSION_NAME -c "$SCRIPT_DIR" "python resume_v3_training.py; echo ''; echo '训练结束，按Enter键退出'; read"

echo "✓ Tmux会话已创建: $SESSION_NAME"
echo ""
echo "连接到会话查看训练:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "WandB链接:"
echo "  https://wandb.ai/pweiii_github/residual_tactile/runs/3ijlxpfh"
