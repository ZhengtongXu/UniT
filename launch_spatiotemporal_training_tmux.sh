#!/bin/bash
# ===================================================================
# Spatiotemporal Residual VQ-VAE 训练启动脚本 (tmux版本)
# ===================================================================
# 功能: 在tmux会话中后台运行时空触觉表征学习训练
#
# 使用方法:
#   bash launch_spatiotemporal_training_tmux.sh
#
# 查看训练:
#   tmux attach -t spatiotemporal_train
#
# 分离会话:
#   Ctrl+B, 然后按 D
# ===================================================================

set -e

SESSION_NAME="spatiotemporal_train"

echo "================================================================"
echo "Spatiotemporal Residual VQ-VAE 训练启动 (tmux版本)"
echo "================================================================"
echo ""

# ===================================================================
# 1. 检查tmux
# ===================================================================
if ! command -v tmux &> /dev/null; then
    echo "❌ 错误: tmux未安装"
    echo "   请运行: sudo apt-get install tmux"
    exit 1
fi

# ===================================================================
# 2. 检查是否已有同名会话
# ===================================================================
if tmux has-session -t ${SESSION_NAME} 2>/dev/null; then
    echo "⚠️  警告: tmux会话 '${SESSION_NAME}' 已存在"
    echo ""
    echo "请选择操作:"
    echo "  1) 附加到现有会话 (tmux attach -t ${SESSION_NAME})"
    echo "  2) 杀死现有会话并重新启动"
    echo "  3) 取消"
    echo ""
    read -p "请输入选择 [1/2/3]: " choice

    case $choice in
        1)
            tmux attach -t ${SESSION_NAME}
            exit 0
            ;;
        2)
            echo "正在杀死现有会话..."
            tmux kill-session -t ${SESSION_NAME}
            ;;
        3)
            echo "已取消"
            exit 0
            ;;
        *)
            echo "无效选择，已取消"
            exit 1
            ;;
    esac
fi

# ===================================================================
# 3. 环境检查
# ===================================================================
echo "[1/4] 检查环境..."

# 检查conda
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "❌ 错误: Conda环境未激活"
    echo "   请运行: conda activate unitenv"
    exit 1
fi

echo "✓ Conda环境: ${CONDA_DEFAULT_ENV}"

# 检查GPU
GPU_ID=${GPU_ID:-0}
echo "✓ 使用GPU: ${GPU_ID}"

# ===================================================================
# 4. 数据集检查
# ===================================================================
echo ""
echo "[2/4] 检查数据集..."

DATASET_PATH="data/residual_tactile_train_v2"

if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "❌ 错误: 数据集未找到: ${DATASET_PATH}"
    exit 1
fi

echo "✓ 数据集: ${DATASET_PATH}"

# ===================================================================
# 5. 配置文件检查
# ===================================================================
echo ""
echo "[3/4] 检查配置文件..."

CONFIG_PATH="UniT/config/spatiotemporal_residual_vqvae.yaml"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "❌ 错误: 配置文件未找到: ${CONFIG_PATH}"
    exit 1
fi

echo "✓ 配置文件: ${CONFIG_PATH}"

# ===================================================================
# 6. 创建tmux会话并启动训练
# ===================================================================
echo ""
echo "[4/4] 创建tmux会话..."

# 获取conda初始化脚本路径
CONDA_BASE=$(conda info --base)
CONDA_INIT="${CONDA_BASE}/etc/profile.d/conda.sh"

# 创建tmux会话
tmux new-session -d -s ${SESSION_NAME}

# 在tmux会话中执行命令
tmux send-keys -t ${SESSION_NAME} "source ${CONDA_INIT}" C-m
tmux send-keys -t ${SESSION_NAME} "conda activate ${CONDA_DEFAULT_ENV}" C-m
tmux send-keys -t ${SESSION_NAME} "cd $(pwd)" C-m
tmux send-keys -t ${SESSION_NAME} "export CUDA_VISIBLE_DEVICES=${GPU_ID}" C-m

# 启动训练
tmux send-keys -t ${SESSION_NAME} "echo '================================================================'" C-m
tmux send-keys -t ${SESSION_NAME} "echo 'Spatiotemporal Residual VQ-VAE 训练'" C-m
tmux send-keys -t ${SESSION_NAME} "echo '理念: Feeling the Unexpected (感受那份意外)'" C-m
tmux send-keys -t ${SESSION_NAME} "echo '================================================================'" C-m
tmux send-keys -t ${SESSION_NAME} "echo ''" C-m

tmux send-keys -t ${SESSION_NAME} "python train.py --config-dir=./UniT/config --config-name=spatiotemporal_residual_vqvae gpus='${GPU_ID}'" C-m

echo ""
echo "================================================================"
echo "✓ tmux会话已创建: ${SESSION_NAME}"
echo "================================================================"
echo ""
echo "使用以下命令管理训练:"
echo ""
echo "  查看训练进度:"
echo "    tmux attach -t ${SESSION_NAME}"
echo ""
echo "  分离会话 (训练继续后台运行):"
echo "    按 Ctrl+B, 然后按 D"
echo ""
echo "  列出所有tmux会话:"
echo "    tmux ls"
echo ""
echo "  杀死会话 (停止训练):"
echo "    tmux kill-session -t ${SESSION_NAME}"
echo ""
echo "================================================================"
echo ""

# 询问是否立即附加
read -p "是否立即附加到训练会话? [y/N]: " attach_now

if [[ "$attach_now" =~ ^[Yy]$ ]]; then
    tmux attach -t ${SESSION_NAME}
fi
