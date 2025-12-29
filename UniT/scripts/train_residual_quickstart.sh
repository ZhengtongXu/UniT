#!/bin/bash
# ============================================
# Residual Tactile Learning 快速开始脚本
# ============================================
#
# 此脚本展示完整的训练流程:
# 1. 数据转换 (多模态对齐)
# 2. 训练Residual VQ-VAE
# 3. 评估和可视化
#
# 使用: bash UniT/scripts/train_residual_quickstart.sh
# ============================================

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Residual Tactile Learning 训练流程${NC}"
echo -e "${GREEN}========================================${NC}\n"

# ============================================
# 阶段0: 环境检查
# ============================================
echo -e "${YELLOW}[阶段0] 检查环境...${NC}"

# 检查conda环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${RED}错误: 未激活conda环境${NC}"
    echo "请先运行: conda activate unitenv"
    exit 1
fi

echo -e "  ✓ Conda环境: $CONDA_DEFAULT_ENV"

# 检查GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}警告: 未检测到NVIDIA GPU${NC}"
else
    echo -e "  ✓ GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
fi

# 检查Python包
python -c "import torch; import pytorch_lightning; import zarr" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 缺少必要的Python包${NC}"
    echo "请确保已安装: torch, pytorch_lightning, zarr"
    exit 1
fi

echo -e "  ✓ Python包完整\n"

# ============================================
# 阶段1: 数据准备
# ============================================
echo -e "${YELLOW}[阶段1] 数据转换...${NC}"

# 数据路径
INPUT_DIR="/mlp_vepfs/share/zpw/tac/script/tac_data/test/peg_in_hole1"
OUTPUT_ZARR="data/residual_tactile_train_v2"  # Updated to match config

# 检查输入数据
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}错误: 输入数据目录不存在: $INPUT_DIR${NC}"
    echo "请确认数据路径正确"
    exit 1
fi

# 如果Zarr已存在,询问是否覆盖
if [ -d "$OUTPUT_ZARR" ]; then
    echo -e "${YELLOW}Zarr数据集已存在: $OUTPUT_ZARR${NC}"
    read -p "是否重新转换? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除旧数据..."
        rm -rf "$OUTPUT_ZARR"
    else
        echo "跳过数据转换"
    fi
fi

# 运行数据转换 (如果需要)
if [ ! -d "$OUTPUT_ZARR" ]; then
    echo "开始转换数据..."
    python UniT/dataset/convert_residual_data.py \
        --input_dir "$INPUT_DIR" \
        --output_zarr "$OUTPUT_ZARR" \
        --visual_size 224 224 \
        --tactile_size 128 160

    if [ $? -ne 0 ]; then
        echo -e "${RED}数据转换失败${NC}"
        exit 1
    fi
    echo -e "${GREEN}  ✓ 数据转换完成${NC}\n"
else
    echo -e "${GREEN}  ✓ 使用现有Zarr数据${NC}\n"
fi

# ============================================
# 阶段2: 训练模型
# ============================================
echo -e "${YELLOW}[阶段2] 训练Residual VQ-VAE...${NC}"

# 训练参数
LAMBDA_SPARSE=${LAMBDA_SPARSE:-0.01}  # 默认0.01,可通过环境变量覆盖
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_EPOCHS=${MAX_EPOCHS:-100}
VAL_INTERVAL=${VAL_INTERVAL:-3}  # 验证频率: 每N个epoch验证一次
GPU_ID=${GPU_ID:-0}

echo "训练参数:"
echo "  Lambda_sparse: $LAMBDA_SPARSE"
echo "  Batch size: $BATCH_SIZE"
echo "  Max epochs: $MAX_EPOCHS"
echo "  Validation interval: $VAL_INTERVAL epochs"
echo "  GPU: $GPU_ID"
echo

# 询问是否开始训练
read -p "开始训练? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "跳过训练"
    exit 0
fi

# 运行训练
python UniT/workspace/train_residual_vqvae_workspace.py \
    model.lambda_sparse=$LAMBDA_SPARSE \
    batch_size=$BATCH_SIZE \
    max_epochs=$MAX_EPOCHS \
    validation.val_check_interval=$VAL_INTERVAL \
    validation.image_log_interval=$VAL_INTERVAL \
    gpus="${GPU_ID},"

if [ $? -ne 0 ]; then
    echo -e "${RED}训练失败${NC}"
    exit 1
fi

echo -e "${GREEN}  ✓ 训练完成${NC}\n"

# ============================================
# 阶段3: 后续步骤提示
# ============================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}训练流程完成!${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo "下一步:"
echo "  1. 查看wandb: https://wandb.ai/your-project/residual_tactile"
echo "  2. 调整lambda_sparse继续训练 (课程学习):"
echo "     LAMBDA_SPARSE=0.05 bash UniT/scripts/train_residual_quickstart.sh"
echo "  3. 集成到Policy:"
echo "     - 使用 UniT/model/residual_feature_extractor.py"
echo "     - 参考计划文件中的集成示例"
echo

# 显示最新checkpoint
LATEST_CKPT=$(find representation_models -name "last.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$LATEST_CKPT" ]; then
    echo -e "最新checkpoint: ${GREEN}$LATEST_CKPT${NC}"
fi
