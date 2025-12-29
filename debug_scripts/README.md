# Residual VQ-VAE 验证框架

本目录包含完整的验证脚本,用于在启动100-epoch完整训练前验证模型实现的正确性。

## 📁 文件结构

```
debug_scripts/
├── README.md                          # 本文件
├── run_all_validations.py            # 主验证脚本 (运行全部)
├── debug_data_pipeline.py            # 1. 数据流验证
├── debug_model_forward.py            # 2. 模型前向验证
├── debug_loss_and_gradients.py       # 3. Loss和梯度验证
└── debug_training_dynamics.py        # 4. 训练动态验证

../debug_outputs/                      # 验证输出目录
├── validation_summary.json            # 综合报告
├── data_samples/                      # 数据可视化
├── model_outputs/                     # 模型输出可视化
├── loss_trends/                       # Loss曲线
├── gradient_analysis/                 # 梯度分析
└── training_logs/                     # 训练日志
```

## 🚀 快速开始

### 方式1: 运行全部验证 (推荐)

```bash
cd /root/code/zpw/tac/UniT-Align
conda activate unitenv
python debug_scripts/run_all_validations.py
```

**预计耗时**: 30-42分钟

**输出**:
- 终端显示所有验证进度和结果
- `debug_outputs/validation_summary.json`: 综合报告
- `debug_outputs/`: 所有可视化和详细报告

### 方式2: 单独运行特定验证

```bash
cd /root/code/zpw/tac/UniT-Align

# 1. 数据流验证 (3-5分钟)
python debug_scripts/debug_data_pipeline.py

# 2. 模型前向验证 (5-7分钟)
python debug_scripts/debug_model_forward.py

# 3. Loss和梯度验证 (7-10分钟)
python debug_scripts/debug_loss_and_gradients.py

# 4. 训练动态验证 (15-20分钟)
python debug_scripts/debug_training_dynamics.py
```

## 📋 验证内容详解

### 1. debug_data_pipeline.py (数据流验证)

**验证项**:
- ✓ Dataset.__getitem__ 输出形状、dtype、值域
  - tactile_image: (3, 128, 160), float32, [0, 1]
  - visual_image: (3, 224, 224), float32, ImageNet归一化
  - robot_state: (8,), float32, Z-score归一化
- ✓ DataLoader批次正确性 (collation、无重复)
- ✓ 训练/验证集划分 (episode边界)
- ✓ 归一化参数一致性 (train/val)

**输出**:
- `debug_outputs/data_pipeline_report.json`
- `debug_outputs/data_samples/sample_*_{tactile,visual,robot_state}.png`

### 2. debug_model_forward.py (模型前向传播验证)

**验证项**:
- ✓ ProphetEncoder: (visual, robot_state) → z_exp (B, 3, 20, 16)
  - Visual backbone: ResNet34
  - State encoder: MLP 8D → 256D
  - Fusion: 768D → 128D → spatial projection
- ✓ ObsEncoder: tactile → z_real (B, 3, 20, 16)
  - 下采样倍数: 128→20 (6.4x), 160→16 (10.0x)
- ✓ Residual计算: residual = z_real - z_exp
  - 恒等验证: z_exp + residual == z_real
- ✓ Decoder重建: z_fused → recon (B, 3, 128, 160)

**输出**:
- `debug_outputs/model_forward_report.json`
- `debug_outputs/model_outputs/z_exp_heatmap.png`
- `debug_outputs/model_outputs/z_real_heatmap.png`
- `debug_outputs/model_outputs/residual_heatmap.png`
- `debug_outputs/model_outputs/reconstruction_comparison.png`

### 3. debug_loss_and_gradients.py (Loss和梯度验证)

**验证项**:
- ✓ Reconstruction Loss (L1 + LPIPS + GAN)
  - 初期 (global_step < 10000): disc未启用
  - 后期 (global_step > 10000): disc启用
- ✓ Sparse Loss: λ_sparse * ||residual||_1
- ✓ Total Loss: recon_loss + sparse_loss
- ✓ ProphetEncoder梯度存在且非零
- ✓ ObsEncoder梯度存在且非零
- ✓ Residual梯度路径:
  - 路径1: sparse_loss → residual
  - 路径2: recon_loss → decoder → residual
  - 路径3: total_loss → residual (双路径汇总)

**输出**:
- `debug_outputs/loss_and_gradients_report.json`
- `debug_outputs/gradient_analysis/gradient_flow_paths.png`

### 4. debug_training_dynamics.py (训练动态验证)

**验证项**:
- ✓ 单步训练: 参数更新验证
- ✓ 多步训练: 20 batches loss趋势
- ✓ 完整epoch: 全量训练+验证 (可选)
- ✓ Checkpoint保存: 文件完整性

**输出**:
- `debug_outputs/training_dynamics_report.json`
- `debug_outputs/loss_trends/multi_step_trend.png`
- `debug_outputs/loss_trends/full_epoch_trend.png` (如果运行完整epoch)
- `debug_outputs/test_checkpoints/test_checkpoint.ckpt`

## ✅ 成功标准

所有4个脚本必须返回 **PASS** 状态:

1. ✓ `debug_data_pipeline.py`: 数据加载、格式、划分正确
2. ✓ `debug_model_forward.py`: 所有模块输出形状正确,无NaN/Inf
3. ✓ `debug_loss_and_gradients.py`: Loss计算正确,梯度流正常
4. ✓ `debug_training_dynamics.py`: 参数更新,loss下降,checkpoint保存成功

**达成后**即可启动完整100-epoch训练。

## 🔧 常见问题和修复

### 问题1: Gradient爆炸 (grad_norm > 1e3)

**症状**: `debug_loss_and_gradients.py` 报告梯度范数异常大

**修复**:
```yaml
# 在 UniT/config/residual_vqvae_representation.yaml 中添加:
trainer:
  gradient_clip_val: 1.0  # 梯度裁剪
```

### 问题2: Residual不稀疏 (residual_norm > 0.5)

**症状**: `debug_model_forward.py` 显示residual范数过大

**修复**:
```yaml
# 增加稀疏性惩罚
model:
  lambda_sparse: 0.05  # 从0.01增加到0.05
```

### 问题3: Discriminator崩溃 (disc_loss < 0.1)

**症状**: `debug_loss_and_gradients.py` 显示disc_loss过小

**修复**:
```yaml
# 推迟discriminator启动
lossconfig:
  params:
    disc_start: 15000  # 从10000增加到15000
```

### 问题4: Loss不下降 (>20 batches无改善)

**症状**: `debug_training_dynamics.py` 显示loss持平或上升

**修复**:
```yaml
# 降低学习率
base_learning_rate: 2.0e-6  # 从4.5e-6降低到2.0e-6
```

### 问题5: 内存泄漏 (每batch增长)

**症状**: GPU内存持续增长

**修复**:
```yaml
dataloader:
  persistent_workers: false  # 禁用persistent workers
  num_workers: 4             # 减少worker数量
```

## 📊 输出文件说明

### JSON报告格式

每个脚本生成一个JSON报告,格式如下:

```json
{
  "script": "debug_xxx.py",
  "status": "PASS" or "FAIL",
  "total_issues": 0,
  "issues": [],
  "checks": {
    "check_name_1": true,
    "check_name_2": true,
    ...
  }
}
```

### 综合报告

`validation_summary.json` 包含:
- 所有脚本的运行状态
- 总问题数和详细列表
- 每个脚本的耗时
- 时间戳

## 🎯 下一步

### 如果所有验证通过:

```bash
cd /root/code/zpw/tac/UniT-Align

# 启动完整100-epoch训练
bash launch_residual_training.sh

# 或使用自定义参数
python UniT/workspace/train_residual_vqvae_workspace.py \
    model.lambda_sparse=0.01 \
    max_epochs=100 \
    batch_size=16 \
    gpus="0"
```

### 如果存在失败:

1. 查看详细报告: `debug_outputs/*_report.json`
2. 查看可视化: `debug_outputs/*/`
3. 根据上述"常见问题和修复"调整配置
4. 重新运行失败的脚本

## 📝 设计原理

本验证框架基于 `/root/.claude/plans/peppy-weaving-dragon.md` 中的完整验证计划:

- **分层设计**: 4个主脚本覆盖数据、模型、loss、训练四大核心
- **自动化**: 无需手动检查,所有验证自动完成
- **可视化**: 关键检查点自动生成可视化图表
- **容错**: 每个脚本独立运行,失败不影响其他脚本
- **完整性**: 覆盖从数据加载到训练动态的全流程

## 🔗 相关文档

- 实现计划: `/root/.claude/plans/peppy-weaving-dragon.md`
- 模型架构: `/root/code/zpw/tac/UniT-Align/RESIDUAL_TACTILE_IMPLEMENTATION.md`
- 训练脚本: `/root/code/zpw/tac/UniT-Align/UniT/workspace/train_residual_vqvae_workspace.py`
- 配置文件: `/root/code/zpw/tac/UniT-Align/UniT/config/residual_vqvae_representation.yaml`
