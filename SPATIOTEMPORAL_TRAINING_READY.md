# Spatiotemporal Residual VQ-VAE - Training Ready ✓

**Status**: 完整实现完成，已通过所有测试，可以开始训练

**Branch**: `spatiotemporal-residual-vae`

**Date**: 2025-12-29

---

## 🎯 核心理念: "Feeling the Unexpected" (感受那份意外)

这是一个基于视频序列的触觉表征学习架构，专注于捕捉**意外事件**而非平滑趋势：

- **Prophet Stream**: 从视觉+状态预测"预期的平滑物理趋势"
- **Observation Stream**: 编码真实触觉视频序列
- **Residual**: `z_real - z_pred` 得到"意外信号"
- **VQ Quantization**: 将连续的意外信号压缩成离散的**Event Code**
- **Dynamic FiLM Decoder**: 使用Event Code动态调制解码过程

---

## ✅ 已完成的实现

### 1. 核心模型 (850+ 行)
**文件**: `UniT/taming/models/spatiotemporal_residual_vqmodel.py`

**核心组件**:
```python
SpatiotemporalResidualVQModel
├─ TimeDistributedObsEncoder    # 对每帧应用2D VAE编码器
├─ ProphetStream                 # Causal TCN预测序列
├─ ResidualTransformerAggregator # Transformer聚合残差
├─ VectorQuantizer2              # 离散化为Event Code
└─ DynamicFiLMDecoder            # 基于Event Code调制解码
    ├─ StyleExpander (GRU)       # 将静态code展开为动态参数
    └─ TemporalFiLMBlock         # 逐帧FiLM调制
```

**参数量**: ~47.7M (可训练参数)

**关键设计**:
- TimeDistributed包装器: 复用2D CNN处理视频帧 (节省显存)
- Causal TCN: 确保t时刻预测只依赖≤t的信息
- Detach操作: `z_real.detach() - z_pred` 防止Obs Encoder适应Prophet
- force_identity机制: Stage 1时FiLM层输出γ=1, β=0 (不调制)

### 2. 数据集 (275 行)
**文件**: `UniT/dataset/spatiotemporal_tactile_dataset.py`

**功能**:
- 从zarr数据中采样长度为T=16的连续片段
- Episode-aware采样 (确保帧连续性)
- 返回格式:
  ```python
  {
      'tactile_video': (3, T, H, W),      # T帧触觉图像
      'visual_context': (3, H_v, W_v),    # 单帧视觉上下文
      'state_seq': (T, state_dim),        # 机器人状态序列
      'action_seq': (T, action_dim)       # 动作序列
  }
  ```

**数据统计** (residual_tactile_train_v2):
- 总帧数: 3564
- Episodes: 50
- 有效样本数: 2818 (每个样本是16帧序列)

### 3. 配置文件 (174 行)
**文件**: `UniT/config/spatiotemporal_residual_vqvae.yaml`

**关键参数**:
```yaml
batch_size: 4             # 视频数据显存占用大
max_epochs: 150
seq_length: 16
base_learning_rate: 4.5e-6

model:
  vq_config:
    n_e: 512              # Event codebook大小
    e_dim: 256            # Event embedding维度
```

**三阶段课程学习**:
- **Stage 1 (0-40 epochs)**: 触觉重建
  - force_identity=True (不使用VQ调制)
  - aux_loss_weight=1.0 (全力训练重建)
  
- **Stage 2 (40-90 epochs)**: VQ训练
  - force_identity=False (开始使用VQ调制)
  - aux_loss_weight=0.5, hybrid_loss_weight=0.5
  
- **Stage 3 (90-150 epochs)**: FiLM Fine-tune
  - force_identity=False
  - aux_loss_weight=0.0, hybrid_loss_weight=1.0 (完全依赖VQ路径)

### 4. 训练Workspace (323 行)
**文件**: `UniT/workspace/train_spatiotemporal_workspace.py`

**功能**:
- PyTorch Lightning训练流程
- SpatiotemporalCurriculumCallback: 动态切换force_identity
- WandB日志记录
- 定期checkpoint保存

### 5. 启动脚本
**文件**: 
- `launch_spatiotemporal_training.sh` (前台运行)
- `launch_spatiotemporal_training_tmux.sh` (后台运行)

**使用方法**:
```bash
# 方式1: 前台运行
bash launch_spatiotemporal_training.sh

# 方式2: tmux后台运行
bash launch_spatiotemporal_training_tmux.sh

# 指定GPU
GPU_ID=0 bash launch_spatiotemporal_training.sh
```

### 6. 详细文档
**文件**: `SPATIOTEMPORAL_IMPLEMENTATION_GUIDE.md` (544 行)

包含完整的架构说明、设计决策、使用指南。

---

## 🧪 测试结果

所有测试均已通过：

### ✅ 环境测试
```
PyTorch version: 2.7.1+cu128
CUDA available: True
CUDA device: NVIDIA GeForce RTX 4090 D
```

### ✅ 数据集测试
```
Dataset loaded: 2818 samples
tactile_video shape: (3, 16, 128, 160)
visual_context shape: (3, 224, 224)
state_seq shape: (16, 8)
action_seq shape: (16, 8)
```

### ✅ 模型测试
```
Total params: 47.70M
Trainable params: 47.70M
Forward pass: ✓
Output shapes:
  - recon_video: (B, 3, 16, 128, 160)
  - z_pred_seq: (B, 16, 256)
  - z_real_seq: (B, 16, 256)
  - residual_seq: (B, 16, 256)
  - event_code: (B, 256)
  - vq_loss: scalar
```

### ✅ 训练步骤测试
```
Training step: ✓
Loss computation: ✓
Total loss: ~2.22
```

### ✅ 配置加载测试
```
Configuration loaded: ✓
Curriculum learning: 3 phases
  - Stage 1: stage1_reconstruction (force_identity=True)
  - Stage 2: stage2_vq_training (force_identity=False)
  - Stage 3: stage3_film_finetune (force_identity=False)
```

### ✅ Workspace初始化测试
```
Workspace initialized: ✓
Model type: SpatiotemporalResidualVQModel
Trainer type: Trainer
```

---

## 🚀 开始训练

### 前提条件
1. ✅ conda环境已激活 (`unitenv`)
2. ✅ 数据集已准备 (`data/residual_tactile_train_v2`)
3. ✅ GPU可用

### 启动训练

**推荐方式 (tmux后台运行)**:
```bash
cd /root/code/zpw/tac/UniT-Align
bash launch_spatiotemporal_training_tmux.sh
```

启动后会创建tmux会话 `spatiotemporal_train`，可以使用以下命令管理：

```bash
# 查看训练进度
tmux attach -t spatiotemporal_train

# 分离会话 (训练继续后台运行)
# 在tmux会话中按: Ctrl+B, 然后按 D

# 列出所有tmux会话
tmux ls

# 杀死会话 (停止训练)
tmux kill-session -t spatiotemporal_train
```

**前台运行方式**:
```bash
bash launch_spatiotemporal_training.sh
```

### 训练监控

**WandB**:
- 项目名称: `spatiotemporal_residual_tactile`
- 自动记录loss曲线、学习率等

**Checkpoints**:
- 保存位置: `representation_models/<timestamp>_spatiotemporal_residual_vqvae/checkpoints/`
- 保存频率: 每10个epoch
- 保留最近5个checkpoint

**课程学习阶段切换**:
训练时会自动打印阶段切换信息：
```
============================================================
课程学习: 切换到阶段 'stage2_vq_training' (Epoch 40)
  force_identity: False
  aux_loss_weight: 0.5
  hybrid_loss_weight: 0.5
  lambda_sparse: 0.01
============================================================
```

---

## 📊 预期训练时长

基于RTX 4090 D + Batch Size 4:

- **Stage 1 (0-40 epochs)**: ~6-8小时
- **Stage 2 (40-90 epochs)**: ~10-12小时
- **Stage 3 (90-150 epochs)**: ~12-15小时
- **总计**: ~28-35小时

*注: 实际时长取决于GPU负载和数据加载速度*

---

## 🔧 常见问题

### Q: 如何调整batch size?
A: 修改配置文件 `UniT/config/spatiotemporal_residual_vqvae.yaml`:
```yaml
batch_size: 4  # 降低可减少显存占用
```

### Q: 如何使用预训练的Observation Encoder?
A: 修改配置文件:
```yaml
model:
  obs_encoder_ckpt: "/path/to/pretrained/encoder.ckpt"
```

### Q: 训练显存不足怎么办?
A: 
1. 降低batch_size (4 → 2)
2. 减小seq_length (16 → 8)
3. 使用梯度累积 (在trainer中添加 `accumulate_grad_batches`)

### Q: 如何从checkpoint恢复训练?
A: 修改配置文件:
```yaml
trainer:
  resume_from_checkpoint: "/path/to/checkpoint.ckpt"
```

---

## 📝 下一步

训练完成后可以：

1. **评估Event Code质量**:
   - 可视化codebook使用分布
   - 分析不同event code对应的触觉模式

2. **下游任务微调**:
   - 使用学习到的表征进行触觉分类
   - 集成到manipulation policy中

3. **消融实验**:
   - 对比有无VQ路径的性能
   - 测试不同codebook大小的影响
   - 验证课程学习的必要性

---

## 📌 重要文件清单

```
UniT-Align/
├─ UniT/
│  ├─ taming/models/
│  │  └─ spatiotemporal_residual_vqmodel.py      (核心模型)
│  ├─ dataset/
│  │  └─ spatiotemporal_tactile_dataset.py       (数据集)
│  ├─ workspace/
│  │  └─ train_spatiotemporal_workspace.py       (训练workspace)
│  └─ config/
│     └─ spatiotemporal_residual_vqvae.yaml      (配置文件)
├─ launch_spatiotemporal_training.sh             (启动脚本)
├─ launch_spatiotemporal_training_tmux.sh        (tmux启动脚本)
├─ SPATIOTEMPORAL_IMPLEMENTATION_GUIDE.md        (详细文档)
└─ SPATIOTEMPORAL_TRAINING_READY.md              (本文档)
```

---

**实现完成时间**: 2025-12-29

**实现者**: Claude Sonnet 4.5

**核心创新**: 
- TimeDistributed架构 (节省显存)
- Event Code表征 (离散化意外事件)
- Dynamic FiLM调制 (基于事件的动态解码)
- 三阶段课程学习 (稳定训练)

**准备就绪，可以开始训练！** 🚀
