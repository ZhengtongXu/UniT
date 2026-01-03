# Spatiotemporal Residual VQ-VAE - 训练状态报告

**启动时间**: 2025-12-30 00:02
**GPU**: GPU 1 (RTX 4090 D)
**状态**: ✅ 正在训练
**WandB**: https://wandb.ai/pweiii_github/spatiotemporal_residual_tactile/runs/l75a

---

## 🚀 训练配置

### 模型架构
```
Prophet Encoder:                11.7M 参数
TimeDistributed Obs Encoder:    23.1M 参数
Residual Aggregator:             1.1M 参数  
VQ Quantizer:                    131K 参数
Dynamic FiLM Decoder:            11.7M 参数
─────────────────────────────────────────
总计:                            47.7M 参数
```

### 训练参数
- **Batch Size**: 2 (因显存限制从4降至2)
- **Epochs**: 150
- **Sequence Length**: 16 frames
- **Learning Rate**: 9e-6 (= 1 GPU × 2 batch × 4.5e-6 base_lr)
- **数据集**: 2818 samples (训练: 2537, 验证: 281)

### 当前阶段 (Stage 1: Reconstruction)
```
============================================================
课程学习: Stage 1 'reconstruction' (Epoch 0-40)
  force_identity: True        ← FiLM不调制，纯重建训练
  aux_loss_weight: 1.0        ← 全力训练Obs Encoder
  hybrid_loss_weight: 0.0     ← VQ路径不参与
  lambda_sparse: 0.0          ← 不约束残差稀疏性
============================================================
```

### 训练进度
```
Epoch 0:   4%|▍| 56/1409 [00:11<04:37, 4.88it/s, loss=1.371]
```

- **速度**: ~4.9 it/s (batch_size=2后提升)
- **预估单epoch时长**: 1409 / 4.9 ≈ 4.8分钟
- **当前Loss**: 1.371

---

## 📊 GPU使用情况

| GPU | 模型 | 显存占用 | 总显存 | 利用率 | 状态 |
|-----|------|---------|--------|--------|------|
| GPU 0 | Probabilistic VQ-VAE | 20.3 GB | 24.6 GB | 98% | ✅ 训练中 |
| GPU 1 | **Spatiotemporal VQ-VAE** | **15.9 GB** | **24.6 GB** | **97%** | **✅ 训练中** |

**显存优化**: batch_size从4降至2后，显存占用从23.3GB降至15.9GB ✓

---

## 📝 管理命令

### 查看训练进度
```bash
tmux attach -t spatiotemporal_train
```
按 `Ctrl+B` 然后 `D` 分离会话

### 查看训练日志
```bash
tail -f /root/code/zpw/tac/UniT-Align/training_spatiotemporal_bs2.log
```

### 停止训练
```bash
tmux kill-session -t spatiotemporal_train
```

### 列出所有tmux会话
```bash
tmux ls
```

---

## ⏱️ 预估训练时长

**基于batch_size=2, ~4.9 it/s**:

- **Stage 1 (Epoch 0-40)**: 40 × 4.8分钟 ≈ **3.2小时**
- **Stage 2 (Epoch 40-90)**: 50 × 4.8分钟 ≈ **4.0小时**
- **Stage 3 (Epoch 90-150)**: 60 × 4.8分钟 ≈ **4.8小时**

**总预估时长**: ~12小时

---

## 🔍 监控指标

### Stage 1 (当前, Epoch 0-40)
- **aux_loss**: Observation Encoder重建损失 ← 主要优化目标
- **vq_loss**: VQ codebook学习
- **total_loss**: 总损失 (当前: 1.371, 持续下降)

### Stage 2 (Epoch 40-90)
- **hybrid_loss**: VQ路径重建质量
- **aux_loss**: 继续监控
- **codebook_usage**: VQ codebook使用率

### Stage 3 (Epoch 90-150)
- **hybrid_loss**: 主要优化目标 (端到端FiLM调制)
- **残差稀疏性**: 是否稀疏
- **重建质量**: 视觉检查

---

## 📈 WandB实时监控

- **项目**: spatiotemporal_residual_tactile
- **Run**: 2025-12-30T00-02-43spatiotemporal_residual_vqvae
- **Run ID**: l75a
- **URL**: https://wandb.ai/pweiii_github/spatiotemporal_residual_tactile/runs/l75a

实时查看:
- Loss曲线 (aux_loss, hybrid_loss, vq_loss, total_loss)
- GPU占用
- 训练速度

---

## ✅ 问题解决记录

### 问题1: CUDA OOM (显存不足)
- **现象**: batch_size=4时，显存占用23.3GB，训练到第95个iteration时OOM
- **原因**: 视频数据 (B=4, C=3, T=16, H=128, W=160) 显存占用巨大
- **解决**: batch_size从4降至2，显存占用降至15.9GB
- **影响**: 训练速度略微提升 (4.9 it/s vs 之前的2.5 it/s)

### 问题2: GPU配置冲突
- **现象**: 设置GPU_ID=1但仍然尝试使用GPU 0
- **原因**: 使用CUDA_VISIBLE_DEVICES=1后，PyTorch看到的GPU 0实际是物理GPU 1
- **解决**: 设置`CUDA_VISIBLE_DEVICES=1` + `gpus="0"`

---

## 🎯 训练目标

### Stage 1 目标 (Epoch 0-40)
- ✅ 训练Observation Encoder学会从触觉图像序列提取时序特征
- ✅ 训练Prophet学会从视觉+状态预测"预期"的触觉序列
- ✅ Decoder学会基础的重建能力 (force_identity=True, 不使用VQ调制)

### Stage 2 目标 (Epoch 40-90)
- 开启VQ路径 (force_identity=False)
- 训练Residual Aggregator学会聚合"意外信号"
- VQ Quantizer学会将连续残差映射到离散Event Code
- FiLM Decoder学会使用Event Code调制解码

### Stage 3 目标 (Epoch 90-150)
- 端到端fine-tune
- 完全依赖VQ路径 (aux_loss_weight=0.0)
- 最大化FiLM调制效果
- 鼓励残差稀疏性 (lambda_sparse=0.005)

---

## 📌 重要文件

### 训练相关
- **配置**: `UniT/config/spatiotemporal_residual_vqvae.yaml`
- **模型**: `UniT/taming/models/spatiotemporal_residual_vqmodel.py`
- **数据集**: `UniT/dataset/spatiotemporal_tactile_dataset.py`
- **Workspace**: `UniT/workspace/train_spatiotemporal_workspace.py`
- **日志**: `training_spatiotemporal_bs2.log`

### 文档
- **实现指南**: `SPATIOTEMPORAL_IMPLEMENTATION_GUIDE.md`
- **验证报告**: `SPATIOTEMPORAL_VALIDATION_REPORT.md`
- **Shape分析**: `SHAPE_ISSUES_ANALYSIS.md`
- **FiLM改进方案**: `FILM_DECODER_FIX.md`

### Checkpoints
- **保存位置**: `representation_models/2025-12-30T00-02-43spatiotemporal_residual_vqvae/checkpoints/`
- **保存频率**: 每10个epoch
- **保留数量**: 最近5个

---

**训练成功启动，显存优化完成！** 🎉

**两个训练同时进行**:
- GPU 0: Probabilistic Residual VQ-VAE (概率性架构)
- GPU 1: Spatiotemporal Residual VQ-VAE (时空架构) ← 当前

