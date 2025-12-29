# Residual VQ-VAE v4 两阶段训练说明
**实验名称:** residual_vqvae_v4_twostage
**训练会话:** residual_v4_twostage
**配置文件:** `UniT/config/residual_vqvae_representation_v4_twostage.yaml`

---

## 🎯 核心思想

### 用户提出的策略 (两阶段分离训练)

**Stage 1 (Epoch 0-50): 专注训练Observation Encoder**
- **目标:** 让 `z_real` 达到最佳重建质量
- **Loss组合:**
  - ✅ Aux Loss (z_real重建) = **1.0** (主导)
  - ✅ Sparse Loss = **0.01 → 0.005** (让Prophet学习预测)
  - ❌ Hybrid Loss (z_exp+VQ重建) = **0.0** (不使用)
- **关键:** Decoder专注学习从 `z_real` 重建,不受VQ干扰
- **状态:** Obs Encoder **可训练**, Prophet Encoder **可训练**

**Stage 2 (Epoch 50-150): 冻结Obs Encoder,训练VQ Decoder**
- **目标:** 让 `z_exp + VQ(residual)` 重建达到 Stage 1 的 Aux 水平
- **Loss组合:**
  - ❌ Aux Loss = **0.0** (完全放弃)
  - ✅ Hybrid Loss = **1.0** (主导)
  - ✅ Sparse Loss = **0.002 → 0.0005** (弱约束,防止Prophet偏离)
- **关键:** 🔒 **冻结 Obs Encoder** (z_real固定), Decoder聚焦适应VQ路径
- **状态:** Obs Encoder **冻结**, Prophet Encoder **可训练**, VQ **可训练**, Decoder **可训练**

---

## 📋 课程学习计划

### Stage 1: 训练Observation Encoder

| Phase | Epoch | aux_weight | hybrid_weight | lambda_sparse | freeze_obs | 目标 |
|-------|-------|------------|---------------|---------------|------------|------|
| 1.1 Warmup | 0-24 | 1.0 | 0.0 | 0.01 | ❌ | Obs Encoder快速收敛 |
| 1.2 Convergence | 25-49 | 1.0 | 0.0 | 0.005 | ❌ | Aux Loss达到最佳 |

**预期结果 (Epoch 50):**
- Aux Recon Loss ≈ **-0.5 ~ 0.0** (最佳重建质量)
- Prophet已经学会初步预测 (Sparse Loss下降)
- Decoder从 z_real 重建能力强

---

### Stage 2: 训练VQ Decoder重建

| Phase | Epoch | aux_weight | hybrid_weight | lambda_sparse | freeze_obs | 目标 |
|-------|-------|------------|---------------|---------------|------------|------|
| 2.1 Adaptation | 50-79 | 0.0 | 1.0 | 0.002 | ✅ | Decoder适应VQ噪声 |
| 2.2 Fine-tuning | 80-119 | 0.0 | 1.0 | 0.001 | ✅ | 精调VQ重建质量 |
| 2.3 Polish | 120-149 | 0.0 | 1.0 | 0.0005 | ✅ | 最终打磨 |

**预期结果 (Epoch 150):**
- Hybrid Recon Loss ≈ **Stage 1 的 Aux Loss** (达到同样重建质量)
- VQ Loss 收敛到稳定值
- Prophet预测准确 (Sparse Loss很小)

---

## 🔑 关键设计

### 1. **为什么Stage 1不使用Hybrid Loss?**

传统方法 (v3):
```
L = L_hybrid + aux_weight * L_aux
```
- 问题: Decoder同时学习两条路径,可能导致冲突
- Decoder可能"作弊": 主要依赖z_real,忽略VQ路径

v4两阶段:
```
Stage 1: L = L_aux (hybrid_weight=0)
Stage 2: L = L_hybrid (aux_weight=0)
```
- 优势: 两个目标**完全分离**,避免冲突
- Stage 1: Decoder专注学习高质量重建 (从z_real)
- Stage 2: Decoder必须适应VQ路径 (z_real路径被冻结)

---

### 2. **为什么Stage 2要冻结Obs Encoder?**

**如果不冻结:**
- z_real 可能会"退化"以适应VQ路径
- Obs Encoder可能学到"配合VQ"的特征,而非最优特征

**冻结后:**
- z_real **固定为Stage 1的最优表征**
- 强制Prophet、VQ、Decoder去适应这个固定目标
- 保证z_real的质量不受Stage 2影响

---

### 3. **Sparse Loss的作用变化**

**Stage 1 (lambda=0.01 → 0.005):**
- 作用: 让Prophet学习预测触觉latent
- 目标: z_exp ≈ z_real (残差小)

**Stage 2 (lambda=0.002 → 0.0005):**
- 作用: **正则约束** - 防止Prophet偏离Stage 1学到的预测
- 不是主要目标,仅防止Prophet"乱飘"

---

## 📊 监控指标

### Stage 1 (Epoch 0-50)

**关键指标:**
- ✅ `train/recon_loss_aux` 应该持续下降 → **-0.5 左右**
- ✅ `train/sparse_loss` 应该下降 (Prophet学习中)
- ⚠️ `train/recon_loss` (hybrid) 应该**很高** (因为hybrid_weight=0,不优化)
- ⚠️ `train/vq_loss` 可能震荡 (VQ没有训练信号)

**目标达成标志:**
- Aux Loss 收敛到最小值
- Sparse Loss 明显下降 (说明Prophet学会了预测)

---

### Stage 2 (Epoch 50-150)

**关键指标:**
- ✅ `train/recon_loss` (hybrid) 应该持续下降 → **接近Stage 1的Aux Loss**
- ✅ `train/vq_loss` 应该收敛 (VQ开始学习)
- ✅ `train/sparse_loss` 应该很小 (Prophet保持预测能力)
- ⚠️ `train/recon_loss_aux` 会**固定** (因为aux_weight=0,不优化)

**Epoch 50 切换时的现象:**
```
📍 Epoch 50: 进入新阶段!
  阶段名称: stage2_adaptation
  aux_loss_weight: 1.0 → 0.0
  hybrid_loss_weight: 0.0 → 1.0
  lambda_sparse: 0.005 → 0.002
  freeze_obs_encoder: False → True
  ✓ 已冻结Observation Stream
```

**预期:**
- Hybrid Recon Loss 会突然从很高的值开始下降
- 前几个epoch可能不稳定 (Decoder适应期)
- 10-20 epoch后应该稳定下降

---

## 🎯 成功标准

### Stage 1成功 (Epoch 50)
- [x] Aux Recon Loss < 0.0 (负值说明重建质量好)
- [x] Sparse Loss 明显低于初始值 (说明Prophet学会了)
- [x] Decoder能够从z_real高质量重建

### Stage 2成功 (Epoch 150)
- [x] Hybrid Recon Loss ≈ Stage 1的Aux Loss (目标达成!)
- [x] VQ Loss 收敛到稳定值 (0.05-0.15)
- [x] Sparse Loss < 0.001 (Prophet保持预测能力)

**最终验证:**
- 用Epoch 150 checkpoint在validation set测试
- Hybrid重建质量应该接近Stage 1的Aux重建质量

---

## 🛠️ 训练命令

### 查看训练状态
```bash
tmux attach -t residual_v4_twostage
```

### 停止训练
```bash
tmux kill-session -t residual_v4_twostage
```

### 查看Checkpoint
```bash
ls -lh representation_models/2025-12-25T15-32-46residual_vqvae_v4_twostage/checkpoints/
```

### 查看WandB
项目: `residual_tactile`
Run名称: `2025-12-25T15-32-46residual_vqvae_v4_twostage`

---

## 🔬 与v3的对比

| 特性 | v3 (失败) | v4 (两阶段) |
|------|-----------|-------------|
| **训练策略** | 单一课程学习 | 两阶段分离训练 |
| **Aux Loss** | 1.0 → 0.5 → 0.2 → 0.1 | Stage1: 1.0, Stage2: 0.0 |
| **Hybrid Loss** | 始终=1.0 | Stage1: 0.0, Stage2: 1.0 |
| **Obs Encoder** | 始终可训练 | Stage1: 可训练, Stage2: 🔒冻结 |
| **Loss爆炸** | Epoch 70+ 爆炸 | **不应该爆炸** |
| **原因** | aux_weight降太快 | 两阶段完全分离 |

**v3的问题:**
- Decoder同时学习两条路径 → 冲突
- aux_weight降到0.1时,VQ路径还没学好 → 崩溃

**v4的优势:**
- Stage 1: Decoder专注学习z_real → 质量最优
- Stage 2: Decoder必须适应VQ → 强制学习
- Obs Encoder冻结 → z_real质量不退化

---

## 📝 预期挑战

### 挑战1: Stage 2初期不稳定

**现象:** Epoch 50-55时,Loss可能震荡

**原因:**
- Decoder从未见过VQ量化噪声
- 需要适应期

**解决:**
- 正常现象,观察10-20 epoch
- 如果持续震荡,降低学习率

---

### 挑战2: VQ Loss可能不收敛

**现象:** VQ Loss持续震荡,不收敛到固定值

**原因:**
- Sparse Loss与VQ Loss冲突
- Codebook利用率低

**解决:**
- **不用担心!** 只要Hybrid Recon Loss下降,VQ就在工作
- VQ Loss绝对值不重要,重要的是重建质量

---

### 挑战3: Hybrid无法达到Aux水平

**现象:** Epoch 150时,Hybrid Loss 仍然高于Stage 1 Aux Loss

**可能原因:**
1. VQ容量不足 (`e_dim=3`太小)
2. Prophet预测不准 (Sparse Loss太大)
3. Decoder容量不足

**解决:**
- 增加VQ容量: `e_dim: 3 → 8`
- 降低Sparse Loss权重
- 延长Stage 2训练时间

---

## 🎓 理论基础

### 两阶段训练的本质

**信息瓶颈视角:**
```
I(X; Z_real) >> I(X; Z_exp + VQ(residual))
```
- Stage 1: 最大化 I(X; Z_real) (学习最优表征)
- Stage 2: 最大化 I(X; Z_exp + VQ(residual)) (在瓶颈约束下逼近最优)

**Teacher-Student视角:**
```
Teacher: Obs Encoder (z_real)
Student: Prophet + VQ + Decoder (z_exp + VQ)
```
- Stage 1: Train Teacher → 最优性能
- Stage 2: Freeze Teacher, Train Student → 知识蒸馏

---

## ✅ 验证清单

训练结束后,验证以下指标:

### Checkpoint: Epoch 50 (Stage 1结束)
- [ ] `val/recon_loss_aux` < 0.0
- [ ] `train/sparse_loss` 明显下降
- [ ] Aux重建图像质量好

### Checkpoint: Epoch 150 (Stage 2结束)
- [ ] `val/rec_loss` (hybrid) ≈ Epoch 50的 `val/recon_loss_aux`
- [ ] `train/vq_loss` < 0.2
- [ ] Hybrid重建图像质量接近Aux重建

### 最终测试
- [ ] 在test set上验证
- [ ] Hybrid重建 vs Aux重建 (视觉对比)
- [ ] 用于downstream task (perception/policy)

---

**祝训练成功! 🚀**
