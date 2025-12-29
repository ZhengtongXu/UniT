# Residual VQ-VAE 训练诊断报告

**训练Run**: `2025-12-24T17-33-04residual_vq_final`
**总Epochs**: 100
**数据集**: residual_tactile_train_v2 (50 episodes, 3564 frames)
**诊断日期**: 2025-12-25

---

## 🔥 核心问题诊断

### **问题1: VQ Loss爆炸 (CRITICAL)**

#### 现象
```
训练集 VQ Loss: 548.6
验证集 VQ Loss: 3232.4
训练/验证比: 0.17 (严重过拟合)
```

#### 根本原因
**VQ Codebook 欠拟合 + 残差分布崩溃**

从训练曲线看：
- **Phase 1 (Epoch 0-29)**: Loss正常 (~3.3), VQ还在学习
- **Phase 2 (Epoch 30-99)**: Loss持续爆炸 (3.3 → 294), VQ完全无法量化残差

**关键指标分析**:
```python
# 残差范数
train_raw_residual_norm: 15.08  # Prophet预测与真实的差异
train_residual_norm: 1.39       # VQ量化后的残差
压缩率: 0.092 (VQ将残差压缩了90.8%)

# 验证集
val_raw_residual_norm: 34.86    # 比训练集大2.3倍！
```

**诊断**:
1. VQ Codebook (256 codes) **太小**，无法表达复杂的残差分布
2. VQ Loss = MSE(raw_residual, codebook_embedding) 持续增大
3. 验证集残差范数是训练集的2.3倍 → **Prophet在验证集上预测能力极差**

---

### **问题2: Prophet Encoder 学习失败**

#### 现象
```
Sparse Loss (训练): 0.015
→ 平均残差L1范数: 15.08

Lambda_sparse: 0.001 (固定)
```

#### 根本原因
**梯度信号太弱 + 没有动作信息**

**梯度流分析**:
```python
# residual_vqmodel.py:224
raw_residual = z_real.detach() - z_exp
```

Prophet Encoder的梯度来源:
1. **Sparse Loss**: λ·||r||₁ = 0.001 × 15.08 = **0.015** (极弱)
2. **VQ Loss**: 通过commitment loss传回，但VQ已经崩溃 (548.6)
3. **Recon Loss**: 通过Decoder间接传回，但信号被稀释

**问题**:
- λ=0.001太小，Prophet学习动力不足
- z_real.detach() 切断了obs_encoder的直接监督
- Prophet **只用V+S预测触觉，缺少动作A** → 无法预测动态接触

---

### **问题3: 课程学习策略失效**

#### 现象
```
Phase 1 (0-29 epoch): aux_loss_weight=1.0, 平均Loss=3.3
Phase 2 (30-99 epoch): aux_loss_weight=0.1, 平均Loss=105.6

Epoch 29→99: Loss从14.3飙升到294.0 (20倍增长)
```

#### 根本原因
**Decoder突然断奶 + VQ瓶颈太紧**

**Phase 1期间**:
- Aux Loss占主导 → Decoder主要学习重建**z_real** (完美的触觉latent)
- Hybrid路径(z_exp + r_vq)是辅助，Decoder不依赖它

**Phase 2切换**:
- Aux权重从1.0→0.1 (削弱10倍)
- Decoder被迫适应**VQ量化后的残差**(r_vq)
- 但VQ只有256 codes，信息严重丢失 → Decoder崩溃

**证据**:
```
重建Loss (Hybrid, 训练): 0.026
重建Loss (Aux, 训练): 0.017
→ Aux/Hybrid比: 0.64 (Aux更容易重建)

重建Loss (Hybrid, 验证): 0.169
重建Loss (Aux, 验证): 0.030
→ Aux/Hybrid比: 0.18 (Hybrid路径在验证集上差6倍！)
```

---

## 📊 关键指标总结

| 指标 | 训练集 | 验证集 | 状态 |
|------|--------|--------|------|
| **Total Loss** | 548.7 | 3232.6 | ⚠️ 过拟合6倍 |
| **VQ Loss** | 548.6 | 3232.4 | 🔥 Codebook崩溃 |
| **Recon Loss (Hybrid)** | 0.026 | 0.169 | ⚠️ 泛化差 |
| **Recon Loss (Aux)** | 0.017 | 0.030 | ✅ obs_encoder正常 |
| **Sparse Loss** | 0.015 | 0.035 | ⚠️ 梯度太弱 |
| **残差范数 (原始)** | 15.08 | 34.86 | 🔥 Prophet预测失败 |
| **残差范数 (VQ后)** | 1.39 | - | ⚠️ 90%压缩损失 |
| **判别器Loss** | 0.999 | 1.010 | ✅ 正常工作 |

---

## 🔍 与Idea1.md的对比

| 设计目标 | 原始构想 | 实际实现 | 问题 |
|----------|----------|----------|------|
| **残差学习** | r = z_real - z_exp | ✅ 实现 | Prophet学习失败 |
| **信息瓶颈** | 稀疏Loss | ✅ VQ(256) + Sparse | VQ太紧，崩溃 |
| **Prophet输入** | V + S + **A** | ❌ V + S only | **缺失动作信息** |
| **梯度隔离** | z_real.detach() | ✅ 实现 | 隔离过度，梯度太弱 |
| **物理先验** | 预测可见接触 | ❌ 无法预测 | 缺少动作A |

---

## 🧪 根本原因分析

### **核心矛盾**:
1. **VQ Bottleneck (256 codes)** 太紧 → 无法量化复杂残差
2. **Prophet没有动作A** → 无法预测动态接触 (shear force, sliding)
3. **Sparse Loss权重太小** (λ=0.001) → Prophet学习动力不足
4. **课程学习切换太突然** (aux_weight: 1.0→0.1) → Decoder适应失败

### **恶性循环**:
```
Prophet预测差
  → 残差大 (raw_residual_norm=15.08)
  → VQ无法量化 (256 codes不够)
  → VQ Loss爆炸 (548.6 → 3232.4)
  → Prophet梯度信号崩溃
  → Prophet继续预测差
```

---

## 🩺 具体问题定位

### **问题A: Prophet为什么预测差？**

**输入**: V (腕部RGB) + S (关节角度)
**缺失**: A (动作/速度)

**物理分析**:
- 静态场景: V+S足够 (机器人不动，触觉只依赖物体形状)
- 动态场景: **必须有A** (机器人推压时，触觉=f(V, S, A, 物体刚度))
  - 例: 同样的V+S，向左推vs向右推，触觉完全不同

**数据集检查**:
```python
# residual_tactile_dataset.py
def __getitem__(self, idx):
    return {
        'tactile_image': ...,
        'visual_image': ...,
        'robot_state': ...,  # 只有当前状态qpos
        # 'action': MISSING!  # 没有动作或速度信息
    }
```

**结论**: Prophet根本没有足够信息来预测触觉！

---

### **问题B: VQ Codebook为什么崩溃？**

**配置**:
```yaml
residual_vq_config:
  n_e: 256              # Codebook大小
  e_dim: 3              # 每个code的维度
  beta: 0.25            # Commitment loss权重
```

**分析**:
- 残差空间: [B, 3, 20, 16] = 960维
- VQ Codebook: 256个3维向量
- 每个空间位置(20×16=320个)都要从256个code中选1个
- 总共需要表达: 256^320 种可能 → **但只有256个基础code**

**问题**:
- 如果残差分布复杂(多模态、高方差)，256 codes远远不够
- VQ Loss = Σ MSE(r[i,j], nearest_code) → 每个位置的误差累积

**验证集VQ Loss爆炸 (3232.4)**:
- 训练集: Prophet记住了训练数据的V+S→T映射，残差小
- 验证集: Prophet无法泛化 → 残差大 (34.86 vs 15.08) → VQ更无法量化

---

### **问题C: 为什么Aux/Hybrid重建差距这么大？**

**训练集**:
```
Hybrid (z_exp + r_vq): 0.026
Aux (z_real): 0.017
→ Hybrid比Aux差 54%
```

**验证集**:
```
Hybrid (z_exp + r_vq): 0.169
Aux (z_real): 0.030
→ Hybrid比Aux差 463%！
```

**原因**:
1. **z_exp质量差** (Prophet没学好)
2. **r_vq丢失太多信息** (VQ只有256 codes)
3. z_exp + r_vq ≠ z_real (残差无法补偿预测误差)

**理想情况**: z_exp + r = z_real (完美分解)
**实际情况**: z_exp + quantize(r) ≈ 垃圾 (VQ瓶颈太紧)

---

## 🔧 推荐修复方案

### **方案1: 扩大VQ Codebook (紧急)**

```yaml
residual_vq_config:
  n_e: 2048  # 256 → 2048 (8倍)
  e_dim: 3
  beta: 0.25
```

**理由**:
- 256 codes明显不够表达残差复杂性
- 对比标准VQ-VAE (8192 codes)，256太激进
- 先放松瓶颈，验证架构是否可行

---

### **方案2: 给Prophet加入动作信息 (关键)**

**修改ProphetEncoder**:
```python
class ProphetEncoder(nn.Module):
    def __init__(self, ..., action_dim: int = 8):  # 新增
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, visual, robot_state, action):  # 新增action
        v_feat = self.visual_encoder(visual)
        s_feat = self.state_encoder(robot_state)
        a_feat = self.action_encoder(action)  # 新增

        fused = torch.cat([v_feat, s_feat, a_feat], dim=1)
        ...
```

**数据集修改**:
```python
# residual_tactile_dataset.py
def __getitem__(self, idx):
    return {
        ...,
        'action': self.zarr_root['data/action'][real_idx],  # 新增
    }
```

**理由**:
- 动态触觉依赖于力和速度，必须有action
- 这是Idea1.md的原始设计，但被遗漏了

---

### **方案3: 增大Sparse Loss权重 (立即可行)**

```yaml
model:
  lambda_sparse: 0.01  # 0.001 → 0.01 (10倍)
```

**或使用课程学习**:
```yaml
curriculum_learning:
  phases:
    - end_epoch: 30
      lambda_sparse: 0.001
      aux_loss_weight: 1.0
    - end_epoch: 60
      lambda_sparse: 0.005  # 渐进增大
      aux_loss_weight: 0.5
    - end_epoch: 100
      lambda_sparse: 0.01
      aux_loss_weight: 0.1
```

**理由**:
- 当前λ=0.001梯度太弱 (Loss只有0.015)
- Prophet需要更强的稀疏约束才能学习

---

### **方案4: 平滑课程学习切换 (重要)**

**当前问题**: aux_weight从1.0突降到0.1 (90%削减)

**改进**:
```yaml
curriculum_learning:
  phases:
    - end_epoch: 30
      aux_loss_weight: 1.0
    - end_epoch: 50
      aux_loss_weight: 0.5   # 新增中间阶段
    - end_epoch: 70
      aux_loss_weight: 0.2
    - end_epoch: 100
      aux_loss_weight: 0.1
```

**理由**:
- 给Decoder更多时间适应VQ量化噪声
- 避免Phase 2的Loss爆炸

---

### **方案5: 检查Prophet的预测能力 (调试)**

**新增监控指标**:
```python
# 在training_step中添加
with torch.no_grad():
    # 预测相关性
    correlation = F.cosine_similarity(
        z_exp.flatten(1),
        z_real.flatten(1),
        dim=1
    ).mean()
    self.log('train/prophet_correlation', correlation)

    # 预测MSE
    pred_mse = F.mse_loss(z_exp, z_real)
    self.log('train/prophet_mse', pred_mse)
```

**目的**:
- 量化Prophet的预测质量
- 如果correlation < 0.3，说明Prophet完全没学到物理规律

---

## 📋 实验优先级

### **Phase 1: 紧急修复 (当务之急)**

1. ✅ **扩大VQ Codebook** (256 → 2048)
   - 工作量: 1行配置修改
   - 预期效果: VQ Loss降低5-10倍

2. ✅ **增大Sparse Loss** (0.001 → 0.01)
   - 工作量: 1行配置修改
   - 预期效果: Prophet学习加速

3. ✅ **平滑课程学习** (1.0 → 0.5 → 0.2 → 0.1)
   - 工作量: 修改4行配置
   - 预期效果: 避免Phase 2崩溃

**重新训练，观察是否解决VQ爆炸问题。**

---

### **Phase 2: 架构改进 (核心问题)**

4. ✅ **加入动作信息** (V+S → V+S+A)
   - 工作量: 修改ProphetEncoder + Dataset (1小时)
   - 预期效果: Prophet预测能力提升50%+

5. ✅ **添加Prophet监控指标**
   - 工作量: 10行代码
   - 目的: 验证Prophet是否学到物理规律

**重新训练，验证Prophet能否学到合理的预期。**

---

### **Phase 3: 消融实验 (验证想法)**

6. 对比实验:
   - Baseline: 无VQ (直接用raw_residual)
   - VQ-256 vs VQ-512 vs VQ-1024 vs VQ-2048
   - 有动作A vs 无动作A
   - λ=0.001 vs 0.01 vs 0.1

7. 可视化分析:
   - z_exp vs z_real的相似度热图
   - 残差分布(raw_residual)的直方图
   - VQ Codebook利用率(每个code被选中的频率)

---

## 🎯 成功标准

修复后应达到:

| 指标 | 当前值 | 目标值 |
|------|--------|--------|
| VQ Loss (验证) | 3232 | < 50 |
| 训练/验证 VQ比 | 0.17 | > 0.7 |
| Prophet相关性 | ? | > 0.5 |
| 残差范数(验证) | 34.86 | < 10 |
| Hybrid/Aux重建比(验证) | 5.6 | < 1.5 |

---

## 📝 结论

**你的"Feeling the Unexpected"想法本身是合理的**，但当前实现存在3个致命缺陷:

1. **VQ Codebook太小** (256 codes) → 信息瓶颈过紧
2. **Prophet缺少动作输入** (V+S不够) → 无法预测动态接触
3. **Sparse Loss权重太弱** (λ=0.001) → 梯度信号不足

**最紧急的修复**:
- 扩大Codebook到2048
- 加入动作信息

**长期验证**:
- 检查Prophet是否真的学到物理规律
- 确认残差r是否只包含"意外"信息

**如果修复后仍然失败，可能需要反思**:
- 256维的residual VQ是否太激进？
- 是否需要更复杂的Prophet架构(如Transformer)？
- 数据集中是否真的包含足够的"意外事件"(Jamming/碰撞)？
