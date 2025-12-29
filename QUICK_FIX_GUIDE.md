# Residual VQ-VAE 快速修复指南

**问题**: VQ Loss从548爆炸到3232，验证集比训练集高6倍

**根本原因**: 3个致命缺陷
1. VQ Codebook太小 (256 codes)
2. Prophet缺少动作信息 (只有V+S)
3. Sparse Loss权重太弱 (λ=0.001)

---

## 🚀 立即修复 (5分钟)

### Step 1: 扩大VQ Codebook

编辑 `UniT/config/residual_vqvae_representation.yaml`:

```yaml
# 第40行
residual_vq_config:
  n_e: 2048  # 改为 2048 (原来256)
  e_dim: 3
  beta: 0.25
  legacy: false
```

---

### Step 2: 增大Sparse Loss

同一个文件，第37行:

```yaml
lambda_sparse: 0.01  # 改为 0.01 (原来0.001)
```

---

### Step 3: 平滑课程学习切换

同一个文件，第106-114行:

```yaml
curriculum_learning:
  enabled: true
  phases:
    - end_epoch: 30
      aux_loss_weight: 1.0
      name: 'teacher_forcing'
    - end_epoch: 50
      aux_loss_weight: 0.5  # 新增中间阶段
      name: 'gradual_adaptation'
    - end_epoch: 70
      aux_loss_weight: 0.2  # 新增
      name: 'student_finetuning'
    - end_epoch: 100
      aux_loss_weight: 0.1
      name: 'final_polish'
```

---

## 重新训练

```bash
python train.py --config-dir=./UniT/config --config-name=residual_vqvae_representation
```

**预期改善**:
- VQ Loss (验证): 3232 → < 100
- 训练/验证比: 0.17 → > 0.7
- Loss不再在Epoch 30爆炸

---

## 🔧 下一步 (需要1小时)

### Step 4: 加入动作信息

**问题**: 当前数据集**没有action**，只有robot_state
- Prophet用 (V + S) 预测触觉
- 但动态接触依赖速度/力 → **必须有动作A**

**检查原始数据**:
```bash
# 检查原始zarr是否有action
python -c "import zarr; print(list(zarr.open('data/residual_tactile_train_v2').keys()))"
```

**如果没有action**:

#### 选项A: 从qpos计算速度 (近似)
```python
# 在 residual_tactile_dataset.py 中
def __getitem__(self, idx):
    real_idx = self.indices[idx]

    # 计算速度 (差分)
    if real_idx > 0:
        qpos_curr = self.zarr_root['data/robot_state'][real_idx]
        qpos_prev = self.zarr_root['data/robot_state'][real_idx - 1]
        velocity = (qpos_curr - qpos_prev) / 0.1  # 假设10Hz采样
    else:
        velocity = np.zeros(8)

    return {
        'tactile_image': ...,
        'visual_image': ...,
        'robot_state': robot_state,
        'action': velocity  # 新增: 速度作为action的代理
    }
```

#### 选项B: 重新采集数据（包含action）
- 修改数据采集脚本，保存每帧的action
- 重新生成zarr数据集

---

### Step 5: 修改Prophet Encoder

编辑 `UniT/taming/models/residual_vqmodel.py`:

```python
class ProphetEncoder(nn.Module):
    def __init__(self,
                 visual_backbone: str = 'resnet34',
                 visual_pretrained: bool = True,
                 state_dim: int = 8,
                 action_dim: int = 8,  # 新增
                 latent_shape: tuple = (3, 16, 20),
                 hidden_dims: list = [512, 256, 128]):
        super().__init__()

        # 视觉编码器 (不变)
        ...

        # 状态编码器 (不变)
        ...

        # 动作编码器 (新增)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True)
        )
        action_feat_dim = hidden_dims[1]

        # Fusion MLP (修改)
        fusion_input_dim = visual_feat_dim + state_feat_dim + action_feat_dim  # 加action
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dims[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        ...

    def forward(self, visual_img, robot_state, action):  # 新增action参数
        v_feat = self.visual_encoder(visual_img)
        s_feat = self.state_encoder(robot_state)
        a_feat = self.action_encoder(action)  # 新增

        fused = torch.cat([v_feat, s_feat, a_feat], dim=1)  # 拼接action
        fused = self.fusion_mlp(fused)

        z_flat = self.to_spatial(fused)
        z_exp = z_flat.reshape(-1, *self.latent_shape)
        return z_exp
```

同一个文件，修改ResidualVQModel.forward:

```python
def forward(self, batch):
    tactile = batch['tactile_image']
    visual = batch['visual_image']
    robot_state = batch['robot_state']
    action = batch['action']  # 新增

    # Prophet Stream: V+S+A → z_exp
    z_exp = self.prophet_encoder(visual, robot_state, action)  # 加action

    # 其余不变
    ...
```

---

### Step 6: 更新配置

编辑 `UniT/config/residual_vqvae_representation.yaml`:

```yaml
model:
  prophet_config:
    visual_backbone: resnet34
    visual_pretrained: true
    state_dim: 8
    action_dim: 8  # 新增
    latent_shape: [3, 20, 16]
    hidden_dims: [512, 256, 128]
```

---

## 📊 监控指标

训练时重点观察:

### 必须正常的指标
- `train/vq_loss < 100` (当前548)
- `val/vq_loss < 150` (当前3232)
- `train/vq_loss / val/vq_loss > 0.7` (当前0.17)

### Prophet质量指标 (新增)
在 `residual_vqmodel.py` 的 `training_step` 中添加:

```python
# 在训练步骤中 (optimizer_idx == 0 时)
with torch.no_grad():
    # Prophet预测相关性
    z_exp_flat = outputs['z_exp'].flatten(1)
    z_real_flat = outputs['z_real'].flatten(1)
    correlation = F.cosine_similarity(z_exp_flat, z_real_flat, dim=1).mean()
    self.log('train/prophet_correlation', correlation, prog_bar=True)

    # Prophet预测MSE
    pred_mse = F.mse_loss(outputs['z_exp'], outputs['z_real'])
    self.log('train/prophet_mse', pred_mse)
```

**目标值**:
- `prophet_correlation > 0.5` (Prophet学到了物理规律)
- `prophet_mse < 5.0` (预测误差合理)

---

## ✅ 成功标准

修复后应该看到:

| 指标 | 修复前 | 修复后目标 |
|------|--------|-----------|
| VQ Loss (验证) | 3232 | < 100 |
| 训练/验证比 | 0.17 | > 0.7 |
| 残差范数(验证) | 34.86 | < 10 |
| Prophet相关性 | ? | > 0.5 |
| Epoch 30 Loss跳变 | 爆炸 | 平滑 |

---

## 🎯 实验顺序

### 第一轮: 快速修复 (今天完成)
1. ✅ 扩大VQ (256→2048)
2. ✅ 增大λ (0.001→0.01)
3. ✅ 平滑课程 (4阶段)

→ **训练100 epoch，观察VQ Loss是否降低**

---

### 第二轮: 加入动作 (明天)
4. ✅ 实现速度计算 (从qpos差分)
5. ✅ 修改ProphetEncoder
6. ✅ 添加监控指标

→ **训练100 epoch，观察Prophet相关性**

---

### 第三轮: 消融实验 (后天)
7. 对比: VQ-512 vs VQ-1024 vs VQ-2048
8. 对比: 有action vs 无action
9. 对比: λ=0.001 vs 0.01 vs 0.1

→ **找到最优配置**

---

## 🔥 常见问题

### Q1: 为什么VQ Codebook要2048，不是256？
**A**: 残差空间有960维 (3×20×16)，需要足够的codes来覆盖复杂分布。256太小会导致量化误差爆炸。

### Q2: 如果没有action数据怎么办？
**A**:
- **临时方案**: 用qpos差分近似速度
- **长期方案**: 重新采集数据，保存action
- **理论上**: 没有action，Prophet无法预测动态接触

### Q3: Prophet学不好怎么办？
**A**: 检查:
1. `prophet_correlation` 是否 > 0.3
2. 数据集是否有足够的静态vs动态场景
3. 是否需要更强的架构 (Transformer vs ResNet)

### Q4: Epoch 30还是爆炸怎么办？
**A**: 进一步平滑课程:
- 6阶段: 1.0 → 0.7 → 0.5 → 0.3 → 0.2 → 0.1
- 延长Phase 1: 0-50 epoch

---

## 📞 下一步行动

1. **立即**: 修改3个配置，重新训练
2. **等待结果**: 观察VQ Loss曲线 (预计3-4小时)
3. **如果成功**: 开始加入action
4. **如果失败**: 回来讨论其他方案

Good luck! 🚀
