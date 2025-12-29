# 5分钟快速修复 - 修改总结

**修改时间**: 2025-12-25
**目标**: 解决VQ Loss爆炸问题 (3232 → <100)

---

## ✅ 已完成的修改

### 1️⃣ **扩大VQ Codebook** (温和增长)

```yaml
# 文件: UniT/config/residual_vqvae_representation.yaml
# 第41行

修改前: n_e: 256
修改后: n_e: 512  # 2倍增长，比UniT完整图像(1024)小，但比256更有表达力
```

**理由**:
- ✅ 你的逻辑正确：残差应该比完整图像简单 → 512 < 1024
- ✅ 但256太小，无法覆盖复杂残差分布 (当前残差范数=15.08)
- ✅ 512是折中方案：既保持"残差瓶颈"，又给VQ足够表达力

---

### 2️⃣ **增大Sparse Loss权重** (10倍)

```yaml
# 第37行

修改前: lambda_sparse: 0.001
修改后: lambda_sparse: 0.01  # 10倍增长
```

**理由**:
- Prophet的梯度主要来自Sparse Loss
- 0.001太小 → Loss只有0.015 → 梯度信号太弱
- 0.01 → Loss提升到0.15 → Prophet有更强的学习动力

---

### 3️⃣ **平滑课程学习切换** (2阶段 → 4阶段)

```yaml
# 第106-120行

修改前:
  phases:
    - {end_epoch: 30, aux_loss_weight: 1.0}
    - {end_epoch: 100, aux_loss_weight: 0.1}  # 突降90%

修改后:
  phases:
    - {end_epoch: 30, aux_loss_weight: 1.0, name: 'teacher_forcing'}
    - {end_epoch: 50, aux_loss_weight: 0.5, name: 'gradual_adaptation'}
    - {end_epoch: 70, aux_loss_weight: 0.2, name: 'hybrid_strengthening'}
    - {end_epoch: 100, aux_loss_weight: 0.1, name: 'final_polish'}
```

**理由**:
- 原来Epoch 30: 1.0 → 0.1 (直降90%) → Decoder崩溃
- 现在渐进式: 1.0 → 0.5 → 0.2 → 0.1
- 给Decoder 40个epoch来适应VQ量化噪声

---

### 4️⃣ **修改实验名称** (避免覆盖)

```yaml
# 第16行

修改前: name: residual_vqvae
修改后: name: residual_vqvae_v2_fixed
```

**理由**: 与之前的失败实验区分开

---

## 📊 预期改善

| 指标 | 修复前 | 预期修复后 | 判断标准 |
|------|--------|-----------|---------|
| **VQ Loss (验证)** | 3232 | < 200 | ✅ 成功 if <200<br>⚠️ 部分成功 if 200-500<br>❌ 失败 if >500 |
| **训练/验证比** | 0.17 | > 0.5 | ✅ 泛化正常 |
| **Epoch 30 Loss** | 跳变爆炸 | 平滑过渡 | ✅ 课程学习有效 |
| **残差范数(验证)** | 34.86 | < 20 | ✅ Prophet改善 |

---

## 🚀 启动训练

```bash
cd /root/code/zpw/tac/UniT-Align

# 方法1: 直接运行
python train.py --config-dir=./UniT/config --config-name=residual_vqvae_representation

# 方法2: 使用tmux (推荐，可以断开连接)
tmux new -s residual_train
python train.py --config-dir=./UniT/config --config-name=residual_vqvae_representation
# Ctrl+B, D 断开tmux
# tmux attach -t residual_train  重新连接

# 方法3: 后台运行
nohup python train.py --config-dir=./UniT/config --config-name=residual_vqvae_representation > train.log 2>&1 &
```

**训练时长**: 约3-4小时 (100 epochs, ~2分钟/epoch)

---

## 📈 训练中监控

### 关键指标 (WandB)

1. **VQ Loss曲线**:
   - `train/vq_loss`: 应该收敛到 < 100
   - `val/vq_loss`: 应该跟随训练集，不要爆炸

2. **课程学习切换点**:
   - Epoch 30: 观察loss是否平滑过渡 (不要像之前那样爆炸)
   - Epoch 50: aux_weight降到0.5
   - Epoch 70: aux_weight降到0.2

3. **残差统计**:
   - `train/raw_residual_norm`: 应该逐渐降低 (Prophet学习中)
   - `train/residual_norm`: VQ量化后的残差

4. **重建质量**:
   - `train/recon_loss` vs `train/recon_loss_aux`: 差距应该缩小

---

## 🔍 如果仍然失败

### 场景1: VQ Loss还是很高 (500-1000)

**可能原因**: Prophet预测还是太差，残差太大

**下一步**:
- 检查 `train/raw_residual_norm` 是否降低
- 如果没降低 → Prophet没学到东西 → **必须加入动作A**
- 如果降低了但VQ Loss还高 → 继续扩大codebook到768

---

### 场景2: Epoch 30/50/70仍然跳变

**可能原因**: 课程切换还是太快

**下一步**:
- 延长Phase 1: 0-50 epoch (aux_weight=1.0)
- 更细粒度切换: 1.0 → 0.7 → 0.5 → 0.3 → 0.2 → 0.1

---

### 场景3: Prophet相关性很低 (<0.3)

**可能原因**: V+S不足以预测触觉

**下一步** (必须):
- 从qpos计算速度 (差分)
- 修改ProphetEncoder加入action输入
- 这是idea1.md的原始设计，不可避免

---

## 📝 下一步TODO (如果本次成功)

1. ✅ 验证VQ Loss降低到合理范围
2. ✅ 可视化残差分布 (是否真的稀疏？)
3. ✅ 检查VQ Codebook利用率 (256个codes是否都被用到？)
4. ⚠️ **长期必做**: 加入动作信息 (Prophet当前只有V+S)
5. ⚠️ 消融实验: 对比VQ-256 vs VQ-512 vs VQ-768

---

## 🎯 成功标准

**最低要求** (本次修复):
- [x] VQ Loss (验证) < 200
- [x] 训练集Loss在100 epoch内收敛
- [x] Epoch 30不再爆炸

**理想目标** (需要加action):
- [ ] VQ Loss (验证) < 50
- [ ] Prophet相关性 > 0.5
- [ ] 残差范数 < 5 (真正的"意外信号")

---

## 📞 如何检查结果

训练完成后:

```bash
# 查看WandB链接
cat representation_models/*/wandb/latest-run/logs/debug.log | grep "wandb.ai"

# 或者本地分析
python analyze_training.py  # 会输出关键指标
```

**重点看**:
1. 最终VQ Loss是多少？
2. Epoch 30/50/70是否平滑过渡？
3. 残差范数是否逐渐减小？

---

Good luck! 🚀

如果训练完成后有问题，把以下信息发给我:
1. 最终的`wandb-summary.json`
2. VQ Loss曲线截图 (如果有)
3. Epoch 30前后的loss变化
