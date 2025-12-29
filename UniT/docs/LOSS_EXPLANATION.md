# Residual VQ-VAE Loss Functions Explained

## 核心损失

### 1. total_loss (总损失)
- **公式**: `L_recon + λ * L_sparse`
- **含义**: 训练目标，结合重建质量和残差稀疏性
- **期望趋势**: 持续下降，最终收敛到0.3-0.5
- **监控**: 进度条实时显示，wandb记录epoch平均值

### 2. recon_loss (重建损失)
- **公式**: `nll_loss + d_weight * disc_factor * g_loss`
- **含义**: 图像重建质量，包含像素、感知、对抗损失
- **期望趋势**: 快速下降后平稳
- **组成**: 由nll_loss（像素+感知）和g_loss（对抗）加权组合

### 3. sparse_loss (稀疏损失)
- **公式**: `λ * ||r||₁`
- **含义**: 约束残差向量稀疏，避免过拟合
- **期望趋势**: 随λ增大而增大（课程学习）
- **λ调度**: 0.001 (预热) → 0.01 (稀疏化) → 0.05 (微调)

### 4. residual_norm (残差范数)
- **公式**: `mean(|r|)`
- **含义**: 平均残差大小，反映预测误差
- **期望值**: 0.1-0.5（取决于λ）
- **相关指标**: residual_max, residual_min

---

## 细粒度组件

### 重建质量

#### nll_loss (负对数似然)
- **公式**: `rec_loss + perceptual_weight * p_loss`
- **含义**: 主要重建损失，结合像素和感知
- **权重**: pixelloss_weight=1.0, perceptual_weight=1.0
- **位置**: VQLPIPSWithDiscriminator.forward()

#### rec_loss (像素损失)
- **公式**: `|inputs - reconstructions|` (L1)
- **含义**: 逐像素差异，衡量重建精度
- **特点**: 简单直接，但可能导致模糊

#### p_loss (感知损失, LPIPS)
- **公式**: VGG16特征空间距离
- **含义**: 基于人眼感知的相似性
- **作用**: 保留纹理细节，防止模糊
- **实现**: 使用预训练VGG16提取多层特征

### GAN组件

#### g_loss (生成器损失)
- **公式**: `-mean(D(recon))` (hinge loss)
- **含义**: 迫使判别器认为重建是真实的
- **启用**: global_step > 10000 (disc_start)
- **特点**: 增强重建真实感，可能引入伪影

#### disc_loss (判别器损失)
- **公式**: 区分真实和重建图像的hinge loss
- **含义**: 训练判别器的鉴别能力
- **平衡**: 与g_loss保持动态平衡

#### d_weight (自适应权重)
- **公式**: `||∇nll|| / ||∇g|| * disc_weight`
- **含义**: 动态平衡重建和对抗目标
- **范围**: [0, 1e4]
- **作用**: 防止GAN训练不稳定

### 其他

#### quant_loss (量化损失)
- **状态**: 禁用 (codebook_weight=0.0)
- **原因**: Residual VQ-VAE不使用向量量化
- **备注**: 原VQ-VAE中用于学习codebook

#### disc_factor
- **含义**: 判别器启用标志
- **值**: 0 (前10k步), 1 (之后)
- **作用**: 前期专注重建质量，后期加入对抗训练

#### logits_real / logits_fake
- **含义**: 判别器对真实/生成图像的输出
- **监控**: 用于判断GAN训练平衡性
- **理想值**: logits_real > 0, logits_fake < 0

---

## Loss计算流程

### Generator (optimizer_idx=0)

```
1. Forward pass
   tactile_image → obs_encoder → z_real
   visual_image + robot_state → prophet_encoder → z_exp
   residual = z_real - z_exp
   z_fused = z_exp + residual
   z_fused → decoder → tactile_recon

2. Reconstruction Loss
   rec_loss = L1(tactile_image, tactile_recon)
   p_loss = LPIPS(tactile_image, tactile_recon)
   nll_loss = rec_loss + perceptual_weight * p_loss

3. GAN Loss (if global_step > disc_start)
   g_loss = -mean(discriminator(tactile_recon))
   d_weight = adaptive_weight(nll_loss, g_loss)

4. Total Loss
   recon_loss = nll_loss + d_weight * disc_factor * g_loss
   sparse_loss = lambda_sparse * mean(|residual|)
   total_loss = recon_loss + sparse_loss

5. Backward & Update
   total_loss.backward()
   optimizer_gen.step()
```

### Discriminator (optimizer_idx=1)

```
1. Discriminator Forward
   logits_real = discriminator(tactile_image)
   logits_fake = discriminator(tactile_recon.detach())

2. Hinge Loss
   disc_loss = mean(max(0, 1 - logits_real)) +
               mean(max(0, 1 + logits_fake))

3. Backward & Update
   disc_loss.backward()
   optimizer_disc.step()
```

---

## 监控建议

### 正常训练特征

**Loss下降曲线**:
- `total_loss`: 1.0+ → 0.3-0.5 (100 epochs)
- `recon_loss`: 0.8+ → 0.2-0.3
- `sparse_loss`: 随λ阶梯式增长 (0.001 → 0.01 → 0.05)
- `residual_norm`: 逐渐收敛到稳定值

**GAN平衡性**:
- `g_loss` vs `disc_loss`: 保持平衡（1:1左右）
- `d_weight`: 在[0.1, 10]范围内动态调整
- `logits_real`: 稳定在正值（~0.5-2）
- `logits_fake`: 稳定在负值（~-0.5 to -2）

**细粒度组件**:
- `nll_loss`: 主导前期训练，持续下降
- `p_loss`: 稳定在较小值（<0.5）
- `rec_loss`: 快速下降到稳定值

### 异常信号

**Loss异常**:
- ❌ `recon_loss`停止下降: 学习率过小或判别器过强
- ❌ `total_loss`震荡: 学习率过大或GAN不稳定
- ❌ `sparse_loss`爆炸: λ过大，残差被过度惩罚

**GAN崩溃**:
- ❌ `g_loss` >> `disc_loss` (>10倍): 判别器过弱
  - **解决**: 减小disc_weight或增加disc训练频率
- ❌ `g_loss` << `disc_loss` (<0.1倍): 判别器过强，生成器崩溃
  - **解决**: 增大disc_weight或减少disc训练频率
- ❌ `logits_real` < 0: 判别器无法识别真实图像
- ❌ `logits_fake` > 0: 判别器无法识别生成图像

**残差异常**:
- ❌ `residual_norm` > 1.0: λ过小，残差不够稀疏
- ❌ `residual_norm` < 0.01: λ过大，残差被过度抑制
- ❌ `residual_max` - `residual_min` 差距过大: 存在异常样本

---

## 配置参数

### 损失权重 (residual_vqvae_representation.yaml)

```yaml
model:
  lambda_sparse: 0.01              # 稀疏损失权重

lossconfig:
  disc_start: 10000                # 判别器启用步数
  disc_weight: 0.8                 # 判别器损失权重
  codebook_weight: 0.0             # VQ损失权重（禁用）
  pixelloss_weight: 1.0            # L1重建损失权重
  perceptual_weight: 1.0           # LPIPS感知损失权重
```

### 课程学习策略 (launch_residual_training.sh)

```bash
# 阶段1: 预热 (0-30 epochs)
lambda_sparse=0.001

# 阶段2: 稀疏化 (30-80 epochs)
lambda_sparse=0.01

# 阶段3: 微调 (80-100 epochs)
lambda_sparse=0.05
```

---

## 常见问题

### Q1: 为什么有这么多loss组件？

**A**: Residual VQ-VAE结合了多个训练目标：
- **重建质量**: nll_loss, rec_loss, p_loss（确保重建准确）
- **真实感**: g_loss, disc_loss（通过GAN增强纹理）
- **稀疏性**: sparse_loss（约束残差，防止过拟合）
- **平衡控制**: d_weight, disc_factor（动态调整各目标权重）

### Q2: quant_loss为什么是0？

**A**: Residual VQ-VAE不使用向量量化（VQ），直接学习连续latent。VQ会导致信息损失，不适合精细残差建模。

### Q3: 如何判断训练是否正常？

**A**: 关注4个核心指标：
1. `total_loss`持续下降
2. `recon_loss`收敛到稳定值
3. `residual_norm`在合理范围（0.1-0.5）
4. `g_loss`和`disc_loss`大致平衡

### Q4: 什么时候调整λ？

**A**: 遵循课程学习策略：
- **阶段1** (0-30 epochs): λ=0.001，专注重建质量
- **阶段2** (30-80 epochs): λ=0.01，加强稀疏约束
- **阶段3** (80-100 epochs): λ=0.05，进一步稀疏化

### Q5: GAN不稳定怎么办？

**A**: 检查以下指标：
- `d_weight`: 应在[0.1, 10]范围内，否则调整disc_weight配置
- `disc_start`: 确保判别器在前10k步未启用
- `logits_real/fake`: 检查判别器是否过强或过弱
- 考虑使用梯度裁剪（gradient_clip_val=1.0已配置）

---

## 参考

**相关文件**:
- 模型实现: `UniT/taming/models/residual_vqmodel.py`
- Loss定义: `UniT/taming/modules/losses/vqperceptual.py`
- LPIPS实现: `UniT/taming/modules/losses/lpips.py`
- 训练配置: `UniT/config/residual_vqvae_representation.yaml`

**论文**:
- VQ-VAE: "Neural Discrete Representation Learning" (van den Oord et al., 2017)
- LPIPS: "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (Zhang et al., 2018)
- Hinge GAN: "Spectral Normalization for Generative Adversarial Networks" (Miyato et al., 2018)
