# Residual Tactile Representation Learning - 实现文档

## 📋 概述

本文档描述了基于idea1.md的**Residual Tactile Representation Learning**架构的完整实现。

**核心思想**: 通过视觉+运动预测建立物理预期,让触觉编码器专注于"违反预期的接触异常"(残差信号),实现更强的异常检测能力。

## 🏗️ 架构设计

```
Prophet Stream: (Visual + Robot_State) → z_exp (预期触觉latent)
Observation Stream: Tactile_Image → z_real (真实触觉latent)
Residual: r_t = z_real - z_exp (意外信号)
Reconstruction: Decoder(z_exp + r_t) → T_recon
Loss: L_recon + λ·||r_t||₁ (稀疏约束)
```

**关键特性**:
- ✅ 双流架构: Prophet预测 + Observation编码
- ✅ Latent空间残差: r_t = z_real - z_exp
- ✅ L1稀疏约束: 鼓励残差稀疏化
- ✅ 零修改现有代码: 完全通过新文件实现

## 📁 文件清单

### 新建文件 (7个)

| 文件 | 路径 | 功能 |
|------|------|------|
| 1️⃣ **数据转换** | `UniT/dataset/convert_residual_data.py` | 多模态数据对齐 + Zarr转换 |
| 2️⃣ **数据集类** | `UniT/dataset/residual_tactile_dataset.py` | PyTorch Dataset加载器 |
| 3️⃣ **模型定义** | `UniT/taming/models/residual_vqmodel.py` | ProphetEncoder + ResidualVQModel |
| 4️⃣ **训练配置** | `UniT/config/residual_vqvae_representation.yaml` | Hydra超参数配置 |
| 5️⃣ **训练脚本** | `UniT/workspace/train_residual_vqvae_workspace.py` | PyTorch Lightning训练流程 |
| 6️⃣ **特征提取** | `UniT/model/residual_feature_extractor.py` | 下游Policy集成API |
| 7️⃣ **快速开始** | `UniT/scripts/train_residual_quickstart.sh` | 一键训练脚本 |

## 🚀 快速开始

### 步骤1: 环境准备

```bash
# 激活conda环境
conda activate unitenv

# 验证依赖
python -c "import torch; import pytorch_lightning; import zarr; print('✓ 依赖完整')"
```

### 步骤2: 数据转换

```bash
# 方式1: 使用快速脚本 (推荐)
bash UniT/scripts/train_residual_quickstart.sh

# 方式2: 手动运行
python UniT/dataset/convert_residual_data.py \
    --input_dir /mlp_vepfs/share/zpw/tac/script/tac_data/test/peg_in_hole1 \
    --output_zarr data/residual_tactile_zarr \
    --visual_size 224 224 \
    --tactile_size 128 160
```

**输出结构**:
```
data/residual_tactile_zarr/
├── data/
│   ├── tactile_image (N, 128, 160, 3) uint8
│   ├── visual_image (N, 224, 224, 3) uint8
│   └── robot_state (N, 8) float32
└── meta/
    ├── episode_id (N,) int32
    └── timestamp (N,) float64
```

### 步骤3: 训练模型

```bash
# 阶段1: 低稀疏惩罚 (λ=0.001)
python UniT/workspace/train_residual_vqvae_workspace.py \
    model.lambda_sparse=0.001 \
    batch_size=16 \
    max_epochs=50

# 阶段2: 中等惩罚 (λ=0.01)
python UniT/workspace/train_residual_vqvae_workspace.py \
    model.lambda_sparse=0.01 \
    max_epochs=100 \
    trainer.resume=representation_models/xxx/checkpoints/last.ckpt

# 阶段3: 高稀疏惩罚 (λ=0.05)
python UniT/workspace/train_residual_vqvae_workspace.py \
    model.lambda_sparse=0.05 \
    max_epochs=150 \
    trainer.resume=representation_models/xxx/checkpoints/last.ckpt
```

### 步骤4: 特征提取 (用于Policy)

```python
from UniT.model.residual_feature_extractor import ResidualFeatureExtractor

# 加载模型
extractor = ResidualFeatureExtractor(
    ckpt_path='representation_models/xxx/checkpoints/last.ckpt'
)

# 提取特征
z_fused, residual = extractor.extract_features(
    tactile_img=tactile,      # (B, 3, 128, 160)
    visual_img=visual,        # (B, 3, 224, 224)
    robot_state=robot_state   # (B, 8)
)

# 池化为1D向量
fused_feat, residual_feat = extractor.to_1d_features(z_fused, residual)
# 输出: (B, 512), (B, 512)
```

## 🔧 关键配置参数

### 超参数调优

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `lambda_sparse` | 0.01 | 0.001 - 0.1 | 稀疏Loss权重,越大越稀疏 |
| `learning_rate` | 4.5e-6 | 1e-6 - 1e-5 | 学习率 |
| `batch_size` | 16 | 8 - 32 | 受GPU内存限制 |
| `disc_start` | 10000 | 5k - 20k | GAN启动步数 |

### 课程学习策略

```yaml
# 阶段1: 预热 (0-20k steps)
lambda_sparse: 0.001
# 目标: Prophet学习基础视-动相关性

# 阶段2: 稀疏化 (20k-70k steps)
lambda_sparse: 0.01
# 目标: 迫使残差只编码"意外"

# 阶段3: 微调 (70k-100k steps)
lambda_sparse: 0.05
# 目标: 最大化稀疏性
```

## 📊 监控指标

### 必须监控的指标

```python
# 重建质量
'val/recon_loss'       # 应 < 0.02 (与VQModel持平)

# 稀疏性
'train/residual_norm'  # 应显著小于 z_real_norm
'train/residual_max'   # 不应爆炸

# 分支平衡
'train/z_exp_norm'     # Prophet输出稳定性
'train/z_real_norm'    # Observation输出稳定性
```

### Wandb可视化

训练过程中会自动记录到Wandb:
- 项目名: `residual_tactile`
- 可视化: 触觉重建、residual激活图、latent heatmap

## 🧪 测试与验证

### 单元测试

```bash
# 测试ProphetEncoder
python UniT/taming/models/residual_vqmodel.py

# 测试Dataset
python UniT/dataset/residual_tactile_dataset.py
```

### 性能指标

**定量指标**:
```python
# 重建质量
PSNR: > 25 dB
SSIM: > 0.85
LPIPS: < 0.1

# 稀疏性
Residual L0 norm (非零元素比例): < 30%
Residual L1 norm: < 0.5 × z_real L1 norm
```

**定性分析**:
1. 可视化residual激活图
   - 正常插入: residual_norm应该很低
   - 卡死碰撞: residual_norm应该激增 (>10×)

2. t-SNE聚类分析
   - 提取不同场景的residual
   - 异常事件应形成明显cluster

## 🔌 集成到下游Policy

### 方式1: 直接替换触觉编码器

```python
# 在DiffusionUnitPolicy中
from UniT.model.residual_feature_extractor import ResidualTactileEncoder

class DiffusionUnitPolicy:
    def __init__(self, ..., residual_ckpt=None):
        # 原有编码器
        self.visual_encoder = ResNet34()

        # 替换触觉编码器
        self.tactile_encoder = ResidualTactileEncoder(
            residual_ckpt=residual_ckpt,
            use_residual=True,
            fusion_mode='concat'  # 'concat'/'add'/'residual_only'
        )

    def encode_obs(self, obs_dict):
        visual_feat = self.visual_encoder(obs_dict['camera_0'])

        # 使用Residual编码器
        tactile_feat = self.tactile_encoder(
            tactile=obs_dict['tactile'],
            visual=obs_dict['camera_0'],
            robot_state=obs_dict['robot_state']
        )

        # 融合
        obs_feat = torch.cat([visual_feat, tactile_feat], dim=-1)
        return obs_feat
```

### 方式2: 添加Residual分支

```python
# 保留原有触觉编码,添加residual作为额外特征
obs_feat = torch.cat([
    visual_feat,      # (B, 512)
    tactile_feat,     # (B, 512) - 原有VQModel
    residual_feat     # (B, 512) - 新增Residual
], dim=-1)
# 总维度: (B, 1536)
```

## ⚠️ 常见问题与解决

### 问题1: Residual崩塌 (全为0)

**症状**: `train/residual_norm` 接近0

**原因**: `lambda_sparse` 过大

**解决**:
```bash
# 降低稀疏惩罚
python train.py model.lambda_sparse=0.0001
```

### 问题2: Prophet过拟合触觉

**症状**: `train/residual_norm` 很大,但重建很差

**原因**: Prophet记忆了触觉模式

**解决**:
```yaml
# 增加Dropout
prophet_config:
  hidden_dims: [256, 128, 64]  # 减少容量

# 或添加数据增强
```

### 问题3: 时间戳对齐误差大

**症状**: 转换时报告 `visual_max_error > 0.1s`

**解决**:
```python
# 在convert_residual_data.py中添加容错
# 允许±1帧的误差
```

### 问题4: GPU内存不足

**解决**:
```yaml
# 减小batch size
batch_size: 8

# 或使用混合精度
trainer:
  precision: 16
```

## 📈 预期结果

### 成功标准

**最低标准 (MVP)**:
- ✅ 重建loss < 0.03
- ✅ residual_norm < 1.0
- ✅ 代码无bug,支持checkpoint恢复

**理想标准**:
- ✅ 重建loss < 0.02
- ✅ residual_norm < 0.5
- ✅ 卡死时residual激活 > 5× 正常
- ✅ 下游policy性能提升 > 10%

**优秀标准**:
- ✅ 重建loss < 0.015
- ✅ residual L0 norm < 20%
- ✅ 异常检测AUC > 0.9
- ✅ 下游policy性能提升 > 20%

## 📚 参考资料

- **原始提案**: `/root/code/zpw/tac/UniT-Align/UniT/idea1.md`
- **实现计划**: `/root/.claude/plans/dapper-dreaming-hearth.md`
- **UniT论文**: VQ-VAE触觉表征基础
- **AlignTac**: 双分支架构参考 (`memory_vqvae.py`)

## 🙋 获取帮助

如有问题,请参考:
1. 计划文件中的"潜在问题与解决方案"章节
2. Wandb日志中的指标趋势
3. 测试代码中的示例用法

## 🎯 下一步行动

1. **立即开始**: 运行快速脚本
   ```bash
   bash UniT/scripts/train_residual_quickstart.sh
   ```

2. **监控训练**: 查看Wandb
   ```
   https://wandb.ai/your-project/residual_tactile
   ```

3. **调优超参**: 根据residual_norm调整lambda_sparse

4. **集成验证**: 在Policy中测试性能提升

---

**实现日期**: 2025-12-22
**状态**: ✅ 所有核心文件已完成
**准备就绪**: 可开始数据转换和训练
