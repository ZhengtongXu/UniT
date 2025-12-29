# Spatiotemporal Residual VQ-VAE 实现指南
## "Feeling the Unexpected" (感受那份意外)

---

## 📋 项目概述

本文档介绍了从**单帧 (Single-Frame)** Residual VQ-VAE 到**时空 (Spatiotemporal)** 架构的升级实现。

**核心理念**:
- Prophet预测"预期的平滑物理趋势"
- Residual捕捉"意外事件" (碰撞、打滑等)
- VQ将连续的意外信号压缩成**离散的事件码 (Event Code)**
- Dynamic FiLM Decoder使用事件码调制解码过程

---

## 🏗️ 架构设计

### 输入/输出维度

```python
# 输入
tactile_video: (B, C=3, T=16, H=128, W=160)  # 触觉视频片段
visual_context: (B, 3, H_v, W_v)             # 单帧视觉上下文
state_seq: (B, T, state_dim=8)               # 机器人状态序列
action_seq: (B, T, action_dim=8)             # 未来动作序列

# 输出
recon_video: (B, 3, T=16, H=128, W=160)      # 重建视频
event_code: (B, event_dim=256)               # 离散事件码 (one-hot)
```

### 核心组件

#### 1. **TimeDistributed Wrapper** ✅ 已实现

```python
class TimeDistributed(nn.Module):
    """
    将2D CNN应用到视频的每一帧

    优势: 节省显存，不使用3D CNN

    Input: (B, C, T, H, W)
    Output: (B, C_out, T, H_out, W_out)
    """
```

**用途**: 复用预训练的2D VAE Encoder处理视频

#### 2. **Prophet Stream (Causal TCN)** ✅ 已实现

```python
class ProphetEncoder(nn.Module):
    """
    预测预期的触觉序列

    Pipeline:
        Visual Context (single frame) → ResNet → Global Pool
        → Broadcast to T → Concat(visual, state, action)
        → Causal TCN → Z^_seq (B, T, latent_dim)

    Causal: 确保t时刻的预测只依赖 ≤t 的信息
    """
```

**关键特性**:
- 使用ResNet18/34提取视觉特征
- Causal TCN处理时序，膨胀卷积扩大感受野
- 输出预期触觉序列 Z^_seq

#### 3. **Residual Stream (Transformer Aggregator + VQ)** ✅ 已实现

```python
class ResidualTransformerAggregator(nn.Module):
    """
    将残差序列聚合为单个事件表征

    Pipeline:
        R_seq = Z_real_seq - Z^_seq (B, T, D)
        → Transformer Encoder (捕捉时序依赖)
        → Global Average Pool
        → (B, D) single event representation
        → VQ → Event Code (discrete)
    """
```

**物理意义**:
- Transformer捕捉整个16帧序列的整体动态
- Global Pool得到**单个事件表征** (例如"一次碰撞")
- VQ离散化为事件码 (如 code_42 = "右侧碰撞")

#### 4. **Dynamic FiLM Decoder (Style Expander)** ✅ 已实现

```python
class StyleExpander(nn.Module):
    """
    将静态Event Code展开为动态调制参数

    Pipeline:
        e_k (B, event_dim) [静态]
        → GRU (时序展开)
        → γ_seq, β_seq (B, T, film_dim) [动态]

    作用: 静态事件码 → 时变FiLM参数
    """

class DynamicFiLMDecoder(nn.Module):
    """
    使用Event Code动态调制解码

    Pipeline:
        Z^_seq (Content/Base) + e_k (Style)
        → Project Z^ to spatial-temporal
        → Style Expander → γ_seq, β_seq
        → FiLM Modulation at each decoder layer
        → Reconstructed Video

    FiLM: F'_t = γ_t ⊙ F_t + β_t
    """
```

**创新点**:
- **Content**: Prophet预测的平滑趋势 (Z^_seq)
- **Style**: VQ离散事件码 (e_k)
- **Dynamic Modulation**: GRU展开静态码为时变参数
- **force_identity**: 支持三阶段训练 (γ=1, β=0 → 不调制)

---

## ✅ 已实现的功能

### 核心模型 (`spatiotemporal_residual_vqmodel.py`)

1. ✅ `TimeDistributed` - 时空包装器
2. ✅ `CausalTCN` - 因果时序卷积
3. ✅ `ProphetEncoder` - Prophet流
4. ✅ `ResidualTransformerAggregator` - 残差聚合器
5. ✅ `StyleExpander` - 风格展开器
6. ✅ `FiLMLayer` - FiLM调制层
7. ✅ `DynamicFiLMDecoder` - 动态FiLM解码器
8. ✅ `SpatiotemporalResidualVQModel` - 主模型

### 测试验证

```bash
✓ Model initialization - 成功
✓ Forward pass - 成功
✓ Output shapes:
  - recon_video: (2, 3, 16, 128, 160) ✓
  - event_code: (2, 256) ✓
  - vq_loss: 0.2201 ✓
```

---

## 🚧 待完成的工作

### 1. **TimeDistributed Observation Encoder** (高优先级)

**当前状态**: 使用placeholder (随机tensor)

**需要实现**:

```python
class TimeDistributedObsEncoder(nn.Module):
    """
    TimeDistributed包装预训练的2D VAE Encoder

    Input: (B, 3, T, H, W)
    Output: (B, T, latent_dim)

    步骤:
    1. 加载预训练的2D VAE Encoder (from residual_vqmodel.py)
    2. 使用TimeDistributed包装
    3. 添加temporal projection层
    """
    def __init__(self, pretrained_ckpt: str, latent_dim: int):
        super().__init__()

        # 1. 加载预训练encoder
        from UniT.taming.modules.diffusionmodules.model import Encoder
        self.encoder_2d = Encoder(...)  # Load from checkpoint

        # 2. TimeDistributed包装
        self.td_encoder = TimeDistributed(self.encoder_2d)

        # 3. Spatial → Temporal projection
        # (B, C, T, H, W) → (B, T, latent_dim)
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.to_temporal = nn.Linear(C, latent_dim)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # video: (B, 3, T, H, W)
        z_spatial = self.td_encoder(video)  # (B, C, T, H', W')
        z_pooled = self.spatial_pool(z_spatial.flatten(0, 1))  # (B*T, C, 1, 1)
        z_pooled = z_pooled.view(B, T, C)  # (B, T, C)
        z_temporal = self.to_temporal(z_pooled)  # (B, T, latent_dim)
        return z_temporal
```

**关键文件**:
- 预训练checkpoint: `representation_models/*/checkpoints/last.ckpt`
- Encoder定义: `UniT/taming/modules/diffusionmodules/model.py:458`

### 2. **数据集适配** (高优先级)

**需求**: 从单帧数据集升级为视频片段数据集

```python
class SpatiotemporalTactileDataset(Dataset):
    """
    时空触觉数据集

    返回:
    {
        'tactile_video': (3, T=16, H, W),     # 触觉视频片段
        'visual_context': (3, H_v, W_v),       # 单帧上下文
        'state_seq': (T, state_dim),           # 状态序列
        'action_seq': (T, action_dim)          # 动作序列
    }

    采样策略:
    - 从episode中采样长度为T的连续片段
    - 确保时序连贯性
    - Visual context可以使用第一帧或中间帧
    """
```

**参考**: `UniT/dataset/residual_tactile_dataset.py`

### 3. **配置文件** (中优先级)

```yaml
# UniT/config/spatiotemporal_residual_vqvae.yaml

name: spatiotemporal_residual_vqvae
seed: 42
base_learning_rate: 4.5e-6
batch_size: 8  # 视频数据占用更多显存
max_epochs: 150
gpus: "0"

model:
  # Prophet配置
  prophet_config:
    visual_backbone: resnet18
    state_dim: 8
    action_dim: 8
    latent_dim: 256
    seq_length: 16
    tcn_layers: 3

  # Observation Encoder配置
  obs_encoder_ckpt: "representation_models/best_model/checkpoints/last.ckpt"

  # Residual Aggregator配置
  residual_config:
    latent_dim: 256
    num_heads: 4
    num_layers: 2
    ff_dim: 512

  # VQ配置
  vq_config:
    n_e: 512      # Event codebook size
    e_dim: 256
    beta: 0.25

  # Decoder配置
  decoder_config:
    latent_dim: 256
    event_dim: 256
    seq_length: 16
    spatial_size: [16, 20]
    out_channels: 3

  lambda_sparse: 0.01

# 数据集配置
dataset:
  dataset_path: data/residual_tactile_train_v2
  seq_length: 16
  visual_size: [224, 224]
  tactile_size: [128, 160]
  val_ratio: 0.1

# 课程学习 (三阶段训练)
curriculum_learning:
  enabled: true
  phases:
    # Stage 1 (0-30): 只训练触觉重建 (force_identity=True)
    - end_epoch: 30
      force_identity: true
      aux_loss_weight: 1.0
      hybrid_loss_weight: 0.0
      name: 'stage1_recon_only'

    # Stage 2 (30-80): 引入VQ，训练Event Code
    - end_epoch: 80
      force_identity: false
      aux_loss_weight: 0.5
      hybrid_loss_weight: 0.5
      name: 'stage2_vq_training'

    # Stage 3 (80-150): FiLM调制，端到端fine-tune
    - end_epoch: 150
      force_identity: false
      aux_loss_weight: 0.0
      hybrid_loss_weight: 1.0
      name: 'stage3_film_finetune'
```

### 4. **训练Workspace** (中优先级)

```python
# UniT/workspace/train_spatiotemporal_workspace.py

class TrainSpatiotemporalWorkspace(BaseWorkspace):
    """
    时空Residual VQ-VAE训练工作空间

    特殊支持:
    - force_identity参数动态切换 (课程学习)
    - 视频可视化 (log_videos)
    - Event code分析 (codebook usage统计)
    """
```

### 5. **Loss函数完善** (低优先级)

**当前**: 只有VQ loss

**需要添加**:

```python
def training_step(self, batch, batch_idx):
    outputs = self(batch, force_identity=self.current_force_identity)

    # 1. 重建Loss
    recon_loss = F.mse_loss(outputs['recon_video'], batch['tactile_video'])

    # 2. VQ Commitment Loss
    vq_loss = outputs['vq_loss']

    # 3. Sparse Loss (鼓励residual稀疏)
    sparse_loss = self.lambda_sparse * outputs['residual_seq'].abs().mean()

    # 4. Perceptual Loss (可选, 提升视频质量)
    # perceptual_loss = self.perceptual(outputs['recon_video'], batch['tactile_video'])

    total_loss = recon_loss + vq_loss + sparse_loss

    # Log
    self.log('train/recon_loss', recon_loss)
    self.log('train/vq_loss', vq_loss)
    self.log('train/sparse_loss', sparse_loss)

    return total_loss
```

### 6. **可视化工具** (低优先级)

```python
@torch.no_grad()
def log_videos(self, batch):
    """
    记录视频可视化到WandB

    显示:
    1. 输入视频 vs 重建视频 (side-by-side)
    2. Prophet预测的Z^_seq可视化 (heatmap)
    3. Residual序列可视化 (时序演化)
    4. Event code分布 (histogram)
    5. FiLM参数演化 (γ, β over time)
    """
```

---

## 📊 三阶段训练策略

### **Stage 1 (0-30 epochs): 触觉重建**

```python
force_identity = True  # γ=1, β=0, 不使用VQ调制
aux_loss_weight = 1.0  # 只训练基础重建
```

**目标**: 训练Observation Encoder + Decoder基础重建能力

**Loss**: `L_recon(Z_real → Decoder → Video)`

### **Stage 2 (30-80 epochs): VQ训练**

```python
force_identity = False  # 开始使用VQ调制
hybrid_loss = 0.5 * L_recon(Z^ + VQ → Video)
aux_loss = 0.5 * L_recon(Z_real → Video)
```

**目标**: 训练Residual Aggregator + VQ学习事件码

**Loss**: `L_hybrid + L_aux + L_vq + L_sparse`

### **Stage 3 (80-150 epochs): FiLM Fine-tune**

```python
force_identity = False
hybrid_loss_weight = 1.0  # 完全依赖VQ路径
aux_loss_weight = 0.0
```

**目标**: 端到端fine-tune，FiLM调制达到最优

**Loss**: `L_hybrid + L_vq + L_sparse`

---

## 🔬 实验验证

### 定性评估

1. **视频重建质量**: 对比原始视频vs重建视频
2. **Event Code语义**: 分析不同事件的code分布
   - 碰撞 → code cluster A
   - 打滑 → code cluster B
   - 摩擦 → code cluster C
3. **FiLM调制效果**: 可视化γ, β的时序演化

### 定量指标

```python
# 1. 重建质量
PSNR = compute_psnr(recon, target)
SSIM = compute_ssim(recon, target)

# 2. Codebook利用率
active_codes = (usage > threshold).sum() / total_codes

# 3. 残差稀疏性
residual_sparsity = (residual.abs() < threshold).float().mean()

# 4. 事件码区分度
code_entropy = -sum(p_k * log(p_k))  # 越高越分散
```

---

## 📂 文件结构

```
UniT-Align/
├── UniT/
│   ├── taming/models/
│   │   ├── residual_vqmodel.py                  # 单帧版本
│   │   └── spatiotemporal_residual_vqmodel.py   # ✅ 时空版本
│   ├── dataset/
│   │   ├── residual_tactile_dataset.py          # 单帧数据集
│   │   └── spatiotemporal_tactile_dataset.py    # 🚧 待实现
│   ├── workspace/
│   │   ├── train_residual_vqvae_workspace.py    # 单帧训练
│   │   └── train_spatiotemporal_workspace.py    # 🚧 待实现
│   └── config/
│       ├── residual_vqvae_representation_v4_twostage.yaml
│       └── spatiotemporal_residual_vqvae.yaml   # 🚧 待实现
├── SPATIOTEMPORAL_IMPLEMENTATION_GUIDE.md       # 本文档
└── launch_spatiotemporal_training.sh            # 🚧 待实现
```

---

## 🎯 下一步行动

### 优先级排序

1. **P0 (必须)**: 实现TimeDistributed Observation Encoder
   - 从预训练checkpoint加载encoder
   - 测试forward pass

2. **P0 (必须)**: 适配数据集为视频片段
   - 修改Dataset返回T=16的序列
   - 测试dataloader

3. **P1 (重要)**: 创建配置文件和训练workspace
   - YAML配置
   - Workspace with curriculum learning

4. **P2 (可选)**: 完善loss函数和可视化
   - Perceptual loss
   - Video logging

---

## 💡 关键设计决策

### 为什么使用TimeDistributed而非3D CNN?

**优势**:
1. **显存友好**: 复用预训练2D模型，不需要从头训练3D
2. **迁移学习**: 利用已有的2D VAE checkpoint
3. **灵活性**: 可以轻松切换不同的2D backbone

**劣势**:
1. **时序建模弱**: 每帧独立编码，缺少时序交互
   - **解决方案**: 在Residual Aggregator使用Transformer捕捉时序

### 为什么将VQ放在序列聚合之后?

**原因**:
1. **事件级表征**: 整个16帧序列代表一个事件，用1个code表示
2. **Codebook效率**: 避免T=16倍的codebook浪费
3. **下游任务**: 事件码可直接用于事件检测/分类

### 为什么使用GRU展开Event Code?

**原因**:
1. **动态调制**: 静态码无法直接调制时序视频
2. **时序一致性**: GRU生成平滑的γ, β序列
3. **可解释性**: 中间hidden state可视化事件演化

---

## 📚 参考文献

### 相关工作

1. **FiLM**: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018
2. **Temporal Coherence**: Tulyakov et al., "MoCoGAN: Decomposing Motion and Content for Video Generation", CVPR 2018
3. **VQ-VAE**: van den Oord et al., "Neural Discrete Representation Learning", NeurIPS 2017

### 代码参考

- 单帧Residual VQ-VAE: `UniT/taming/models/residual_vqmodel.py`
- 2D VAE Encoder: `UniT/taming/modules/diffusionmodules/model.py`
- VQ Quantizer: `UniT/taming/modules/vqvae/quantize.py`

---

**作者**: Claude Code
**日期**: 2025-12-29
**分支**: spatiotemporal-residual-vae
**状态**: 核心架构已实现 ✅, 数据和训练待完成 🚧
