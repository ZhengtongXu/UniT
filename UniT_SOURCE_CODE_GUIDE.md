# UniT源码学习指导文档

## 目录
1. [项目概览](#项目概览)
2. [整体架构](#整体架构)
3. [核心模块详解](#核心模块详解)
4. [数据流程](#数据流程)
5. [训练流程](#训练流程)
6. [关键设计特点](#关键设计特点)

---

## 项目概览

**UniT (Universal Tactile Representation)** 是一个结合**触觉感知**和**视觉感知**的机器人操控策略学习框架。核心思想是：
- 使用VQ-VAE (Vector Quantized Variational AutoEncoder)学习触觉信号的紧凑表征
- 将触觉表征与视觉特征融合，用于Diffusion Policy的条件生成
- 支持多种感知模型(VQ-VAE, MAE, BYOL, T3, ResNet)进行触觉理解

**应用场景**: 双臂机器人操作任务(如chicken legs hanging, peg-in-hole等)，需要精确的触觉反馈。

---

## 整体架构

```
UniT/
├── config/              # 配置文件(YAML)：定义训练参数、模型架构
├── dataset/             # 数据集加载与预处理
├── model/               # 模型组件(视觉编码器、触觉工具)
├── perception/          # 触觉感知模型(VQ-VAE/MAE/BYOL/T3等)
├── policy/              # 策略网络(Diffusion-based)
├── taming/              # VQ-VAE/VQ-GAN实现(改编自taming-transformers)
├── utils/               # 工具类(触觉数据记录器)
└── workspace/           # 训练工作空间(训练循环、日志、检查点)
```

### 架构层次关系
```
Workspace (训练循环) 
    ↓
Policy (策略网络: DiffusionUnitPolicy)
    ↓
├─ Obs Encoder (TimmObsEncoder: 视觉特征)
└─ VQ-VAE (VQModel: 触觉特征) → ConvPoolingHead → 触觉嵌入
    ↓
Conditional Unet1D (生成动作序列)
```

---

## 核心模块详解

### 1. **config/ - 配置文件**

**功能**: 使用Hydra框架，通过YAML文件定义所有超参数和模型配置。

**关键配置文件**:
- `train_legs_unit_policy.yaml`: UniT策略训练配置
  - 定义数据集路径、观测模态(相机+触觉)
  - 配置VQ-VAE预训练模型路径
  - 设置Diffusion Policy参数(噪声调度、U-Net结构)
  
- `vqvae_representation_key.yaml`: VQ-VAE表征学习配置
  - VQ-VAE架构参数(编码器/解码器通道数、残差块数)
  - 损失函数配置(感知损失+判别器)

- `key_vqvae_perception.yaml`: 触觉感知任务配置
  - 使用预训练VQ-VAE作为特征提取器
  - 添加下游任务头(姿态估计/分类)

**特点**:
- 使用`_target_`指定类的完整路径，支持Hydra动态实例化
- 支持多任务配置(3D姿态、6D姿态、分类)

---

### 2. **dataset/ - 数据加载**

#### 2.1 `aloha_dataset.py` - 主数据集类

**核心类**: `AlohaDataset(BaseImageDataset)`

**功能**: 
- 加载机器人演示数据(zarr格式)
- 处理多模态观测:
  - RGB相机(cam_low, cam_high, wrist相机)
  - 触觉图像(tactile_left_image, tactile_right_image)
  - 关节位置(qpos)
- 返回observation dict + action序列

**关键方法**:
```python
__getitem__(idx):
    1. 通过SequenceSampler采样时序数据
    2. 对图像resize到目标尺寸
    3. 归一化到[0,1]并转换为CHW格式
    4. 返回 {'obs': {相机+触觉+qpos}, 'action': 动作序列}
```

**数据流**:
```
zarr文件 → ReplayBuffer → SequenceSampler → AlohaDataset → Dataloader
```

#### 2.2 `tactile_representation_dataset.py` - 触觉表征数据集

**用途**: 专门用于训练VQ-VAE的触觉重建任务

**特点**:
- 只返回触觉图像(用于自监督学习)
- 支持多种触觉模态(image/depth/shear)

#### 2.3 `real_data_conversion.py` - 数据格式转换

**功能**: 将原始机器人数据(zarr)转换为ReplayBuffer格式

---

### 3. **perception/ - 触觉感知模型**

这是UniT的核心创新点，提供多种触觉编码器。

#### 3.1 `vqvae_perception.py` - VQ-VAE感知模型

**核心类**: `VqvaePerception(nn.Module)`

**架构**:
```python
触觉图像 (B, H, W, C)
    ↓
VQ-VAE Encoder (冻结)
    ↓
Latent Code (B, 3, 28, 28)
    ↓
ConvPoolingHead (3层卷积+GroupNorm+SE注意力)
    ↓
MLP Head
    ↓
输出 (3D姿态/6D姿态/分类)
```

**特点**:
- **冻结VQ-VAE编码器**: 使用预训练的触觉表征
- **ConvPoolingHead**: 
  - 3层卷积逐步提取特征(128→256→512通道)
  - 使用GroupNorm替代BatchNorm(小batch友好)
  - SE(Squeeze-and-Excitation)注意力模块增强特征
- **多任务支持**:
  - `3Dpose`: 输出4D四元数(旋转)
  - `6Dpose`: 输出3D位置+4D四元数
  - `classification`: 输出类别概率

**损失函数**:
- 位置误差: MSE Loss × 1000
- 旋转误差: Quaternion Angle Loss (四元数夹角)

#### 3.2 `mae_perception.py` - MAE感知模型

**核心**: 使用Masked AutoEncoder (MAE)作为视觉Transformer编码器

**特点**:
- ViT-based架构，使用patch embedding
- 适合大规模触觉图像的自监督预训练

#### 3.3 `t3_perception.py` - T3感知模型

**核心**: 使用T3 (Tactile Transformer)预训练模型

**架构**:
```python
Encoder (mini) → Shared Trunk → MLP Head
```

#### 3.4 `byol_perception.py` & `resnet_perception.py`

- **BYOL**: 自监督对比学习
- **ResNet**: 基于ImageNet预训练的CNN

---

### 4. **policy/ - 策略网络**

#### 4.1 `diffusion_unit_policy.py` - UniT策略 (核心!)

**核心类**: `DiffusionUnitPolicy(BaseImagePolicy)`

**架构流程**:
```
观测输入:
  - RGB图像 → TimmObsEncoder → 视觉特征 (B, D_vis)
  - 触觉图像 → VQ-VAE.to_latent() → Latent (B*T, 3, 16, 20)
                ↓
            ConvPoolingPolicyHead → 触觉嵌入 (B*T, 512)
                ↓
            Reshape → (B, T*512)

全局条件 = concat(视觉特征, 左触觉, 右触觉)  # (B, D_vis+T*512*2)

动作生成:
  噪声轨迹 (B, horizon, action_dim)
      ↓
  Conditional Unet1D(轨迹, 时间步t, global_cond)
      ↓
  DDIM Scheduler逐步去噪 (num_inference_steps步)
      ↓
  预测动作序列 (B, horizon, action_dim)
```

**关键创新**:
1. **触觉-视觉融合**: 
   - 视觉: ResNet34提取全局场景特征
   - 触觉: VQ-VAE提取局部接触信息
   - 融合: 简单的特征拼接(concatenation)

2. **ConvPoolingPolicyHead**:
   ```python
   3层卷积(128→256→512) + LayerNorm + LeakyReLU
   → AdaptiveAvgPool → Flatten → Linear(512)
   ```
   - 将VQ-VAE的latent code压缩成固定维度向量
   - 使用SE注意力捕捉通道间相关性

3. **Diffusion过程**:
   - 训练: 对动作序列加噪，预测噪声(epsilon预测)
   - 推理: 从随机噪声逐步去噪生成动作
   - 支持输入扰动(input_pertub)缓解曝光偏差

**训练流程**:
```python
compute_loss(batch):
    1. 归一化观测和动作
    2. 编码视觉+触觉特征 → global_cond
    3. 采样随机时间步t和噪声ε
    4. 对动作加噪: noisy_action = scheduler.add_noise(action, ε, t)
    5. 预测噪声: pred_ε = model(noisy_action, t, global_cond)
    6. 计算MSE损失: loss = ||pred_ε - ε||²
```

**推理流程**:
```python
predict_action(obs_dict):
    1. 编码观测 → global_cond
    2. 初始化随机轨迹
    3. 循环num_inference_steps次:
        - 预测噪声
        - 使用DDIM scheduler去噪
    4. 返回去噪后的动作序列
```

#### 4.2 `diffusion_unet_timm_policy.py` - 纯视觉策略

**区别**: 不使用触觉，仅用RGB图像

---

### 5. **model/ - 模型组件**

#### 5.1 `model/vision/timm_obs_encoder.py` - 视觉编码器

**核心类**: `TimmObsEncoder(ModuleAttrMixin)`

**功能**: 统一处理多相机+低维观测

**特点**:
- **多模型支持**: 通过timm库支持ResNet/ConvNeXt/ViT
- **特征聚合策略** (`feature_aggregation`):
  - `attention_pool_2d`: 注意力池化(类似CLIP)
  - `spatial_embedding`: 可学习的空间嵌入
  - `transformer`: Transformer encoder聚合特征图
  - `avg/max`: 简单池化
- **多相机处理**:
  ```python
  for 每个相机:
      提取特征 → 聚合 → Flatten → Concat
  Concat(所有相机特征, qpos) → 全局特征向量
  ```

#### 5.2 `model/tactile/utils.py` - 触觉工具

**关键组件**:

1. **ConvPoolingHead**: 触觉特征提取头
   ```python
   Conv(3→128) → GroupNorm → LeakyReLU → SE → MaxPool
   Conv(128→256) → GroupNorm → LeakyReLU → SE → MaxPool
   Conv(256→512) → GroupNorm → LeakyReLU → SE → MaxPool
   → Flatten
   ```

2. **quaternion_angle_loss**: 四元数角度损失
   ```python
   angle = 2 * arccos(|q1 · q2|)  # 旋转角度差
   ```
   - 归一化四元数
   - 计算点积(考虑四元数双重覆盖)
   - 返回旋转角度差的均值

3. **MlpHead**: 通用MLP头
   ```python
   Linear(in→256) → ReLU → Dropout(0.2)
   Linear(256→128) → ReLU → Dropout(0.2)
   Linear(128→out)
   ```

---

### 6. **taming/ - VQ-VAE实现**

改编自[Taming Transformers](https://github.com/CompVis/taming-transformers)项目。

#### 6.1 `taming/models/vqgan.py` - VQ-VAE模型

**核心类**: `VQModel(pl.LightningModule)`

**架构**:
```
Encoder: 卷积下采样 (128→128→256→512)
    ↓
quant_conv: 将特征映射到codebook维度
    ↓
VectorQuantizer: 向量量化
    ↓
post_quant_conv: 恢复通道数
    ↓
Decoder: 卷积上采样 (512→256→128→3)
```

**关键方法**:
```python
to_latent(x): 
    # 提取VQ-VAE的连续latent(不量化)
    h = encoder(x)
    h = quant_conv(h)
    return h  # (B, embed_dim, H', W')
```
- **为什么不量化?**: 保留连续特征，避免量化误差，适合下游任务

**训练**:
- 重建损失(L1/L2)
- 感知损失(LPIPS)
- 判别器损失(GAN)
- Codebook损失(向量量化承诺损失)

#### 6.2 `taming/modules/vqvae/quantize.py` - 向量量化

**核心**: `VectorQuantizer2` (修复了VectorQuantizer的bug)

**量化过程**:
```python
1. 将特征展平: z (B,C,H,W) → (B*H*W, C)
2. 计算与codebook的距离: d = ||z - e||²
3. 找最近的code: min_indices = argmin(d)
4. 查表获取量化值: z_q = embedding[min_indices]
5. 反向传播: 使用straight-through estimator
```

---

### 7. **workspace/ - 训练工作空间**

#### 7.1 `train_diffusion_unet_image_workspace.py` - 策略训练

**核心类**: `TrainDiffusionUnetImageWorkspace(BaseWorkspace)`

**训练循环**:
```python
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # 前向传播
        loss = model.compute_loss(batch)
        
        # 梯度累积
        loss.backward()
        if step % gradient_accumulate_every == 0:
            optimizer.step()
            lr_scheduler.step()
        
        # EMA更新
        ema.step(model)
        
        # 验证和保存
        if step % val_every == 0:
            validate()
        if step % checkpoint_every == 0:
            save_checkpoint()
```

**特点**:
- 集成Wandb日志
- TopK检查点管理
- EMA (Exponential Moving Average)模型
- 支持resume训练

#### 7.2 `train_vqvae_representation_workspace.py` - VQ-VAE训练

**基于PyTorch Lightning**:
```python
Trainer.fit(VQModel, TactileDataset)
```
- 自动处理多GPU训练
- 定期保存检查点
- 支持测试集评估

---

## 数据流程

### 完整训练数据流
```
1. 数据加载:
   zarr文件 → real_data_conversion.py
        ↓
   ReplayBuffer (内存缓存)
        ↓
   SequenceSampler (时序采样)
        ↓
   AlohaDataset.__getitem__()
        ↓
   DataLoader (batch组装)

2. 观测编码:
   batch['obs'] = {
       'cam_low': (B, T, 3, 224, 224),
       'tactile_left_image': (B, T, 3, 128, 160),
       'qpos': (B, T, 14)
   }
        ↓
   TimmObsEncoder: RGB → 视觉特征 (B, D_vis)
   VQ-VAE: 触觉 → Latent (B*T, 3, 16, 20) → 触觉嵌入 (B, T*512)
        ↓
   global_cond = concat(视觉, 触觉)  # (B, D_total)

3. 动作生成:
   batch['action']: (B, horizon, 14)
        ↓
   加噪: noisy_action = add_noise(action, ε, t)
        ↓
   Conditional Unet1D(noisy_action, t, global_cond)
        ↓
   预测噪声: pred_ε
        ↓
   损失: MSE(pred_ε, ε)
```

### 推理数据流
```
实时观测 (相机+触觉+qpos)
    ↓
编码器 → global_cond
    ↓
随机噪声 → DDIM采样(16步) → 动作序列
    ↓
执行前n_action_steps个动作
    ↓
获取新观测，重复
```

---

## 训练流程

### 阶段1: VQ-VAE预训练

**目标**: 学习触觉信号的紧凑表征

```bash
# 配置文件: vqvae_representation_key.yaml
python workspace/train_vqvae_representation_workspace.py

# 训练任务: 重建触觉图像
输入: 触觉图像 (128, 160, 3)
输出: 重建图像 (128, 160, 3)
损失: L1 + LPIPS + GAN + VQ-loss
```

**检查点**: 保存在`representation_models/YYYY-MM-DDTHH-MM-SS-vq/`

### 阶段2: 触觉感知任务(可选)

**目标**: 验证VQ-VAE特征的质量

```bash
# 配置文件: key_vqvae_perception.yaml
python workspace/train_vqvae_perception_workspace.py

# 下游任务:
# - 3D姿态估计(旋转)
# - 6D姿态估计(位置+旋转)
# - 物体分类
```

### 阶段3: UniT策略训练

**目标**: 学习视触觉融合的操控策略

```bash
# 配置文件: train_legs_unit_policy.yaml
python workspace/train_diffusion_unet_image_workspace.py

# 关键配置:
vq_model_config:
    ckpt_path: "path/to/vqvae_checkpoint.ckpt"  # 加载预训练VQ-VAE
    
# 训练:
输入: 多相机RGB + 双手触觉 + qpos
输出: 14维动作序列(双臂7自由度)
```

---

## 关键设计特点

### 1. **模块化设计**

- **感知与策略解耦**: 可独立更换触觉编码器(VQ-VAE/MAE/T3)
- **配置驱动**: 所有超参数通过YAML管理，便于实验
- **多任务支持**: 同一框架支持姿态估计、分类、策略学习

### 2. **VQ-VAE的关键作用**

**为什么使用VQ-VAE?**
- 触觉传感器数据高维(128×160×3)且噪声大
- VQ-VAE学习离散码本(1024个code)，实现数据压缩
- Latent code (3×16×20)包含语义信息(接触位置、力度)

**与策略的连接**:
- 不使用量化后的离散code(会丢失信息)
- 直接使用`to_latent()`的连续特征
- 通过ConvPoolingHead进一步压缩成向量

### 3. **Diffusion Policy的优势**

- **多模态输出**: 可以表示复杂的动作分布(而非单一动作)
- **长期规划**: 一次生成整个horizon的动作序列
- **训练稳定**: 相比GAN，Diffusion模型更容易训练

### 4. **特征融合策略**

**简单拼接 vs 复杂融合**:
- UniT使用简单的concat融合
- 优点: 简单高效，让模型自己学习交互
- 可改进方向: Cross-attention、FiLM调制

### 5. **效率优化**

- **冻结VQ-VAE**: 减少参数量，加速训练
- **共享RGB模型**: `share_rgb_model=False`时每个相机独立编码器
- **EMA模型**: 提升推理稳定性

---

## 学习路径建议

### 入门路径
1. **理解数据**: 
   - 查看`dataset/aloha_dataset.py`的`__getitem__`
   - 打印一个batch的shape和数值范围

2. **运行VQ-VAE**:
   - 训练一个小VQ-VAE(减小数据集)
   - 可视化重建结果

3. **理解Diffusion**:
   - 阅读`policy/diffusion_unit_policy.py`的`compute_loss`
   - 理解加噪-去噪过程

### 进阶路径
1. **修改感知模块**:
   - 替换VQ-VAE为MAE/BYOL
   - 对比不同编码器的效果

2. **改进融合方式**:
   - 实现Cross-attention融合
   - 添加时序建模(LSTM/Transformer)

3. **消融实验**:
   - 训练纯视觉策略(无触觉)
   - 训练纯触觉策略(无RGB)
   - 对比融合策略的增益

### 调试技巧
1. **可视化latent code**:
   ```python
   latent = vqgan.to_latent(tactile_image)
   plt.imshow(latent[0, 0].cpu())  # 查看第一个通道
   ```

2. **检查特征维度**:
   ```python
   global_cond = obs_encoder(obs_dict)
   print(global_cond.shape)  # 应该是(B, feature_dim)
   ```

3. **监控损失曲线**:
   - Wandb日志查看train_loss/val_loss
   - 检查是否过拟合

---

## 配置文件解读

### train_legs_unit_policy.yaml 关键参数

```yaml
# 数据
horizon: 16              # 动作序列长度
n_obs_steps: 2           # 观测历史长度
n_action_steps: 8        # 实际执行的动作数

# VQ-VAE
vq_model_config:
  ckpt_path: "..."       # 必须指定预训练模型
  embed_dim: 3           # Codebook维度
  n_embed: 1024          # Codebook大小
  z_channels: 3          # Latent通道数

# 观测编码器
obs_encoder:
  model_name: 'resnet34' # ResNet backbone
  frozen: False          # 不冻结(fine-tune)
  feature_aggregation: 'attention_pool_2d'  # 特征聚合方式
  downsample_ratio: 32   # 下采样比例(7×7特征图)

# Diffusion
noise_scheduler:
  num_train_timesteps: 75   # 训练时间步数
  beta_schedule: 'squaredcos_cap_v2'  # 噪声调度
  prediction_type: 'epsilon'  # 预测噪声

num_inference_steps: 16    # 推理去噪步数(越多越慢)

# U-Net
diffusion_step_embed_dim: 128  # 时间嵌入维度
down_dims: [256, 512, 1024]    # U-Net下采样通道
kernel_size: 5
n_groups: 8                    # GroupNorm分组数

# 训练
batch_size: 128
lr: 1.0e-4
num_epochs: 500
use_ema: True           # 使用EMA模型
gradient_accumulate_every: 1
```

---

## 常见问题

### Q1: 为什么VQ-VAE使用`to_latent`而不是`encode`?
**A**: 
- `encode`: 返回量化后的离散code(会丢失信息)
- `to_latent`: 返回量化前的连续特征(保留更多信息)
- 策略学习需要精确的触觉特征，所以用连续的

### Q2: 触觉特征维度是如何确定的?
**A**:
```python
输入: (B, T, 3, 128, 160)
VQ-VAE Encoder: 4层下采样(×2,×2,×2,×2) → (B*T, 3, 128/16, 160/16) = (B*T, 3, 8, 10)
实际配置中latent_shape=[3,16,20]: 可能是不同的分辨率设置
ConvPoolingHead: 3层卷积+池化 → (B*T, 512)
Reshape: (B, T*512)  # T=2时, 1024维
```

### Q3: 如何调整模型大小?
**A**:
- **减小VQ-VAE**: 减少`ch`和`ch_mult`
- **减小Obs Encoder**: 使用`resnet18`替代`resnet34`
- **减小U-Net**: 减小`down_dims`(如[128,256,512])

### Q4: 训练需要多少数据?
**A**:
- VQ-VAE预训练: 建议>1000 episodes(触觉图像)
- 策略训练: 50-200 episodes(取决于任务复杂度)

---

## 代码示例

### 示例1: 加载预训练VQ-VAE

```python
from UniT.taming.models.vqgan import VQModel

# 定义配置
vq_config = {
    'embed_dim': 3,
    'n_embed': 1024,
    'ddconfig': {
        'z_channels': 3,
        'resolution': 128,
        'in_channels': 3,
        'out_ch': 3,
        'ch': 128,
        'ch_mult': [1,1,2,4],
        'num_res_blocks': 2,
        'attn_resolutions': [16],
    },
    'lossconfig': {...}  # 推理时不需要
}

# 加载模型
vqgan = VQModel(**vq_config)
checkpoint = torch.load('path/to/checkpoint.ckpt')
vqgan.load_state_dict(checkpoint['state_dict'], strict=False)
vqgan.eval()

# 提取触觉特征
tactile_image = torch.randn(1, 128, 160, 3)  # (B, H, W, C)
tactile_image = tactile_image.permute(0, 3, 1, 2)  # → (B, C, H, W)
latent = vqgan.to_latent(tactile_image)
print(latent.shape)  # (1, 3, 8, 10)
```

### 示例2: 可视化Diffusion去噪过程

```python
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy

# 加载策略
policy = DiffusionUnetImagePolicy.from_checkpoint('checkpoint.ckpt')
policy.eval()

# 推理
obs_dict = {...}  # 观测数据
trajectories = []

# 修改conditional_sample以记录中间步骤
with torch.no_grad():
    for t in range(num_inference_steps):
        trajectory = policy.conditional_sample(...)
        trajectories.append(trajectory.cpu().numpy())

# 可视化去噪过程
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, len(trajectories), figsize=(20, 3))
for i, traj in enumerate(trajectories):
    axes[i].plot(traj[0, :, 0])  # 第一个动作维度
    axes[i].set_title(f'Step {i}')
plt.show()
```

### 示例3: 自定义触觉编码器

```python
class CustomTactileEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义你的编码器
        self.cnn = nn.Sequential(...)
        
    def forward(self, tactile_image):
        # tactile_image: (B, T, C, H, W)
        B, T = tactile_image.shape[:2]
        x = tactile_image.reshape(B*T, *tactile_image.shape[2:])
        features = self.cnn(x)  # (B*T, D)
        return features.reshape(B, -1)  # (B, T*D)

# 在DiffusionUnitPolicy中替换VQ-VAE
class CustomDiffusionPolicy(DiffusionUnitPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 替换VQ-VAE相关组件
        delattr(self, 'vqgan')
        delattr(self, 'cp_head_left')
        delattr(self, 'cp_head_right')
        
        self.tactile_encoder = CustomTactileEncoder()
```

---

## 参考资源

### 论文
- **Diffusion Policy**: [Paper](https://diffusion-policy.cs.columbia.edu/)
- **VQ-VAE**: Oord et al. "Neural Discrete Representation Learning"
- **Taming Transformers**: Esser et al. "Taming Transformers for High-Resolution Image Synthesis"

### 代码库
- Diffusion Policy: https://github.com/real-stanford/diffusion_policy
- Taming Transformers: https://github.com/CompVis/taming-transformers
- Timm: https://github.com/huggingface/pytorch-image-models

---

## 总结

UniT是一个设计精巧的触觉-视觉融合框架:
1. **两阶段训练**: VQ-VAE预训练 → 策略学习
2. **多模态融合**: 简单拼接RGB特征和触觉特征
3. **Diffusion建模**: 生成式策略，适合复杂任务
4. **模块化设计**: 易于扩展和实验

**核心创新点**:
- 使用VQ-VAE学习紧凑的触觉表征
- 将离散表征(VQ-VAE)与连续策略(Diffusion)结合
- 支持多种感知模型的即插即用

**适用场景**:
- 需要精细触觉反馈的操作任务
- 双臂协同操作
- 接触丰富的任务(如组装、悬挂)

祝学习顺利! 🚀
