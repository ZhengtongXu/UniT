# AlignTac Implementation Guide

本文档说明AlignTac（Memory-Augmented VQ-VAE）的实现和使用方法。

## 📋 实现概览

AlignTac已成功实现，所有核心组件均已完成：

### ✅ 已实现组件

#### 1. MD-Map生成工具
- **文件**: `/mlp_vepfs/share/zpw/tac/script/md_map_visualizer.py` (已增强)
- **功能**: 
  - 可视化MD-Map（HSV编码）
  - 保存所有帧为.npz格式（`--save-all-frames`）
  - 用于生成训练所需的GT数据

#### 2. 批量GT生成脚本
- **文件**: `UniT/scripts/generate_mdmap_gt.py`
- **功能**: 
  - 遍历数据集批量生成MD-Map GT
  - 支持并行处理和断点续传
  - 输出格式：(T, 240, 320, 3) numpy数组

#### 3. Memory模块
- **文件**: `UniT/model/aligntac/memory_module.py`
- **架构**: FIFO队列 + Cross-Attention
- **功能**: 
  - 维护K帧历史窗口
  - 时序信息聚合
  - 支持训练时的梯度传播

#### 4. MD-Map解码器
- **文件**: `UniT/model/aligntac/mdmap_decoder.py`
- **功能**: 
  - 复用UniT的Decoder架构
  - 输出3通道MD-Map（dx, dy, energy）
  - 与RGB解码器并行工作

#### 5. MemoryVQVAE主模型
- **文件**: `UniT/model/aligntac/memory_vqvae.py`
- **架构**: 
  ```
  Input → Shared Encoder
      ↓
      ├─ Proj_Geo → Z_geo → Quantize_Geo → Decoder_RGB → RGB重建
      │
      └─ Proj_Dyn → Z_dyn → Memory → Z_dyn_mem → Quantize_Dyn → Decoder_MDMap → MD-Map预测
  ```
- **特性**:
  - 双流架构（几何+动力学解耦）
  - Memory增强的时序建模
  - 双目标学习（RGB重建 + MD-Map预测）

#### 6. MDMap数据集
- **文件**: `UniT/dataset/mdmap_dataset.py`
- **功能**: 
  - 加载触觉图像（zarr格式）
  - 加载MD-Map GT（.npz格式）
  - 支持数据增强和train/val划分

#### 7. 训练配置
- **文件**: `UniT/config/memory_vqvae_representation.yaml`
- **包含**: 
  - 模型超参数（embedding维度、codebook大小、Memory参数）
  - 数据集路径和预处理参数
  - 训练参数（batch size、learning rate、loss权重）

#### 8. 训练工作空间
- **文件**: `UniT/workspace/train_memory_vqvae_workspace.py`
- **功能**: 
  - PyTorch Lightning训练管道
  - 自动checkpoint管理
  - Wandb日志集成

---

## 🚀 使用流程

### Phase 0: 准备MD-Map GT数据

#### Step 1: 生成单个视频的MD-Map（测试）

```bash
cd /mlp_vepfs/share/zpw/tac/script

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mdmap

python md_map_visualizer.py \
    --input tac_data/test/peg_in_hole1/164327/v1/gelsight/gelsight_left_rgb.mp4 \
    --output output/test_mdmap.mp4 \
    --save-all-frames \
    --no-concat \
    --window-size 5 \
    --norm-coeff 0.8
```

输出：
- `output/test_mdmap.mp4` - 可视化视频
- `output/test_mdmap_mdmap_gt.npz` - GT数据（形状: T×240×320×3）

#### Step 2: 批量生成所有episode的MD-Map GT

```bash
cd /mlp_vepfs/share/zpw/tac/UniT-Align

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mdmap

python UniT/scripts/generate_mdmap_gt.py \
    --data-root data/allen_key_gelsight \
    --output-root data_mdmap_gt/allen_key_gelsight \
    --K 5 \
    --norm-coeff 0.8
```

输出结构：
```
data_mdmap_gt/allen_key_gelsight/
├── episode_001/
│   └── mdmap_gt.npz
├── episode_002/
│   └── mdmap_gt.npz
└── ...
```

### Phase 1: 训练MemoryVQVAE

#### Step 1: 修改配置文件

编辑 `UniT/config/memory_vqvae_representation.yaml`:

```yaml
dataset:
  data_root: data/allen_key_gelsight          # 触觉图像路径
  mdmap_root: data_mdmap_gt/allen_key_gelsight  # MD-Map GT路径
  
model:
  memory_K: 5           # 历史窗口大小
  lambda_rgb: 1.0       # RGB重建权重
  lambda_mdmap: 1.0     # MD-Map预测权重
```

#### Step 2: 启动训练

```bash
cd /mlp_vepfs/share/zpw/tac/UniT-Align

# 激活UniT环境（需要pytorch-lightning）
conda activate unitenv

# 开始训练
python UniT/workspace/train_memory_vqvae_workspace.py
```

训练输出：
- 模型checkpoints: `representation_models/{timestamp}_memory_vqvae_/checkpoints/`
- 日志: Wandb (项目名: "aligntac")

#### Step 3: 监控训练

训练会记录以下指标：
- `train/rgb_loss` - RGB重建损失
- `train/mdmap_loss` - MD-Map预测损失
- `train/quant_loss_geo` - 几何分支量化损失
- `train/quant_loss_dyn` - 动力学分支量化损失
- `train/total_loss` - 总损失

### Phase 2: 消融实验（可选）

测试Memory模块的作用：

#### 实验1: 不使用Memory（Baseline）

修改 `memory_vqvae.py` 中的Memory模块：
```python
# 在 forward() 中注释掉Memory
# z_dyn_mem = self.memory(z_dyn)
z_dyn_mem = z_dyn  # 直接使用Z_dyn，不经过Memory
```

预期结果：
- Ch1-2（dx, dy）预测正常
- Ch3（累积能量）预测失败（无法积分）

#### 实验2: 不同Memory窗口大小K

在配置文件中测试不同K值：
```yaml
model:
  memory_K: 3   # 短期记忆
  memory_K: 5   # 中期记忆（推荐）
  memory_K: 10  # 长期记忆
```

---

## 📊 关键参数说明

### MD-Map生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--window-size (-K)` | 5 | 历史窗口大小（帧数） |
| `--norm-coeff` | 0.8 | 归一化系数（0.6-1.0） |
| `--sigma` | auto | 高斯核标准差（auto=0.5×marker_spacing） |

### 模型架构参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `geo_embed_dim` | 3 | 几何分支embedding维度 |
| `dyn_embed_dim` | 3 | 动力学分支embedding维度 |
| `n_embed` | 1024 | Codebook大小 |
| `memory_K` | 5 | Memory窗口大小 |
| `memory_num_heads` | 1 | Attention头数 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `base_learning_rate` | 4.5e-6 | 基础学习率 |
| `batch_size` | 16 | 批大小 |
| `lambda_rgb` | 1.0 | RGB重建权重 |
| `lambda_mdmap` | 1.0 | MD-Map预测权重 |

---

## 🔍 验证模型输出

### 检查MD-Map预测质量

训练完成后，可视化预测的MD-Map：

```python
import torch
import numpy as np
from UniT.model.aligntac.memory_vqvae import MemoryVQVAE

# 加载模型
model = MemoryVQVAE.load_from_checkpoint("path/to/checkpoint.ckpt")
model.eval()

# 推理
with torch.no_grad():
    outputs = model(image_batch)
    mdmap_pred = outputs['mdmap_pred']  # (B, 3, H, W)
    
    # 检查Ch3（能量通道）是否有合理的非零值
    energy_channel = mdmap_pred[:, 2, :, :]
    print(f"Energy range: [{energy_channel.min():.3f}, {energy_channel.max():.3f}]")
```

预期：
- Ch1-2（dx, dy）：有正负值，表示位移方向
- Ch3（energy）：非负值，表示累积能量（需要Memory才能正确预测）

---

## 📁 文件结构总结

```
/mlp_vepfs/share/zpw/tac/
├── script/
│   ├── md_map_visualizer.py                    # MD-Map可视化工具（已增强）
│   └── modules/video_processor.py              # 支持save_all_frames
│
└── UniT-Align/
    ├── UniT/
    │   ├── model/aligntac/                     # AlignTac模型代码
    │   │   ├── __init__.py
    │   │   ├── memory_module.py                # Memory模块
    │   │   ├── mdmap_decoder.py                # MD-Map解码器
    │   │   └── memory_vqvae.py                 # 主模型
    │   │
    │   ├── dataset/
    │   │   └── mdmap_dataset.py                # MD-Map数据集
    │   │
    │   ├── config/
    │   │   └── memory_vqvae_representation.yaml # 训练配置
    │   │
    │   ├── workspace/
    │   │   └── train_memory_vqvae_workspace.py # 训练工作空间
    │   │
    │   └── scripts/
    │       └── generate_mdmap_gt.py            # 批量GT生成
    │
    └── ALIGNTAC_IMPLEMENTATION.md              # 本文档
```

---

## ⚠️ 已知限制和后续工作

### 当前限制
1. **环境依赖**: `pytorch-lightning==1.0.8` 未安装在unitenv中（需要重新创建环境或手动安装）
2. **分辨率不匹配**: MDMapDecoder输出64×80而非240×320（训练时会自动resize，但可能需要调整decoder层数）
3. **内存消耗**: 所有MD-Map GT加载到内存（可改为lazy loading）

### 后续优化
1. **动态分辨率**: 调整decoder的`ch_mult`参数以输出240×320
2. **lazy loading**: 修改MDMapDataset支持大规模数据集
3. **multi-GPU训练**: 配置DDP for distributed training
4. **Policy集成**: 将trained MemoryVQVAE集成到DiffusionUnitPolicy中

---

## 🎯 总结

AlignTac的完整实现已经完成，包括：

**✅ 核心创新**:
- Memory Module实现时序信息聚合（区分"静止成功" vs "振荡卡住"）
- 几何-动力学双流解耦（RGB重建 + MD-Map预测）
- MD-Map作为密集表征（相比稀疏光流更适合网络学习）

**✅ 工程实现**:
- 模块化设计（易于调试和消融实验）
- 完整训练流程（数据准备 → 训练 → 验证）
- 配置文件管理（Hydra支持快速实验迭代）

**🚀 下一步**:
1. 安装pytorch-lightning: `conda install pytorch-lightning==1.0.8`
2. 生成MD-Map GT数据（使用batch script）
3. 启动训练并监控Wandb指标
4. 运行消融实验验证Memory的作用

祝训练顺利！ 🎉
