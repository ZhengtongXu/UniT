# Spatiotemporal架构Shape问题分析

## 发现的问题

### 1. ❌ Dataloader Shape不匹配

**当前返回** (单个样本):
```python
{
    'tactile_video': (3, T, H, W),      # ✓ 正确
    'visual_context': (3, H_v, W_v),    # ✓ 正确
    'state_seq': (T, state_dim),        # ✗ 缺少batch维度
    'action_seq': (T, action_dim)       # ✗ 缺少batch维度
}
```

**模型期望** (batched):
```python
{
    'tactile_video': (B, 3, T, H, W),
    'visual_context': (B, 3, H_v, W_v),
    'state_seq': (B, T, state_dim),     # ← 需要batch维度
    'action_seq': (B, T, action_dim)    # ← 需要batch维度
}
```

**问题**: DataLoader的collate_fn会自动添加batch维度，但单样本测试时会出错。

**修复方案**: 无需修改，collate_fn会自动处理。但需要确认batch后的shape。

---

### 2. ⚠️ DynamicFiLMDecoder只在第一层使用FiLM

**当前实现**:
```python
# Layer 1: FiLM + Conv
x = self.film1(x, gamma_seq, beta_seq, force_identity)  # (B, 128, T, H, W)
x = self.conv1(x)  # (B, 128, T, H, W)

# Layer 2-4: 只有Conv，没有FiLM调制
x = self.conv2(x)  # (B, 64, T, 2H, 2W)
x = self.conv3(x)  # (B, 32, T, 4H, 4W)
recon_video = self.conv_out(x)  # (B, 3, T, 8H, 8W)
```

**问题**: 
- 注释说"为简化只在第一层使用FiLM"
- 但这样Event Code的影响力有限，可能无法充分调制解码过程

**改进方案**:
需要为每层创建对应channel数的FiLM参数：
```python
# StyleExpander应输出多组FiLM参数
gamma_seq_128, beta_seq_128 = ..  # For layer1 (128 channels)
gamma_seq_64, beta_seq_64 = ..    # For layer2 (64 channels)
gamma_seq_32, beta_seq_32 = ..    # For layer3 (32 channels)
```

---

### 3. ✓ Shape流动验证

**Decoder shape flow**:
```
输入: z_pred_seq (B, T, 256)
  ↓ Linear projection
  (B, T, 128*16*20) → reshape → (B, T, 128, 16, 20) → permute → (B, 128, T, 16, 20)
  ↓ Conv1 (128→128, no upsample)
  (B, 128, T, 16, 20)
  ↓ Conv2 (128→64, ×2 upsample)
  (B, 64, T, 32, 40)
  ↓ Conv3 (64→32, ×2 upsample)
  (B, 32, T, 64, 80)
  ↓ Conv4 (32→3, ×2 upsample)
  (B, 3, T, 128, 160) ← 正确! 匹配tactile_size
```

**结论**: Decoder的upsample设计正确。

---

### 4. ✓ TimeDistributed实现验证

**当前实现**:
```python
class TimeDistributed(nn.Module):
    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x_reshape = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)  # (B*T, C, H, W)
        out = self.module(x_reshape)  # Apply 2D module
        _, C_out, H_out, W_out = out.shape
        out = out.view(B, T, C_out, H_out, W_out).permute(0, 2, 1, 3, 4)  # (B, C_out, T, H_out, W_out)
        return out
```

**结论**: 实现正确，能正确处理video tensor。

---

## 建议修复优先级

### 高优先级
1. **改进DynamicFiLMDecoder**: 在多层使用FiLM调制
   - 当前只在第一层使用，限制了Event Code的表达能力
   - 应该在至少前2-3层都使用FiLM

### 中优先级
2. **验证DataLoader的batch collation**: 
   - 确认collate_fn正确处理state_seq/action_seq
   - 添加batch测试

### 低优先级  
3. **StyleExpander的初始化策略**: 
   - 当前γ→1, β→0是合理的
   - 但可能需要更激进的初始化来加速收敛

