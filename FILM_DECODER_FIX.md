# DynamicFiLMDecoder 多层FiLM调制改进方案

## 当前问题

当前实现中，StyleExpander只输出一组FiLM参数 (gamma_seq, beta_seq: B×T×128)，只能用于第一层 (128 channels)。
后续层 (64, 32 channels) 没有使用FiLM调制，限制了Event Code的表达能力。

## 解决方案

### 方案1: 简单方案 - 投影复用
**优点**: 最小改动
**缺点**: 表达能力有限

```python
# 在DynamicFiLMDecoder.forward中:
gamma_seq_128, beta_seq_128 = self.style_expander(event_code)  # (B, T, 128)

# Layer 1
x = self.film1(x, gamma_seq_128, beta_seq_128, force_identity)
x = self.conv1(x)  # (B, 128, T, H, W)

# Layer 2: 使用projection降维
gamma_seq_64 = gamma_seq_128[..., :64]  # 简单截取前64维
beta_seq_64 = beta_seq_128[..., :64]
x = self.conv2(x)  # (B, 64, T, 2H, 2W)
x = self.film2(x, gamma_seq_64, beta_seq_64, force_identity)

# Layer 3
gamma_seq_32 = gamma_seq_128[..., :32]
beta_seq_32 = beta_seq_128[..., :32]
x = self.conv3(x)  # (B, 32, T, 4H, 4W)
x = self.film3(x, gamma_seq_32, beta_seq_32, force_identity)
```

### 方案2: 标准方案 - 多头投影 (推荐)
**优点**: 更强表达能力，每层有独立的FiLM参数
**缺点**: 需要修改StyleExpander

**修改StyleExpander**:
```python
class StyleExpander(nn.Module):
    def __init__(self, event_dim, film_dims=[128, 64, 32], seq_length=16, hidden_dim=256):
        super().__init__()
        self.seq_length = seq_length
        self.film_dims = film_dims
        
        self.gru = nn.GRU(event_dim, hidden_dim, num_layers=2, batch_first=True)
        
        # 为每层创建独立投影
        self.gamma_heads = nn.ModuleList([nn.Linear(hidden_dim, dim) for dim in film_dims])
        self.beta_heads = nn.ModuleList([nn.Linear(hidden_dim, dim) for dim in film_dims])
        
        # Identity初始化
        for gamma_head, beta_head in zip(self.gamma_heads, self.beta_heads):
            nn.init.constant_(gamma_head.bias, 0.0)
            nn.init.constant_(beta_head.bias, 0.0)
            nn.init.xavier_uniform_(gamma_head.weight, gain=0.01)
            nn.init.xavier_uniform_(beta_head.weight, gain=0.01)
    
    def forward(self, event_code):
        B = event_code.shape[0]
        event_seq = event_code.unsqueeze(1).expand(B, self.seq_length, -1)
        gru_out, _ = self.gru(event_seq)  # (B, T, hidden_dim)
        
        # 为每层生成FiLM参数
        film_params = []
        for gamma_head, beta_head in zip(self.gamma_heads, self.beta_heads):
            gamma_seq = gamma_head(gru_out) + 1.0
            beta_seq = beta_head(gru_out)
            film_params.append((gamma_seq, beta_seq))
        
        return film_params  # [(B,T,128), (B,T,64), (B,T,32)]
```

**修改DynamicFiLMDecoder**:
```python
class DynamicFiLMDecoder(nn.Module):
    def __init__(self, ...):
        # ...
        self.style_expander = StyleExpander(
            event_dim=event_dim,
            film_dims=[128, 64, 32],  # 对应3个FiLM层
            seq_length=seq_length
        )
        # ...
    
    def forward(self, z_pred_seq, event_code, force_identity=False):
        # ...
        # 获取多组FiLM参数
        film_params = self.style_expander(event_code)  # [(γ,β)×3]
        gamma_128, beta_128 = film_params[0]  # (B, T, 128)
        gamma_64, beta_64 = film_params[1]    # (B, T, 64)
        gamma_32, beta_32 = film_params[2]    # (B, T, 32)
        
        x = z_spatial  # (B, 128, T, H, W)
        
        # Layer 1: FiLM + Conv
        x = self.film1(x, gamma_128, beta_128, force_identity)
        x = self.conv1(x)  # (B, 128, T, H, W)
        
        # Layer 2: Conv + FiLM
        x = self.conv2(x)  # (B, 64, T, 2H, 2W)
        x = self.film2(x, gamma_64, beta_64, force_identity)
        
        # Layer 3: Conv + FiLM
        x = self.conv3(x)  # (B, 32, T, 4H, 4W)
        x = self.film3(x, gamma_32, beta_32, force_identity)
        
        # Layer 4: Conv out (no FiLM)
        recon_video = self.conv_out(x)  # (B, 3, T, 8H, 8W)
        
        return recon_video
```

## 建议

**当前情况考虑**:
1. 由于架构已经基本完成，且DataLoader、训练流程都已测试通过
2. 多层FiLM是一个"改进"而非"修复"
3. 可以先用当前版本训练，观察单层FiLM的效果

**建议做法**:
- **阶段1**: 先用当前简化版本训练 (只用第一层FiLM)
- **阶段2**: 如果发现VQ路径效果不理想，再升级为多层FiLM

**快速测试多层FiLM**:
如果想立即测试，使用**方案1 (简单投影复用)**，只需修改DynamicFiLMDecoder.forward，无需改StyleExpander。

