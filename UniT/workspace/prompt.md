
**系统设定:** 你是一位精通 PyTorch、VAE 和机器人表征学习（Representation Learning）的资深算法工程师。

**任务目标:** 我需要重构现有的 "Prophet-Residual" VAE 架构。目前的版本是**确定性 (Deterministic)** 的，通过直接相减计算残差。我想将其升级为**概率性 (Probabilistic)** 架构，引入不确定性（Aleatoric Uncertainty）估计，以实现对传感器噪声的自适应归一化。

**当前架构 (Current Architecture):**

1. **Prophet (Stream A):** 输入视觉+状态 → 预测确定性的触觉隐变量 zpred​。
    
2. **Observer (Stream B):** 输入真实触觉 → 编码为 zreal​。
    
3. **Residual:** 直接计算 r=zreal​−zpred​。
    
4. **VQ:** 对 r 进行量化。
    
5. **Decoder:** 输入 zpred​+VQ(r) 进行重建。
    
6. **Loss:** MSE 重建 Loss + VQ Loss。
    

**我想要的修改逻辑 (The New Logic):** 核心思路是实现 **"基于精度加权的残差 (Precision-Weighted Residuals)"**。

**1. 修改 Prophet (概率化):**

- Prophet Encoder 的输出层不再是一个向量，而是输出高斯分布的参数：均值 μ 和 对数方差 log(σ2)（为了数值稳定性）。
    
- 维度应与 zreal​ 保持一致。
    

**2. 修改残差计算方式 (自适应归一化):**

- 在送入 VQ 之前，不再做简单的减法，而是进行 **Z-score 归一化**。
    
- **公式:** zinput_to_vq​=σzreal​−μ​
    
- **物理含义:** 利用预测出的不确定度 σ 对残差进行缩放。在摩擦大（噪声大、不确定度高）的区域，σ 变大，从而抑制无关的噪声残差；在碰撞（意外）发生时，σ 较小，残差会被放大。
    

**3. 修改 VQ 模块:**

- VQ 现在接收的是归一化后的 zinput_to_vq​。
    
- 此时 Codebook 学习的是“标准差倍数”（Standardized Surprise），例如 1倍 σ 或 3倍 σ 的事件。
    

**4. 修改 Decoder 输入 (反归一化):**

- **关键点:** 在解码重建时，必须把量化后的残差还原回去。
    
- **公式:** Decoder Input = μ+(Quantized(zinput_to_vq​)×σ)。
    

**5. 修改 Loss 函数:**

- **Reconstruction Loss:** 保持不变 。
    
- **Prophet Training Loss:** 为了让 Prophet 学会预测不确定度，需要增加 **Gaussian Negative Log-Likelihood (NLL) Loss**。
    
    - 目标是最大化 zreal​ 在预测分布 N(μ,σ) 下的似然概率。这会迫使网络在误差大（很难预测准）的地方自动调大 σ 以降低 Loss。
