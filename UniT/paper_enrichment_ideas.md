# Feeling the Unexpected - 论文丰富方案
*基于当前Residual VQ-VAE工作的补充实验建议*

---

## 🎯 核心问题
虽然故事很好，但需要更多**实证证据**来支撑"残差=意外信号"的核心Claim。

---

## 📋 建议的5个研究方向

### **方向1: 残差信号的可解释性分析** 🔥 **最重要，必须做**

**核心问题**: 证明 $r_t = z_{real} - z_{exp}$ 真的捕获了"物理意外"

#### 1.1 时间序列可视化 (Temporal Residual Activation)
**实验设计**:
- 收集一个完整的插孔episode（从接近→接触→插入→卡住→成功）
- 逐帧计算：
  - `||r_t||` (残差L2范数) - 整体意外强度
  - `max(|r_t|)` (残差最大值) - 局部突变
- 在时间轴上绘制残差曲线，标注关键事件：
  - `t=0-10`: 空中移动 (预期: ||r_t|| ≈ 0)
  - `t=15`: 初次接触 (预期: 小峰值)
  - `t=30`: 卡住 (Jamming) (预期: **大峰值**)
  - `t=45`: 成功插入 (预期: 峰值消失)

**预期结果**:
```
||r_t||
  ^
  |     ┌──┐              ← Jamming spike!
  |     │  │
  |  ┌┐ │  │  ┌┐
  |  ││ │  │  ││
  |──┴┴─┴──┴──┴┴─────→ time
     接触  卡  成功插入
```

**论文价值**:
- 定量证明残差在"异常时刻"激活
- 可以做成论文的**核心Figure** (类似AlphaGo的价值网络曲线)

---

#### 1.2 空间热图分析 (Spatial Residual Heatmap)
**实验设计**:
- 选择几个关键帧（正常接触 vs. Jamming）
- 可视化：
  - `z_exp` (Prophet预测的触觉)
  - `z_real` (真实观测)
  - `r_t = z_real - z_exp` (残差热图)
- 在残差热图上标注："这个区域是peg卡在了孔的左壁"

**预期结果**:
- 正常帧: `r_t` 接近0 (Prophet预测准确)
- Jamming帧: `r_t` 在接触点有强烈激活 (Prophet没预料到的力)

**论文价值**:
- 视觉化证明残差的物理意义
- 可以做成补充材料的视频

---

#### 1.3 与Force/Torque传感器的相关性分析 ⭐
**实验设计**:
- 如果您有F/T传感器数据，计算：
  - Correlation(`||r_t||`, `||F_unexpected||`)
  - 其中 `F_unexpected = F_measured - F_predicted_by_vision`
- 如果相关系数 > 0.7，这是**强有力的证据**

**论文价值**:
- 用物理量（力）验证了残差的物理含义
- 可以写成一个短的Analysis section

---

### **方向2: 下游任务验证 - Policy Learning** 🔥 **强烈推荐**

**核心问题**: 当前只做了representation learning，没有真正用到downstream task

#### 2.1 插孔Policy训练 (Peg-in-Hole Policy)
**实验对比**:
1. **Baseline**: Policy只用 `[visual, tactile_full]`
2. **Ours**: Policy用 `[visual, residual_only]`
3. **Ablation**: Policy用 `[visual, z_exp, r_t]` (三者都用)

**评估指标**:
- Success Rate
- Insertion Time
- Recovery from Jamming (关键!)

**预期结果**:
- `residual_only` 应该在Jamming场景表现最好
- `z_exp + r_t` 可能是最优的（充分利用所有信息）

**论文价值**:
- 证明残差表征的**实用价值**
- 这是RA-L/ICRA必须有的downstream验证

---

#### 2.2 Jamming Detection任务 (二分类)
**实验设计**:
- 训练一个简单的分类器：`Classifier(r_t) → {Normal, Jamming}`
- 对比特征：
  - `Baseline1`: 直接用原始tactile image
  - `Baseline2`: 用VQ-VAE的full latent
  - `Ours`: 用residual `r_t`

**评估指标**:
- Precision/Recall (检测Jamming的准确率)
- Latency (多快能检测到)

**预期结果**:
- 残差应该在F1-score上明显领先

**论文价值**:
- 简单但有效的downstream任务
- 可以作为Section 4.2的补充实验

---

### **方向3: 消融实验 - 证明设计选择的必要性**

---

#### 3.2 Prophet设计的消融
**对比实验**:
1. **No Visual**: Prophet只用robot state
2. **No State**: Prophet只用visual
3. **Ours**: Visual + State

**评估指标**:
- Prophet prediction error (`||z_real - z_exp||`)
- Residual activation (`mean(||r_t||)`)

**预期结果**:
- Visual+State应该预测最准（residual最稀疏）

---

### **方向4: 泛化性实验 - 证明方法的通用性**

#### 4.1 跨物体泛化 (Cross-Object Generalization)
**实验设计**:
- 在Allen Key上训练
- 在不同形状的peg上测试 (圆形、方形、六角形)

**评估指标**:
- Zero-shot performance on new objects
- Fine-tuning efficiency

**论文价值**:
- 证明方法不是over-fit到一个特定物体
- 这是UniT的核心卖点，您的方法也应该继承

---

#### 4.2 跨任务泛化 (Cross-Task Transfer)
**实验设计**:
- 在插孔任务上训练Residual Encoder
- 迁移到其他任务：

**评估指标**:
- Transfer learning efficiency
- Performance on new tasks

---

### **方向5: 理论分析 - Information Bottleneck视角** 🤓

**核心问题**: 从信息论角度解释为什么残差学习有效

#### 5.1 互信息分析 (Mutual Information)
**分析目标**:
- 计算 `I(r_t; Jamming_event)` (残差与异常事件的互信息)
- 对比 `I(z_full; Jamming_event)` (完整latent的互信息)

**预期结果**:
- 残差应该对异常事件有更高的互信息（更相关）

**论文价值**:
- 理论支撑，适合投ICLR/NeurIPS这种理论会议
- 可以作为Theoretical Analysis section

---

#### 5.2 Disentanglement度量
**分析目标**:
- 计算 `z_exp` 和 `r_t` 的正交性
- 使用MIG (Mutual Information Gap) 或 SAP Score

**预期结果**:
- 两者应该是正交的（解耦的）

**论文价值**:
- 证明架构确实实现了"预期-意外"的解耦

---

## 🎯 优先级建议

### **必须做** (论文核心支撑):
1. ✅ **方向1.1**: 时间序列残差激活曲线 (核心Figure)
2. ✅ **方向1.2**: 空间残差热图 (可视化证明)
3. ✅ **方向2.1**: 插孔Policy实验 (downstream验证)

### **强烈推荐** (增加论文深度):
6. ⭐ **方向2.2**: Jamming检测任务 (简单但有效)
7. ⭐ **方向3.2**: Prophet设计消融

### **加分项** (提升论文档次):
9. 🌟 **方向5.1**: 互信息分析 (理论支撑)

---


## 🔧 技术实现提示

### 对于方向1.1 (残差时间曲线):
```python
# 伪代码
def analyze_residual_activation(model, episode_data):
    residuals = []
    for t, frame in enumerate(episode_data):
        outputs = model(frame)
        r_t = outputs['residual']
        residuals.append(r_t.norm(dim=(1,2,3)).mean().item())

    plt.plot(residuals)
    plt.axvline(x=jamming_frame, color='red', label='Jamming')
    plt.xlabel('Time step')
    plt.ylabel('||r_t||')
    plt.title('Residual Activation over Time')
```

### 对于方向2.1 (Policy训练):
- 可以复用UniT的policy代码框架
- 只需修改observation encoding部分
- 建议从简单的baseline开始（只用residual）

### 对于方向3.1 (课程学习消融):
- 您已经有3个checkpoint了（不同λ阶段）
- 可以直接加载评估，不需要重新训练

---

## 📝 论文结构建议

基于这些补充实验，论文可以这样组织：

```
1. Introduction
   - 动机: 精密装配中的"视-触冲突"
   - 核心Idea: 残差 = 意外信号

2. Related Work
   - Tactile Representation Learning
   - Predictive Coding in Robotics
   - Residual Learning

3. Method
   3.1 Prophet-Residual Architecture
   3.2 Training with Sparsity Constraint
   3.3 Curriculum Learning Strategy

4. Experiments
   4.1 可解释性分析 (方向1) ← 核心贡献
       - 时间序列激活
       - 空间热图
       - F/T相关性

   4.2 Downstream任务 (方向2)
       - Policy Learning
       - Jamming Detection

   4.3 Ablation Study (方向3)
       - Prophet设计

   4.4 泛化性实验 (方向4)

5. Theoretical Analysis (方向5, optional)

6. Conclusion
```


