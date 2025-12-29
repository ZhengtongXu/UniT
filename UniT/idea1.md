[Research Proposal] Feeling the Unexpected: Residual Tactile Representation Learning via Visuo-Motor Prediction

**(感受那份意外：基于视-动预测的残差触觉表征学习)**
Feeling the unexpected：意外触碰是对操作任务最有用的触觉信息。利用当前时刻腕部图像预测当前时刻触觉，将预测触觉与真实触觉的信息差作为“意外信号”，将意外信号作为额外的触觉表征送入下游policy。理想情况下是对齐过程中出现卡死、碰撞等事件时（腕部图像看不到），意外信号突增，从而有益于policy的学习。

### 1. The Elevator Pitch (一句话核心)

针对精密装配中“视觉看不清、触觉难解释”的痛点，我们提出一种**“残差学习”**范式：利用视觉和动作建立物理预期，强迫触觉 Encoder 只关注**“违反预期的接触异常” (Residual Surprise)**，从而极大地提升机器人对微小对齐误差和卡顿 (Jamming) 的感知灵敏度。

---

### 2. 动机与痛点 (Motivation)

在精密插孔 (Peg-in-Hole) 任务中，现有的触觉表征方法面临两大挑战：

1. **信息冗余 (Redundancy):** 大部分时间的触觉信号（如平稳接触）是完全可以通过视觉 ($V$) 和 动作 ($A$) 预测出来的。现有的 Encoder 浪费了大量带宽去编码这些“显而易见”的信息。
    
2. **视-触觉冲突 (Visual-Tactile Conflict):** 在插入瞬间，视觉往往因为透视或遮挡认为“已对齐”，而触觉感受到了“卡壁”。如果直接融合，Policy 容易被视觉主导而选择硬怼。我们需要一个**纯粹的错误信号**来打破这种僵局。
    

**核心洞察:** 借鉴神经科学的**预测编码 (Predictive Coding)** —— 大脑只有在“预测失效”时才会激活强烈的感知信号。

---

### 3. 方法论 (Methodology): The "Prophet-Residual" Architecture

我们设计了一个非对称的双流架构，引入物理先验来“过滤”触觉信号。
![[Pasted image 20251229125004.png]]

#### 3.1 架构设计

- **流 A：预期生成器 (The Prophet Stream)**
    
    - **Input:** Vt​ (视觉) + St​ (机器人状态)
        
    - **Output:** zexp​ (Expected Tactile Latent) —— "大脑想象中的触觉"。
        
- **流 B：观测编码器 (The Observation Stream)**
    
    - **Input:** Tt​ (真实 GelSight 图像)
        
    - **Output:** zreal​ (Observed Tactile Latent) —— "真实发生的触觉"。
        
- **核心创新：离散化残差模块 (The Quantized Residual)**
    
    - **Step 1: 原始残差:** 计算隐空间差异 Δz=zreal​−zexp​。
        
    - **Step 2: 向量量化 (VQ):** 引入一个可学习的“意外码本” (Surprise Codebook) C={e1​,...,eK​}。我们将 Δz 映射到码本中最近的向量上：
        
        zq​=ek​,where k=argjmin​∣∣Δz−ej​∣∣2​
        
    - **物理含义:**
        
        - **Code 0 (The Void):** 对应 Δz≈0 的情况。表示“一切符合预期”，VQ 会自动过滤掉细微的传感器噪声，输出纯净的零信号。
            
        - **Code K (The Crash):** 对应不同类型的意外。例如，Code 1 代表“左侧碰撞”，Code 2 代表“底部硬接触”。VQ 将连续误差**聚类**成了离散的物理事件。
            
- **解码器 (Decoder):**
    
    - **Input:** zexp​+zq​ (预期 + 意外)
        
    - **Target:** 重建真实的触觉图像 Tt​。
        

#### 3.2 训练机制 (Training with VQ-Bottleneck)

Loss 函数现在包含三部分：重建误差、VQ 承诺损失 (Commitment Loss) 和 稀疏约束。



---

### 4. 下游策略应用 (Policy Implementation)

在下游模仿学习或 RL 中，我们采用**全量信息 + 残差引导**的输入策略：

**Policy Input:**

$$State = Concat(\ e_{vision}, \ S_t, \ z_{real}, \ \mathbf{r_t}\ )$$

- **$e_{vision}$ (The Navigator):** 负责全局导航（往哪走）。
    
- **$z_{real}$ (The Base):** 提供基础接触状态（如当前受力大小），作为安全兜底。
    
- **$r_t$ (The Interrupt):** **关键创新点。** 它是一个高信噪比的“报警器”。
    
    - 当 $r_t \approx 0$: Policy 听视觉的，大胆执行。
        
    - 当 $r_t$ 突增: Policy 收到明确信号——“这里不对劲”，立即执行纠偏或退刀动作。

![[Pasted image 20251218230715.png]]

---

### 5. 数据策略：变废为宝 (Data Strategy)

本方法的成功依赖于特定的数据分布，我们将通常被视为“失败”的数据转化为核心训练资源。

- **成功数据 (Golden Runs, ~30%):** 一次性顺滑插入。
    
    - _作用:_ 训练 Prophet，建立“什么是正常”的基准。
        
- **纠错数据 (Retry/Recovery Trajectories, ~70%):** 包含卡顿、摩擦、调整、再插入的全过程。
    
    - _作用:_ 这是 $r_t$ 的激活源。在卡顿瞬间，$z_{exp}$ 预测继续进，$z_{real}$ 感受阻力，导致 $r_t$ 爆发。这教会了 Policy **“当 $r_t$ 变大时，该如何调整动作”**。
        

---

### 6. 实验验证计划 (Experimental Plan)

硬件: Robotic Arm + RealSense (Eye-in-Hand) + GelSight (Fingertip).

任务: 高精度 (公差 < 0.5mm) USB/工业连接器盲插。

**验证指标:**

1. **Success Rate:** 在存在初始位置误差下的纠错成功率。
    
2. **Contact Force:** 插入过程中的峰值力（证明 $r_t$ 能让机器人更“轻柔”地处理异常，而不是硬怼）。
    
3. **Visualization:** 绘制 $r_t$ 激活曲线，展示其在 Jamming 瞬间的尖峰响应。
    

---

### 7. 找 Mentor 聊的“话术” (Talking Points)

如果 Mentor 问起，你可以这样由浅入深地介绍：

1. **切入点:** “老师，我看了最近关于 Action-Conditioned Tactile 的论文（如 TPP），我觉得它们虽然引入了动力学，但还是在‘预测未来’。对于精密装配，我觉得更重要的是‘预测偏差’。”
    
2. **核心Idea:** “我想做一个减法。用视觉和动作去预测触觉的‘背景信息’，然后只把预测不到的那个‘残差’提取出来。这个残差就是最纯粹的‘意外信号’，专门用来解决卡顿和对齐问题。”
    
3. **可行性:** “这套架构不需要新的硬件，只需要在数据采集时保留那些‘插坏了再调整’的数据。那些所谓的‘脏数据’其实正好包含了物理冲突的信息，是训练这个残差模型的关键。”
    
4. **兜底方案:** “下游 Policy 我会同时输入原始特征 $z_{real}$ 和残差 $r_t$，这样既保留了绝对信息，又有了高灵敏度的异常检测，风险很低，但上限很高。”

## 神经学故事
"生物智能的一个核心特征并非是对环境的被动感知，而是主动预测 (Active Prediction)。根据Friston的自由能原理，大脑通过最小化**预测误差 (Prediction Error)** 来理解世界。

这种机制在精细运动控制中尤为关键。神经学研究表明，生物利用运动指令的**副本 (Efference Copy)** 来预测感官后果，并抑制自发的感官输入，从而敏锐地检测到外部的扰动——这就解释了为什么人类能轻松区分‘主动按压’与‘意外碰撞’，而机器人却往往受困于视触觉信号的冲突与混淆。

受此启发，我们并未沿用传统的‘重构即学习’范式，而是提出了一种**类脑的‘Prophet-Residual’架构**。我们将触觉表征学习重构为一个**感觉运动预测 (Sensorimotor Prediction)** 问题：

1. **Prophet (The Forward Model):** 模仿小脑功能，利用视觉上下文和动作意图，幻想预期的触觉流。
    
2. **Residual (The Surprise):** 提取真实触觉与预期的偏差，作为高信噪比的‘意外’信号。
    

这种设计赋予了机器人类似于生物的**‘感觉衰减’ (Sensory Attenuation)** 能力——忽略符合预期的物理接触，而对微米级的对齐误差（Surprise）保持极高的敏感度。

### 理论支柱一：预测编码理论 (Predictive Coding) —— "大脑是一台预测机器"

**核心来源：** 计算神经科学、Karl Friston 的**自由能原理 (Free Energy Principle)**。

- **理论故事：** 传统的认知观认为大脑是“接收输入 -> 处理 -> 输出”的被动装置。但现代神经科学（Friston, 2010）认为，大脑是一个**主动的预测机器**。大脑皮层时刻在通过内部模型（Internal Model）自顶向下地预测感觉输入。
    
    - **关键点：** 大脑只传递**预测误差 (Prediction Error)**。如果感官输入完全符合预期，神经元甚至不会放电（被抑制了）。只有当“意外”发生时，误差信号才会向上传递，引起注意和学习。
        
    - **哲学隐喻：** **"Surprise is the only thing that carries information." (只有意外才携带信息)**。在香农信息论中，发生概率越小的事情（越意外），信息量越大。
        
- **对应你的论文：**
    
    - **Prophet (Stream A):** 就是大脑的 **"Generative Model" (生成模型)**，它基于视觉和动作指令，自顶向下地生成预期的触觉流。
        
    - **Residual (rt​):** 就是神经科学中的 **"Sensory Prediction Error" (感官预测误差)**。
        
    - **Story升级：** “现有的触觉Encoder试图记住所有的触觉细节，这是一种低效的‘死记硬背’。受大脑预测编码机制的启发，我们设计了一个最小化‘自由能（惊喜）’的系统。我们的Encoder本质上是在学习抑制‘预料之中’的冗余信息，从而让‘预料之外’的接触异常（Jamming/Slip）像黑夜中的火炬一样显眼。”
        

---

### 理论支柱二：感知-运动副本 (Efference Copy / Corollary Discharge) —— "为什么你无法挠痒自己？"

**核心来源：** 运动控制神经学 (Motor Control)、小脑 (Cerebellum) 功能研究。

- **理论故事：** 生物体在发出运动指令（Action）时，大脑运动皮层会发送两份信号：
    
    1. 一份给肌肉（Efferent Signal），驱动肢体运动。
        
    2. 一份给感觉皮层，叫**“副本” (Efference Copy)**。
        
    
    - **作用：** 大脑利用这个“副本”提前预测这一动作会产生的感官后果，并主动对冲掉这些感觉。
        
    - **经典例子：** **“为什么自己挠痒痒不痒？”** 因为大脑知道你的手在动，通过Efference Copy预测到了触觉，并抑制了它。只有别人（外部不可控因素）挠你时，由于缺乏动作副本，预测失效，你才会感到“痒”（强烈的触觉刺激）。
        
- **对应你的论文：**
    
    - **引入Action的必要性：** 你在Prophet中引入 At​，就是在构建这个 **Efference Copy** 机制。
        
    - **Story升级：** “在精密装配中，机器人自身的运动会产生大量‘自发触觉噪声’（Self-generated tactile noise）。如果Encoder无法区分‘我用力按压产生的变形’和‘意外碰撞产生的变形’，控制就会失败。我们的架构通过引入Action作为Efference Copy，实现了**‘自发感觉衰减’ (Sensory Attenuation)**，使得模型能够像生物一样，专注于外部环境施加的‘非我’（Ex-afference）干扰。”
        

---

### 理论支柱三：小脑的前向模型与误差修正 (Cerebellar Forward Models)

**核心来源：** 机器人学与神经科学的交叉领域 (Wolpert & Kawato, 1998)。

- **理论故事：** 人脑中的**小脑 (Cerebellum)** 被认为是人体最精密的“卡尔曼滤波器”。它存储了身体的**前向动力学模型 (Forward Internal Model)**。
    
    - **机制：** 小脑接收动作指令，快速预测即时的感官后果（因为视觉和触觉传回大脑太慢了，有几十毫秒延迟）。
        
    - **功能：** 如果实际反馈与小脑的预测不符，小脑会立刻发出纠错信号，微调动作。这正是人类能进行穿针引线、精密手术等高难度动作的原因。
        
- **对应你的论文：**
    
    - **Prophet as Forward Model:** 你的 Prophet 模块本质上就是机器人的 **"Neural Cerebellum" (神经小脑)**。
        
    - **解决延迟：** 就像我们在上一次对话中讨论的，利用 At​ 预测 t+1，正是模仿小脑克服生物神经传输延迟的机制。
        
    - **Story升级：** “精密装配的挑战在于高频、微小的接触变化往往快于视觉反馈回路。我们提出的Prophet模块充当了一个可学习的‘小脑前向模型’。由此产生的残差信号不仅仅是特征，更是一种**‘实时误差校正信号’ (Real-time Error Correction Signal)**，这解释了为什么我们的方法在Downstream Policy中能实现超高精度的纠偏。”

