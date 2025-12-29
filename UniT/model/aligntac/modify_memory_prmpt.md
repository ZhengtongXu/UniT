任务：优化 PyTorch 的 MemoryModule 类（保持有状态设计）
背景： 这个模块使用 FIFO 队列存储过去 K 帧特征。请在保持当前 有状态（Stateful）设计（即保留 self.memory_queue）的前提下，修复初始化逻辑和数据流问题。不要修改 forward 的输入输出接口。
请执行以下核心修改：
1. 实现“首帧复制”初始化 (Replication Padding)

触发条件：
self.memory_queue 为空（或为 None）时。
或者当前输入的 batch size 与队列中存储的 batch size 不一致时（需触发重置）。
初始化动作：
不要只存入一帧。而是将当前帧 z_dyn (必须 detach) 复制 K 次，并将这 K 个副本填满队列。
目的：让视频的第一帧就能拥有完整的 K 长度的历史上下文（虽然都是它自己的副本），从而进行有效的 Self-Attention 计算。
2. 统一的更新逻辑

步骤：
检查是否需要初始化（如上所述）。
入队：将当前帧 z_dyn (detach 后) 加入队列尾部。
出队：如果队列长度 >K，弹出最旧的一帧。
注意：如果是第一帧，初始化时已经填满了 K 个，紧接着又 Append 了一个当前帧，此时长度变为 K+1，必须立即 Pop 一个，最终保持长度为 K。
3. 修正 Attention 数据流

Key/Value 构建：使用队列中 所有 K 帧（包含当前帧）。严禁使用 [:-1] 排除当前帧。
全流程执行：删除 if len(self.memory_queue) < 2: return 这种跳过逻辑。
一致性：确保 每一帧（包括第一帧）都完整经过 Cross-Attention -> Residual Add -> LayerNorm。这保证了第一帧（作为 Self-Attention）和后续帧输出特征分布的一致性。

预期行为 (设 K=5):
第一帧：检测到队列为空 -> 初始化放入 5 个 F1 -> Append F1 -> Pop F1 -> 队列仍为 5 个 F1。Attention 计算 Q=F1, KV=[F1...F1]。正常输出。
第二帧：Append F2 -> Pop F1 -> 队列变为 [F1, F1, F1, F1, F2]。Attention 计算 Q=F2, KV=[历史+当前]。