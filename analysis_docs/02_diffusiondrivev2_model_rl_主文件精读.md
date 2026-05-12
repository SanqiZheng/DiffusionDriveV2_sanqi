# 02. diffusiondrivev2_model_rl.py 主文件精读

主文件路径：

- [navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py)

这一份文件可以分成 6 块：

1. PDM 打分辅助函数
2. 整体模型 `V2TransfuserModel`
3. 目标检测头 `AgentHead`
4. 扩散轨迹 decoder 组件
5. 自定义 scheduler `DDIMScheduler_with_logprob`
6. 真正训练核心 `TrajectoryHead`

---

## 1. 顶部辅助函数：PDM 分数怎么拿回来

### 1.1 `_pairwise_scores`

位置：

- [diffusiondrivev2_model_rl.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py) 第 75 行

作用：

- 从 `PDMScorer` 的内部缓存里重新算出每条候选轨迹的最终 score
- 返回 shape 为 `(G,)`

关键点：

1. 它不是直接相信 `scorer.score_proposals` 的全局归一化结果。
2. 它把 `progress` 指标重新按“参考 proposal vs 当前 proposal”的方式归一化。
3. 最终分数还是：

`multiplicative metrics * weighted metrics`

物理意义：

- 一条轨迹先要“不过碰撞/不过界”等硬门槛
- 然后再看 progress、ttc、comfort 等软指标

### 1.2 `_pairwise_subscores`

位置：

- [diffusiondrivev2_model_rl.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py) 第 38 行

作用：

- 把 `no_collision / drivable_area / progress / ttc / comfort / dir_weighted / final` 全部拆出来

为什么要拆：

- 后面训练不只看最终分数
- 如果 `no_collision` 或 `drivable_area` 不达标，代码会直接把优势设成负值或无效

### 1.3 `_pdm_worker`

位置：

- [diffusiondrivev2_model_rl.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py) 第 119 行

作用：

- 在子进程里加载 `metric_cache.pkl`
- 调用 `pdm_score_para`
- 返回总分和子分数

开发者为什么这么做：

- PDM simulator 和 scorer 很重
- batch 内每个样本的 cache 文件不同
- 放到 `ProcessPoolExecutor` 里并行，比主进程串行快很多

---

## 2. `V2TransfuserModel`：从传感器到 query 的总入口

### 2.1 模型骨架

位置：

- [diffusiondrivev2_model_rl.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py) 第 143 行

它的结构可以理解为：

1. `TransfuserBackbone` 提取图像和激光雷达融合特征
2. 一个 transformer decoder 产生 `trajectory_query` 和 `agents_query`
3. `TrajectoryHead` 负责轨迹扩散规划
4. `AgentHead` 负责 2D agent box 预测

### 2.2 输入张量

来自 feature builder：

- `camera_feature`: `[B, 3, 256, 1024]`
- `lidar_feature`: `[B, 1, 256, 256]`，如果 `use_ground_plane=True` 则通道会变多
- `status_feature`: `[B, 8]`

`status_feature` 由以下部分拼成：

- driving command: 4 维
- ego velocity: 2 维
- ego acceleration: 2 维

对应 [transfuser_features.py](../navsim/agents/diffusiondrivev2/transfuser_features.py) 第 39-52 行。

### 2.3 backbone 输出怎么变成 token

关键逻辑在 `V2TransfuserModel.forward`：

- [diffusiondrivev2_model_rl.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py) 第 221 行

步骤如下：

1. `self._backbone(camera_feature, lidar_feature)` 输出：
   - `bev_feature_upscale`
   - `bev_feature`
   - 第三个返回值未用
2. `bev_feature` 经过 `1x1 conv` 降到 `256` 通道，再 flatten 成 token。
3. `status_feature` 经过线性层编码成一个额外 token。
4. 把 BEV token 和 status token 拼到一起，形成 `keyval`。

如果 backbone 最后一层是 `8x8` 特征图，那么：

- `bev_feature` 原始约为 `[B, 512, 8, 8]`
- 降维后是 `[B, 256, 8, 8]`
- flatten 后是 `[B, 64, 256]`
- 拼接 status token 后是 `[B, 65, 256]`

这里有个很重要的工程耦合：

- `_keyval_embedding = nn.Embedding(8**2 + 1, 256)`

这说明作者默认最后 BEV token 网格就是 `8x8`，不是完全自适应的。

### 2.4 `cross_bev_feature` 是什么

很多初学者读到这里会疑惑：既然已经有 `keyval` 了，为什么还要再造一个 `cross_bev_feature`？

原因是：

- `keyval` 适合 transformer 做全局 token 交互
- 但轨迹点 refinement 时，更适合“拿具体轨迹点坐标去地图上取局部特征”

所以代码做了两件事：

1. 把 token 形式的 `keyval[:,:-1]` 重新还原成 `8x8` feature map
2. 上采样到和 `bev_feature_upscale` 一样大
3. 与 `bev_feature_upscale` concat 后再投影成 `256` 通道

这样后面的 `GridSampleCrossBEVAttention` 就可以直接在高分辨率 BEV 图上按轨迹点采样。

### 2.5 为什么要把 `_trajectory_head` 调两次

最关键的一段在：

- [diffusiondrivev2_model_rl.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py) 第 258-264 行

先看逻辑：

1. 先在 `torch.no_grad()` 下跑一次 `_trajectory_head(...)`，得到 `old_pred`
2. 再带着 `old_pred` 跑第二次 `_trajectory_head(...)`

这其实就是 RL 的“两阶段”：

1. 第一次相当于采样 rollout，得到旧策略下的轨迹链、reward、advantages
2. 第二次相当于在当前参数下重算 log-prob，然后做 policy gradient loss

正向思维：

- RL 训练不只是“前向一次出 loss”
- 需要先有一条采样轨迹链，才能评价它好不好
- 所以第一次 forward 更像“收集经验”，第二次 forward 更像“根据经验算梯度”

---

## 3. `AgentHead`：不是重点，但要知道它在干嘛

位置：

- [diffusiondrivev2_model_rl.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py) 第 274 行附近

作用：

- 从 `agents_query` 预测周围车辆的框和存在性

输出：

- `agent_states`: `[B, num_bounding_boxes, 5]`
- `agent_labels`: `[B, num_bounding_boxes]`

这部分主要服务于多任务训练遗留接口，对 RL 轨迹主链路不是核心。

---

## 4. 扩散解码器部分：条件是怎么注入轨迹的

### 4.1 `DiffMotionPlanningRefinementModule`

位置：

- [diffusiondrivev2_model_rl.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py) 第 314 行

作用：

- 对每个 mode 输出分类分数和轨迹残差

输入：

- `traj_feature`: `[B, 20, 256]`

输出：

- `plan_reg`: `[B, 20, 8, 3]`
- `plan_cls`: `[B, 20]`

这里的 `20` 不是 group 数，而是 anchor 数。

### 4.2 `ModulationLayer`

位置：

- [diffusiondrivev2_model_rl.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py) 第 360 行

它做的是 FiLM 风格调制：

`traj_feature = traj_feature * (1 + scale) + shift`

这里 `scale, shift` 来自 diffusion timestep embedding。

直觉：

- 同一条 noisy trajectory，在不同去噪阶段应该有不同的修正幅度
- 早期更粗，后期更细

### 4.3 `CustomTransformerDecoderLayer`

位置：

- [diffusiondrivev2_model_rl.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py) 第 401 行

这是全文件最值得反复读的模块之一。

它的条件注入顺序是：

1. `cross_bev_attention`
2. `cross_agent_attention`
3. `cross_ego_attention`
4. `ffn`
5. `time_modulation`
6. `task_decoder`

#### 第一步：`cross_bev_attention`

不是对整张图做 global attention，而是：

- 用 noisy trajectory 上的 `8` 个点去 BEV 图上 `grid_sample`
- 每个 query 只取自己对应轨迹点附近的局部地图特征

对应复用模块：

- [blocks.py](../navsim/agents/diffusiondrive/modules/blocks.py) 第 42 行 `GridSampleCrossBEVAttention`

这非常符合规划任务的物理直觉：

- 一条候选轨迹该不该改，不需要盯整张图
- 最有用的是“这条路线上附近的地形和障碍”

#### 第二步：看周围车

`cross_agent_attention` 让每个轨迹 query 去读 `agents_query`。

物理意义：

- 轨迹不能只看车道线
- 还得知道周围车在哪里、朝哪开

#### 第三步：看 ego 全局状态

`cross_ego_attention` 读单个 `ego_query`。

可以理解成给局部轨迹点一个全局驾驶意图或场景上下文补充。

#### 第四步：输出残差轨迹

最后 `poses_reg[..., :2] += noisy_traj_points`

这说明网络不是凭空生成一条新轨迹，而是在当前 noisy trajectory 基础上做 residual refinement。

这比直接从零回归更稳定，因为：

- 初始轨迹已经有 anchor 提供的结构先验
- 网络只要负责“往哪个方向改”

### 4.4 `CustomTransformerDecoder`

位置：

- [diffusiondrivev2_model_rl.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py) 第 492 行

它会把上层输出的 `poses_reg[..., :2]` detach 后作为下一层的 `traj_points`。

虽然当前只堆了 1 层，但这个接口明显是给多层 refinement 预留的。

---

## 5. `DDIMScheduler_with_logprob`：为什么要魔改 scheduler

位置：

- [diffusiondrivev2_model_rl.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py) 第 527 行

### 5.1 普通 diffusion scheduler 不够用

普通 DDIM/DDPM 在推理里只关心：

- 给定 `x_t`
- 模型输出
- 算出 `x_{t-1}`

但 RL 训练还需要知道：

- 当前策略以多大概率产生这一步 transition

也就是：

- `log p(x_{t-1} | x_t)`

所以作者自定义了这个 scheduler。

### 5.2 最关键的一句：预测目标是 `sample`

文件注释写得很直白：

- 模型学习的是 `x_start`
- 不是 `noise_epsilon`
- 不是 `velocity`

当 `prediction_type == "sample"` 时：

- `model_output` 直接被当成 `pred_original_sample`

这和很多 diffusion 教程默认的 epsilon-prediction 不一样。

### 5.3 乘性噪声为什么适合轨迹

作者没有只加普通加性高斯噪声，而是引入了乘性噪声：

- `prev_sample = prev_sample_mean * variance_noise_mul + std_dev_t_add * variance_noise_add`

直觉上：

- 轨迹近处点和远处点尺度不同
- 纯加性噪声会让局部结构容易被打断
- 乘性噪声更像“按比例放缩”，更容易保留轨迹整体形状

这和论文里 `scale-adaptive multiplicative noise` 是对上的。

### 5.4 `log_prob` 是怎么来的

代码最后计算：

`log_prob = - (prev_sample - mean)^2 / (2 sigma^2) - log sigma - const`

然后对轨迹点和坐标维求和。

也就是说，一步 denoising transition 被近似成高斯策略，RL 用这一步的 log-prob 做策略梯度。

---

## 6. `TrajectoryHead`：整份文件真正的主角

位置：

- [diffusiondrivev2_model_rl.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py) 第 695 行

### 6.1 初始化阶段在准备什么

它初始化了几类东西：

1. 两个 scheduler
   - `diffusion_scheduler`
   - `diffusionrl_scheduler`
2. anchor 轨迹
3. anchor 编码器
4. 时间步编码器 `time_mlp`
5. 轨迹 refinement decoder
6. PDM 打分所需的 simulator/scorer 和多进程池

### 6.2 `norm_odo` / `denorm_odo`

位置：

- 第 790 行
- 第 799 行

归一化方式：

- `x / 50`
- `y / 20`

这说明作者认为：

- 前向距离范围通常更长
- 横向偏移范围通常更短

所以需要按不同尺度归一化，避免一个维度主导训练。

### 6.3 `forward` 的三路分发

位置：

- 第 810 行

逻辑：

- 训练且 `old_pred is None` -> `forward_train_rl`
- 训练且 `old_pred is not None` -> `get_rlloss`
- 测试 -> `forward_test_rl`

这就是为什么外层 `V2TransfuserModel.forward` 要先 no-grad 调一次，再正式调一次。

---

## 7. `forward_train_rl`：第一阶段，采样 rollout 并计算 advantage

位置：

- [diffusiondrivev2_model_rl.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py) 第 835 行

### 7.1 候选轨迹是怎么铺开的

关键 shape：

1. `self.plan_anchor`: `[20, 8, 2]`
2. `unsqueeze + repeat` 后：`[B, num_groups, 20, 8, 2]`
3. `view` 后：`[B, num_groups * 20, 8, 2]`

如果 `num_groups = 4`：

- 总候选数就是 `80`

它的真实含义是：

- 20 个 anchor
- 每个 anchor 采样 4 次

### 7.2 初始噪声从哪来

不是从全 0 或纯随机轨迹来，而是：

1. 先把 anchor 归一化
2. 对 anchor 加截断噪声
3. 得到 `diffusion_output`

所以这里的 `diffusion_output` 可以理解为：

- “带噪 anchor 轨迹”
- 而不是“毫无结构的随机轨迹”

### 7.3 每一步去噪在干什么

循环里每一步都做 6 件事：

1. `clamp` 当前轨迹到 `[-1, 1]`
2. `denorm_odo` 还原到真实坐标尺度
3. 用 `gen_sineembed_for_position` 做轨迹点位置编码
4. 用 `time_mlp` 编码时间步
5. 用 `diff_decoder` 预测 `x_start`
6. 调 `diffusionrl_scheduler.step` 得到下一个 sample 和 `log_prob`

这里最关键的对象关系是：

- `diffusion_output`: 当前噪声状态 `x_t`
- `noisy_traj_points`: 把 `x_t` 还原后的真实坐标版本
- `x_start`: 网络预测的干净轨迹 `x_0`
- `prev_sample`: 下一步状态 `x_{t-1}`

### 7.4 为什么还要 `bezier_xyyaw`

循环结束后，代码把最终 `xy` 轨迹变成 `(x, y, yaw)`：

- [diffusiondrivev2_model_rl.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py) 第 1158 行

原因：

- PDM scorer 需要完整状态
- yaw 由 Bézier 曲线导数恢复，能保证方向变化更光滑

### 7.5 reward 怎么算

代码会把：

- 模型候选轨迹
- `targets["trajectory"]`

拼到一起，再送进 `get_pdm_score_para`。

这里要特别小心一个容易误解的点：

- `PDMScorer` 内部第 0 条 reference trajectory 来自 `metric_cache.trajectory`
- 它不是这里手工拼进去的 `targets["trajectory"]`

也就是说，reward 体系里至少有三种轨迹同时存在：

1. `metric_cache.trajectory`: PDM closed planner 的参考轨迹
2. `targets["trajectory"]`: 监督未来轨迹
3. `diffusion_output`: 模型候选

### 7.6 advantage 是怎么构造的

这一段非常重要。

`reward_group` 被 reshape 成：

`[B, num_groups, 20]`

这说明代码是把“同一个 anchor 的多次探索”放在 `num_groups` 维上。

然后：

`mean_grouped_rewards = reward_group.mean(dim=1)`

`std_grouped_rewards = reward_group.std(dim=1)`

于是每个 anchor 的 advantage 是在“同 anchor 的多次采样”之间标准化的。

这正对应论文里的 `intra-anchor GRPO`。

然后又做两步截断：

1. 只保留 `reward > reward_gt` 的样本
2. 如果 `no_collision` 或 `drivable_area` 失败，直接把 advantage 置成负值

这对应论文里强调的“约束低质量 mode”和“truncated”思想。

### 7.7 返回值不是 loss，而是“经验包”

`forward_train_rl` 最后返回：

- `all_diffusion_output`
- `advantages`
- `reward`
- `sub_rewards`

注意：

- 这里还没有真正形成最终训练 loss
- 它更像 rollout buffer

---

## 8. `forward_test_rl`：测试时怎么采样

位置：

- [diffusiondrivev2_model_rl.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py) 第 977 行

几个关键差异：

1. `step_num = 2`
2. `num_groups = 4` 被硬编码
3. scheduler step 时 `eta=0.0`

这说明测试更像 deterministic DDIM 风格采样，而不是训练时带随机探索的采样。

返回：

- `loss`: 用 GT 做的 L1，仅作日志参考
- `reward`
- `sub_rewards`
- `all_diffusion_output`
- `log_probs`

---

## 9. `get_rlloss`：第二阶段，真正把 reward 变成 loss

位置：

- [diffusiondrivev2_model_rl.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py) 第 1063 行

这是最值得啃透的函数。

### 9.1 输入不是原始数据，而是第一次 forward 的结果

它拿到：

- `old_diffusion_output = old_pred["all_diffusion_output"]`
- `advantages = old_pred["advantages"]`

也就是：

- 一整条旧策略采样链
- 每条轨迹每一步对应的优势

### 9.2 它在做什么

它会沿着旧链的每一个 step：

1. 重新把 `x_t` 送进当前模型
2. 重新预测 `x_start`
3. 用 `prev_sample=chains_prev[..., i]` 强制评估“当前策略给旧动作的概率”
4. 拿到 `log_prob`

这就是典型 RL 里“on-policy 采样后，再用当前策略评估动作概率”的味道。

### 9.3 这一句最像策略梯度

`per_token_loss = -torch.exp(per_token_logps - per_token_logps.detach()) * advantages`

直觉解释：

- 前向值里 `exp(logp - stopgrad(logp)) = 1`
- 但梯度仍然会沿着 `logp` 传播
- 所以它相当于实现了“按 advantage 加权的 score function gradient”

这是一种工程上很常见的 trick。

### 9.4 为什么还要加 IL loss

代码里除了 RL loss，还加了一个辅助 imitation loss：

- 对每个 step、每层 decoder 的 `poses_reg` 做 L1

动机很直接：

- 纯 RL 太容易发散
- 加一点轨迹监督能稳住训练

更细一点：

- 如果这一批没有正 advantage 样本，`il_weight = 1.0`
- 否则 `il_weight = 0.1`

这代表开发者在用 IL 当“安全绳”：

- 有有效 RL 信号时，以 RL 为主
- 没有有效 RL 信号时，退回监督学习

---

## 10. `bezier_xyyaw`：为什么不用网络直接回归 yaw

位置：

- [diffusiondrivev2_model_rl.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py) 第 1158 行

函数逻辑：

1. 在最前面插入固定原点 `(0, 0)`
2. 把 8 个预测点当作 Bézier 控制点
3. 求 Bézier 曲线的一阶导
4. 用 `atan2(dy, dx)` 得到每个点的 yaw

直觉：

- 轨迹方向本来就是曲线切线方向
- 用导数恢复 yaw，比生硬地让网络直接猜角度更自然

工程价值：

- 避免角度回归不连续
- 让方向随轨迹几何自动一致

---

## 11. 你读这个主文件时最该抓住的主链

一条训练样本的主链可以写成：

`camera/lidar/status -> backbone -> transformer query -> trajectory_head第一次采样 -> PDM reward/advantage -> trajectory_head第二次重算logprob -> RL+IL loss`

如果你把这条链读通，这个文件就基本吃透了。
