# 07. 逐函数精读：`DDIMScheduler_with_logprob.step` 与 `forward_train_rl`

主分析对象：

- [diffusiondrivev2_model_rl.py](/home/yihang/Downloads/CodeReference/diff_based/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py)

这篇专门讲训练时最重要的两段：

1. `DDIMScheduler_with_logprob.step`
2. `TrajectoryHead.forward_train_rl`

如果说上一份文档解释了“条件是怎么准备好的”，这一份就是解释：

“扩散采样到底是怎么一步步走的，为什么 reward 最后能变成 RL 信号。”

---

## 1. 先记住这一版 diffusion 学什么

这一版不是学噪声 `epsilon`，而是学 `x_start`。

代码证据：

```python
prediction_type="sample"
```

这在两个 scheduler 初始化里都写得很明确。

所以整个采样链的心智图应该是：

1. 当前我手里有 noisy trajectory `x_t`
2. 网络直接告诉我“它对应的干净轨迹 `x_0` 应该是什么”
3. scheduler 再根据这个 `x_0` 估计去生成 `x_{t-1}`

---

## 2. `DDIMScheduler_with_logprob.step` 在做什么

### 2.1 这不是普通 DDIM 的唯一输出版本

普通 scheduler 通常只返回：

- `prev_sample`

这个版本还要返回：

- `log_prob`
- `prev_sample_mean`

原因很简单：

- RL 更新不是只看采样结果
- 还要知道“当前策略产生这一步 transition 的概率”

所以开发者把 diffusion 的一步反推，解释成了策略的一步动作分布。

---

## 3. 这一步函数的数学角色

可以把它写成：

`p_theta(x_{t-1} | x_t, c)`

其中：

- `x_t` 是当前 noisy trajectory
- `x_{t-1}` 是下一步更干净一点的轨迹
- `c` 是场景条件

网络自己先输出：

- `x0_hat`

scheduler 再把它转成：

- transition mean
- transition variance
- sampled `x_{t-1}`
- 以及这次 sampled transition 的 `log_prob`

---

## 4. 逐步拆解 `step`

### 4.1 先得到前一个 timestep

代码：

```python
prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
```

含义：

- 当前在 `t`
- 要回推到 `t-1`

这里的 `t-1` 不是简单减 1，而是按当前 inference 步数在 1000 个训练时间步上等间隔跳。

### 4.2 取出扩散系数

代码：

```python
alpha_prod_t = self.alphas_cumprod[timestep]
alpha_prod_t_prev = ...
beta_prod_t = 1 - alpha_prod_t
```

物理意义：

- `alpha_prod_t` 决定当前样本还保留多少“原样本成分”
- `beta_prod_t` 决定当前样本里有多少“噪声成分”

### 4.3 最关键的一步：把 `model_output` 当 `x_start`

代码：

```python
elif self.config.prediction_type == "sample":
    pred_original_sample = model_output
    pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
```

意思是：

- 网络直接预测 `x_0`
- 再由 `x_t` 和 `x_0` 反推出隐含噪声 `epsilon`

这和 epsilon-prediction 的区别在于：

- epsilon-prediction 是先猜噪声，再推 `x_0`
- 这里是先猜 `x_0`，再补算噪声

对轨迹规划来说，这样更直观，因为网络直接在“未来轨迹空间”里工作。

### 4.4 计算无噪声均值 `prev_sample_mean`

代码：

```python
pred_sample_direction = ...
prev_sample_mean = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
```

这一步的直觉是：

- 如果不再注入随机噪声
- 纯靠当前预测的 `x_0` 回推
- 理论上下一步最合理的均值在哪

所以 `prev_sample_mean` 就是“策略分布的均值”。

---

## 5. 为什么这里用乘性噪声

### 5.1 代码长相

```python
variance_noise_mul = ...
prev_sample = prev_sample_mean * variance_noise_mul + std_dev_t_add * variance_noise_add
```

而且当前实现里：

- `std_dev_t_add` 基本就是 `0`

所以主要生效的是：

`prev_sample ≈ prev_sample_mean * variance_noise_mul`

### 5.2 物理直觉

轨迹不是图片像素。  
它的不同时间点、不同方向，尺度差异很明显：

- 前向 `x` 通常变化大
- 横向 `y` 通常变化小
- 远端点可容忍更大探索
- 近端点不宜乱抖

纯加性噪声像“每个位置加同样尺度的随机偏移”，对轨迹几何很不友好。  
乘性噪声更像“沿原轨迹尺度做相对伸缩”，更容易保留整体走势。

这就是论文里说的 `scale-adaptive multiplicative noise` 背后的开发者直觉。

### 5.3 为什么还分 horizontal / vertical

代码分别采样：

- `variance_noise_horizon`
- `variance_noise_vert`

再拼成 2 维坐标噪声。

说明开发者知道：

- `x` 和 `y` 的探索统计性质不完全一样
- 不应该简单共享同一个噪声标量

---

## 6. `log_prob` 为什么能用于 RL

代码：

```python
log_prob = (
    -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t_mul**2))
    - torch.log(std_dev_t_mul)
    - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
)
log_prob = log_prob.sum(dim=(-2, -1))
```

这就是高斯分布对数概率密度。

意思是：

- 把 `prev_sample_mean` 当作策略均值
- 把 `std_dev_t_mul` 当作策略标准差
- 问一句：这次采样出来的 `prev_sample` 在当前策略下有多大概率

对 diffusion 来说，一整条轨迹生成过程被拆成很多步 transition。  
对 RL 来说，每一步 transition 都是一个可求 `log_prob` 的动作。

---

## 7. `forward_train_rl` 的职责

这个函数不是直接输出最终 loss。  
它更像“采样经验 + 计算优势”的阶段。

它输出的是：

- `all_diffusion_output`
- `advantages`
- `reward`
- `sub_rewards`

也就是说，它干的是：

1. 生成轨迹
2. 评估轨迹
3. 形成训练信号

真正把这些信号变成 `loss` 的，是下一步 `get_rlloss`。

---

## 8. 逐步拆 `forward_train_rl`

### 8.1 设置 denoising 时间表

代码：

```python
self.diffusionrl_scheduler.set_timesteps(1000, device)
step_num = 10
step_ratio = 20 / step_num
roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1]
```

直觉：

- 训练里不是跑很长 diffusion 链
- 而是只跑一个很短的截断链

所以这不是标准“1000 步慢慢去噪”，而是“在很少的关键步上快速 refinement”。

这正是 truncated diffusion。

### 8.2 构造初始候选轨迹

代码：

```python
plan_anchor = self.plan_anchor.unsqueeze(0).unsqueeze(0).repeat(bs, num_groups, 1, 1, 1)
plan_anchor = plan_anchor.view(bs, num_groups * self.ego_fut_mode, ...)
```

如果：

- `self.plan_anchor.shape = [20, 8, 2]`
- `num_groups = 4`

那么最后就是：

- `[B, 80, 8, 2]`

这 80 条轨迹不是 80 个完全不同语义，而是：

- 20 个驾驶意图 anchor
- 每个意图复制 4 次做探索

### 8.3 先归一化，再加截断噪声

代码：

```python
diffusion_output = self.norm_odo(plan_anchor)
noise = torch.randn(diffusion_output.shape, device=device)
trunc_timesteps = torch.ones((bs,), ...) * 8
diffusion_output = self.diffusionrl_scheduler.add_noise(...)
```

这里的 `diffusion_output` 就是初始 `x_t`。

它的性质很重要：

- 不是纯随机噪声
- 也不是干净 anchor
- 而是“anchor 附近的带噪轨迹”

这和从白噪声开始采样非常不一样，因为它一开始就带驾驶意图先验。

---

## 9. 每一步 denoising 到底干了什么

循环里的每一步都可以拆成下面 7 小步。

### 9.1 `clamp` 当前样本

```python
x_boxes = torch.clamp(diffusion_output, min=-1, max=1)
```

原因：

- 扩散空间是归一化后的轨迹空间
- 防止数值爆掉

### 9.2 还原真实坐标

```python
noisy_traj_points = self.denorm_odo(x_boxes)
```

现在 `noisy_traj_points` 的 shape 还是：

- `[B, 80, 8, 2]`

但语义已经从“归一化坐标”变回“真实轨迹坐标”。

为什么要还原：

- 地图采样和位置编码都需要真实几何尺度

### 9.3 对 noisy trajectory 做位置编码

```python
traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
traj_pos_embed = traj_pos_embed.flatten(-2)
traj_feature = self.plan_anchor_encoder(traj_pos_embed)
```

这里发生了两件事：

1. 把每个轨迹点的几何位置编码成高维正余弦表示
2. 再映射到 `d_model=256`

直觉：

- 网络不直接吃原始 `(x, y)`
- 而是吃一个更适合表达相对位置和周期结构的 embedding

### 9.4 编码当前时间步

```python
time_embed = self.time_mlp(timesteps).view(bs, 1, -1)
```

扩散里时间步不是普通整数标签，而是很关键的条件：

- 早期去噪该做粗修正
- 后期去噪该做细修正

### 9.5 调用 `diff_decoder`

```python
poses_reg_list, poses_cls_list = self.diff_decoder(...)
poses_reg = poses_reg_list[-1]
```

它会综合：

- noisy trajectory 的位置特征
- BEV 稠密地图
- agents_query
- ego_query
- time_embed

输出：

- refined trajectory

### 9.6 取出模型预测的 `x_start`

```python
x_start = poses_reg[..., :2]
x_start = self.norm_odo(x_start)
```

注意：

- `poses_reg` 里有 `(x, y, heading)`
- 真正送 scheduler 的只取前两维 `(x, y)`

这再次证明：

- diffusion 主空间里真正被预测的是 `xy`
- heading 不是扩散主变量

### 9.7 scheduler 回推一步

```python
prev_sample, log_prob, _ = self.diffusionrl_scheduler.step(
    model_output=x_start,
    timestep=k,
    sample=diffusion_output,
    eta=eta,
)
```

此时：

- `sample=diffusion_output` 是当前 `x_t`
- `model_output=x_start` 是模型预测的 `x_0`
- `prev_sample` 是下一步 `x_{t-1}`

然后：

```python
diffusion_output = prev_sample
all_log_probs.append(log_prob)
all_diffusion_output.append(prev_sample)
```

这就把整条 denoising chain 存下来了。

---

## 10. 为什么要保存整条 `all_diffusion_output`

这是后面 `get_rlloss` 必须要用的。

因为第二阶段不是重新随便采一条链，而是：

- 沿着第一次采出来的旧链
- 重新计算当前策略对这条链每一步 transition 的 log-prob

所以这里保存的不是普通中间变量，而是 RL 的“轨迹经验”。

---

## 11. 从最终 `xy` 到可打分轨迹

循环结束后：

```python
diffusion_output = self.denorm_odo(diffusion_output)
diffusion_output = self.bezier_xyyaw(diffusion_output)
```

此时从：

- `[B, 80, 8, 2]`

变成：

- `[B, 80, 8, 3]`

第三维是用 Bézier 切线方向恢复出来的 yaw。

为什么必须这样做：

- PDM simulator / scorer 需要完整姿态
- 单有 `xy` 不够

---

## 12. reward 是怎么来的

接着代码会把：

- 模型候选轨迹
- `targets["trajectory"]`

拼起来：

```python
target_traj = targets['trajectory'].unsqueeze(1)
diffusion_output_with_gt = torch.cat((diffusion_output, target_traj), dim=1)
```

然后送进：

```python
reward_group, metric_cache, sub_rewards_group = self.get_pdm_score_para(...)
```

### 12.1 `get_pdm_score_para` 里面发生了什么

它会：

1. 把每个 batch 样本提交给 `_pdm_worker`
2. 子进程加载 `metric_cache.pkl`
3. 调 `pdm_score_para(...)`
4. 用 `PDMSimulator` 跑车辆动力学
5. 用 `PDMScorer` 计算闭环分数

所以 reward 不是“网络内部学出来的”，而是“外部模拟器打回来的”。

---

## 13. advantage 的组装逻辑

这部分是这段代码最核心的 RL 设计。

### 13.1 先 reshape

```python
reward_group = reward_group.view(bs, num_groups, self.ego_fut_mode)
```

如果：

- `num_groups = 4`
- `ego_fut_mode = 20`

那么每个样本是：

- 4 个 group
- 每个 group 对应 20 个 anchor

也可以反过来理解为：

- 对每个 anchor，都采了多个探索样本

### 13.2 组内标准化

```python
mean_grouped_rewards = reward_group.mean(dim=1)
std_grouped_rewards = reward_group.std(dim=1)
advantages = (reward_group - mean_grouped_rewards.unsqueeze(1)) / (std_grouped_rewards.unsqueeze(1) + 1e-4)
```

这里等价于：

`A = (r - mean_same_anchor_group) / std_same_anchor_group`

它的直觉是：

- 不同驾驶意图之间不应该直接比大小
- 只在同一类意图附近的探索之间比较谁更好

这正是论文里的 `intra-anchor GRPO`。

### 13.3 正样本截断

```python
mask_positive = (reward_group > (reward_gt - 1e-6))
advantages = advantages.clamp(min=0) * mask_positive.float()
```

含义：

- 只有比参考轨迹更好的样本，才保留正优势
- 负优势直接清掉

这是非常强的工程 bias：

- 训练重点不是“惩罚所有坏样本”
- 而是“强化那些真的比参考更好的样本”

### 13.4 安全指标再做一次硬过滤

```python
if k == 'no_collision' or k == 'drivable_area':
    zero_mask = (v != 1)
    advantages = torch.where(zero_mask, torch.full_like(advantages, -1.0), advantages)
```

意思是：

- 哪怕 final reward 看起来还行
- 只要碰撞或越界不达标
- advantage 直接打坏

这体现了自动驾驶里很重要的一个原则：

- 安全硬约束优先于一切软指标

### 13.5 时间折扣

```python
discount = torch.tensor([0.8 ** (step_num - i - 1) for i in range(step_num)])
advantages = advantages * discount
```

意思是：

- 越后面的 denoising step 权重越大
- 越前面的高噪声阶段权重越小

直觉上非常合理，因为：

- 早期步很粗，随机性更强
- 后期步更接近最终决策

---

## 14. `forward_train_rl` 最终输出的本质

返回值：

```python
{
  "all_diffusion_output": all_diffusion_output,
  "advantages": advantages,
  "reward": reward,
  "sub_rewards": sub_rewards_mean
}
```

这不是最终 loss，而是一包“训练素材”：

- 采样链
- 每步优势
- 日志奖励

下一步 `get_rlloss` 再把它们变成真正反传的标量损失。

---

## 15. 这一段最值得你记住的开发者思维

如果你站在作者角度，会发现这段代码不是在做“标准 diffusion 训练”，而是在做：

1. 用 anchor 约束动作空间
2. 用乘性噪声鼓励局部探索
3. 用短链 diffusion 快速 refinement
4. 用 closed-loop PDM score 判断轨迹好坏
5. 用 advantage 把“好轨迹”转成策略梯度信号

也就是说，它把 diffusion 变成了一个“可探索、可评分、可强化”的生成式规划策略。
