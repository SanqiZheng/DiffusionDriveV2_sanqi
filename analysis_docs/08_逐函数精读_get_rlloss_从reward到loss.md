# 08. 逐函数精读：`get_rlloss`，从 reward 到 loss

主分析对象：

- [diffusiondrivev2_model_rl.py](/home/yihang/Downloads/CodeReference/diff_based/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py)

如果前两份逐函数文档分别回答了：

- 条件是怎么准备好的
- 轨迹是怎么采样出来的

那这一份就专门回答最后一个关键问题：

“reward 到底是怎么被变成可以反向传播的 loss 的？”

这也是初学者最容易卡住的一步。

---

## 1. `get_rlloss` 的输入不是普通 batch，而是“旧策略经验”

函数一开始：

```python
old_diffusion_output = old_pred['all_diffusion_output']
advantages = old_pred['advantages']
chains = old_diffusion_output[...,:-1]
chains_prev = old_diffusion_output[...,1:]
```

这里最重要的不是张量 shape，而是语义：

- `old_diffusion_output`
  是第一次 `forward_train_rl` 采出来的完整 denoising chain
- `advantages`
  是第一次 rollout 后由 PDM reward 算出来的优势

所以这一步不是“重新随机采样”，而是：

- 固定旧轨迹链
- 重新评估当前策略对这条旧链的概率

这和 RL 里“基于采样轨迹做策略更新”是同一个思想。

---

## 2. 这一步为什么不直接对 reward 反传

因为 reward 来自：

- `PDM simulator`
- `PDM scorer`

这些外部评测组件不是可微的。

也就是说：

- 你不能把 reward 当普通神经网络 loss，直接链式求导

所以只能走策略梯度路线：

1. 采样动作
2. 得到回报
3. 用 `log_prob * advantage` 估计梯度

这就是 `get_rlloss` 存在的根本原因。

---

## 3. 先重建每一步的当前输入

循环里每个 step 都会做：

```python
diffusion_output = chains[..., i]
x_boxes = torch.clamp(diffusion_output, min=-1, max=1)
noisy_traj_points = self.denorm_odo(x_boxes)
traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
traj_feature = self.plan_anchor_encoder(...)
time_embed = self.time_mlp(...)
poses_reg_list, poses_cls_list = self.diff_decoder(...)
```

这段和 `forward_train_rl` 很像，因为它本质上是在重新做一遍：

- “当前看到的 noisy sample 是什么”
- “当前网络会把它去噪成什么”

区别在于：

- 这里不是为了重新 rollout
- 而是为了重新算当前策略对旧 transition 的概率

---

## 4. 最关键的一行：`prev_sample=chains_prev[..., i]`

代码：

```python
_, log_prob, _ = self.diffusionrl_scheduler.step(
    model_output=x_start,
    timestep=k,
    sample=diffusion_output,
    eta=eta,
    prev_sample=chains_prev[...,i]
)
```

这句特别关键。

### 4.1 如果不传 `prev_sample`

scheduler 会：

- 根据当前策略分布重新随机采样一个新 `x_{t-1}`

### 4.2 现在传了 `prev_sample`

它就不会重新采样，而是改成：

- 用当前策略分布去评估“旧链里真实发生的那个 `x_{t-1}`”的对数概率

所以这里做的是：

`log pi_theta(old_action | old_state)`

也就是：

- 旧动作固定
- 当前策略评估它有多合理

这正是策略梯度更新所需要的量。

---

## 5. `all_log_probs` 的 shape 到底是什么

循环结束后：

```python
all_log_probs = torch.stack(all_log_probs, dim=-1)
per_token_logps = all_log_probs.view(bs, self.num_groups * self.ego_fut_mode, -1)
```

语义上就是：

- 第 1 维：batch
- 第 2 维：每条候选轨迹
- 第 3 维：每个 denoising step 的 log-prob

如果：

- `num_groups = 4`
- `ego_fut_mode = 20`
- `step_num = 10`

那么 shape 大致就是：

- `[B, 80, 10]`

每个元素表示：

- 第 `b` 个样本
- 第 `g` 条候选轨迹
- 在第 `t` 个去噪步骤
- 当前策略给旧 transition 的对数概率

---

## 6. 真正的策略梯度损失在哪里

代码：

```python
per_token_loss = -torch.exp(per_token_logps - per_token_logps.detach()) * advantages
```

这一句看起来怪，但它其实是一个很经典的工程写法。

### 6.1 为什么不是直接 `-log_prob * A`

如果写成最教科书的形式，会是：

`L = - log pi * A`

但作者这里用了：

`exp(logp - stopgrad(logp))`

原因是：

- 前向值稳定
- 数值上等于 1
- 但梯度仍然等价于对 `logp` 求导

可以这样理解：

`exp(x - x.detach())`

前向时：

- 等于 `exp(0) = 1`

反向时：

- 梯度还是从 `x` 这条路径走

所以它是一种“前向不放大，反向保留 score function 梯度”的 trick。

### 6.2 为什么要乘 `advantages`

因为 advantage 的意义是：

- 这条轨迹比组内平均好多少

所以：

- `A > 0`
  说明这个旧动作值得提高概率
- `A <= 0`
  说明不值得强化，或者已经被截断

---

## 7. `mask_nz` 为什么存在

代码：

```python
mask_nz = per_token_loss != 0
RL_loss_b = (per_token_loss * mask_nz).sum(dim=1) / mask_nz.sum(dim=1).clamp_min(1)
RL_loss_b = RL_loss_b.mean(dim=-1)
```

这是因为：

- 不是所有轨迹、所有 step 都有有效 advantage
- 很多样本会被 `mask_positive` 或安全约束截断成 0

所以作者只在“有有效训练信号的 token”上做平均。

开发者为什么这么做：

- 避免大量 0 信号把 loss 冲淡
- 让真正有价值的正样本梯度占主导

---

## 8. 为什么还要再加一个 IL loss

代码：

```python
IL_loss_b = torch.zeros_like(RL_loss_b)
target_traj = targets['trajectory'].unsqueeze(1).repeat(...)
for poses_reg_list in poses_reg_steps_list:
    for poses_reg in poses_reg_list:
        traj_l1 = F.l1_loss(poses_reg[...,:2], target_traj[...,:2], reduction='none')
        IL_loss_b += traj_l1.mean()
```

这说明作者并没有让模型完全脱离监督学习，而是保留了一个辅助 imitation 项。

### 8.1 直觉上为什么必须加

因为 diffusion + RL 这个组合很容易出现两个问题：

1. reward 方差大
2. 训练容易漂

IL loss 的作用像一个“轨道”：

- 不让模型完全乱飞
- 保证轨迹至少还像真实驾驶数据

### 8.2 为什么权重要动态切换

代码：

```python
has_positive = (advantages > 0).any(dim=2).any(dim=1)
il_weight = torch.where(has_positive == 0, 1.0, 0.1)
```

含义很清楚：

- 如果这批样本里根本没有正向 RL 信号
  那就主要靠监督学习
- 如果有正向 RL 信号
  那就让 RL 主导，只保留轻量 IL 稳定项

这是一种非常现实的训练策略。

开发者的思路不是“RL 替代 IL”，而是：

- RL 提供方向
- IL 提供稳定性

---

## 9. 总损失怎么组出来

最后：

```python
loss_b = RL_loss_b + il_weight * IL_loss_b
loss = loss_b.mean()
```

也就是：

`L_total = L_RL + lambda * L_IL`

其中 `lambda` 会按当前 batch 是否存在正优势样本自动切换。

---

## 10. 这一整段代码在 RL 视角下怎么理解

可以把 `get_rlloss` 翻译成一句话：

“沿着旧策略采出来的 denoising 链，把每一步 `x_t -> x_{t-1}` 都看成一个连续动作，然后用当前策略重新评估这些旧动作的 log-prob，再用 rollout 阶段得到的 advantage 对这些动作做加权，从而提高高质量轨迹的生成概率。”

这就是为什么：

- reward 虽然不可微
- 但最终仍然能训练网络

因为真正被反传的不是 reward 本身，而是：

- “提高好动作概率”的策略梯度 surrogate objective

---

## 11. 这个函数最值得你真正吃透的 5 个点

1. `old_pred` 不是普通中间结果，而是 rollout 经验包。
2. `prev_sample=chains_prev[..., i]` 是把旧动作固定下来、评估当前策略概率的关键。
3. `per_token_logps` 的每个元素都对应 diffusion 一步 transition 的 log-prob。
4. `per_token_loss` 本质上是策略梯度的工程实现，不是奇怪的黑魔法。
5. `IL_loss` 不是多余的，它是 RL 训练稳定器。

---

## 12. 你现在应该怎样对照源码读这段

最推荐的方式不是从上往下机械读，而是带着下面 4 个问题读：

1. 这里的“状态”是什么。
2. 这里的“动作”是什么。
3. 这里的“奖励信号”是从哪里来的。
4. 这里的“概率”是怎么计算出来的。

把这 4 个问题一一对上，你就会发现：

- 这个项目虽然表面是 diffusion 模型
- 但在训练时已经被作者改造成了一个连续动作的生成式 RL policy

这正是 DiffusionDriveV2 最有意思的地方。
