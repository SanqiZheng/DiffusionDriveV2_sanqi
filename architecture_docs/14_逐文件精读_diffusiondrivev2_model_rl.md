# 14 逐文件精读：`diffusiondrivev2_model_rl.py`

本文对应源码：

- `navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py`

这份文件是 Stage I 生成器 + RL 训练主链路的核心实现。它不是一个“普通模型文件”，而是同时承担了：

1. 场景条件编码后的顶层组装
2. anchor-based truncated diffusion 轨迹生成
3. PDM reward 桥接
4. diffusion-policy 风格的 RL loss

如果你只想先吃透一个文件，就先把这份文件读透。

---

## 1. 读这个文件前，先记住 4 个全局事实

### 1.1 这个文件里的扩散模型学的不是噪声，而是 `x_start`

源码位置：

- `DDIMScheduler(... prediction_type="sample")`
- `DDIMScheduler_with_logprob(... prediction_type="sample")`
- 对应 `TrajectoryHead.__init__`

这意味着模型预测的是“干净样本本身”，而不是 `epsilon`。

在这个项目里，“干净样本”具体就是：

- 归一化后的未来轨迹 `xy`

也就是说，它学习的是：

- 不是 `noise`
- 不是 `velocity`
- 不是 `score`
- 而是 `x_start`

### 1.2 扩散对象本质上是未来 8 个轨迹点的 `xy`

虽然中间很多地方张量会写成 `(..., 8, 3)`，但主扩散链真正反复处理的是：

- `x_start = poses_reg[..., :2]`

最后的 `yaw` 不是主扩散目标，而是通过 `bezier_xyyaw(...)` 从轨迹几何中推出来。

### 1.3 候选轨迹来自“20 个 anchor mode x 多个 group”

你会在 `TrajectoryHead.__init__` 里看到：

- `self.ego_fut_mode = 20`
- `self.plan_anchor` shape 为 `(20, 8, 2)`

你会在 RL config 里看到：

- `num_groups = 4`

所以训练时总候选数是：

```text
G_all = num_groups * ego_fut_mode = 4 * 20 = 80
```

### 1.4 训练不是一遍 forward，而是“两遍式”

顶层 `V2TransfuserModel.forward` 会先：

1. `torch.no_grad()` 下跑一次 `_trajectory_head(...)`
2. 拿到 rollout 链和 reward
3. 再把这条链重放一遍，计算带梯度的 `log_prob`

这就是 RL 阶段最重要的实现细节。

---

## 2. 文件结构总览

最关键的定义如下：

- `_pairwise_subscores`：PDM 子分数拆解
- `_pairwise_scores`：GT-vs-candidate 最终 reward
- `_pdm_worker` / `_init_pool`：多进程 PDM 评估
- `V2TransfuserModel`：顶层场景编码 + query 组装
- `AgentHead`：周围目标框预测
- `CustomTransformerDecoderLayer`：扩散去噪的核心 layer
- `DDIMScheduler_with_logprob`：返回 `log_prob` 的 scheduler
- `TrajectoryHead`：生成、reward、RL loss 主体
- `bezier_xyyaw`：从 `xy` 恢复 `yaw`

建议阅读顺序：

1. `V2TransfuserModel`
2. `CustomTransformerDecoderLayer`
3. `DDIMScheduler_with_logprob`
4. `TrajectoryHead.__init__`
5. `forward_train_rl`
6. `get_rlloss`
7. `forward_test_rl`
8. `bezier_xyyaw`

---

## 3. 顶层模型：`V2TransfuserModel`

源码位置：

- `class V2TransfuserModel` 附近
- `__init__`
- `forward`

### 3.1 它的职责是什么

这个类不直接生成轨迹，而是负责把 backbone 输出整理成适合“轨迹生成头”使用的条件表示。

你可以把它理解成：

```text
camera / lidar / ego status
    -> backbone
    -> BEV token + query token
    -> trajectory_query / agents_query
    -> TrajectoryHead
```

### 3.2 最关键的成员

- `_query_splits = [1, num_bounding_boxes]`
- `_backbone`
- `_keyval_embedding`
- `_query_embedding`
- `_bev_downscale`
- `_status_encoding`
- `_tf_decoder`
- `_agent_head`
- `_trajectory_head`

这里最重要的设计是：

- 先用 backbone 把场景变成 memory
- 再用 query decoder 分出 ego query 和 agent query
- 最后才交给轨迹生成器

### 3.3 一次 forward 的 shape 主线

默认配置下，可以先把 shape 理成下面这样：

```text
camera_feature: [B, 3, 256, 1024]
lidar_feature:  [B, 1, 256, 256]
status_feature: [B, 8]

backbone 输出:
bev_feature_upscale: [B, 64, H_bev, W_bev]
bev_feature:         [B, 512, 8, 8]

下采样并展平后:
bev_feature token:   [B, 64, 256]
status token:        [B, 1, 256]
keyval:              [B, 65, 256]

query embedding:
query:               [B, 31, 256]

decoder 输出:
trajectory_query:    [B, 1, 256]
agents_query:        [B, 30, 256]
```

### 3.4 学习版代码摘录

```python
def forward(self, features, targets=None, eta=0.0, metric_cache=None, cal_pdm=True, token=None):
    # 1. 取出三路输入
    camera_feature = features["camera_feature"]   # [B, 3, 256, 1024]
    lidar_feature = features["lidar_feature"]     # [B, 1, 256, 256]
    status_feature = features["status_feature"]   # [B, 8]

    # 2. backbone 同时输出高分辨率 BEV 和低分辨率 fused feature
    bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)

    # 3. 低分辨率 BEV -> 64 个 token
    bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1).permute(0, 2, 1)

    # 4. ego 状态也编码成一个 token
    status_encoding = self._status_encoding(status_feature)   # [B, 256]
    keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)  # [B, 65, 256]

    # 5. 用固定 query 去“读”场景 memory
    query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)    # [B, 31, 256]
    query_out = self._tf_decoder(query, keyval)

    # 6. 拆成 1 个 ego 规划 query + 30 个 agent query
    trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

    # 7. 轨迹头真正负责生成轨迹和 RL loss
    pred = self._trajectory_head(...)
```

### 3.5 为什么这样设计

因为作者不想让扩散头直接吃原始图像 / 原始点云，而是先把场景压成：

- 一份全局 memory
- 一份 ego query
- 一份 agent query

这样后面的轨迹生成器可以专注于“沿着候选轨迹去查询地图和车辆上下文”，而不是重新做一遍感知融合。

---

## 4. 去噪核心层：`CustomTransformerDecoderLayer`

源码位置：

- `class CustomTransformerDecoderLayer`

### 4.1 这一层做了什么

这一层是“轨迹去噪器”的核心，它按顺序注入四类条件：

1. 地图条件：`cross_bev_attention`
2. 周围车条件：`cross_agent_attention`
3. ego 全局条件：`cross_ego_attention`
4. 扩散时间条件：`time_modulation`

### 4.2 为什么它很关键

这就是论文里“条件生成”在代码中的真正落点。

条件信息不是一次性 `concat` 进去，而是按语义逐层注入：

- 地图负责“这条轨迹经过的位置有什么”
- agent query 负责“周围车是什么状态”
- ego query 负责“全局驾驶意图是什么”
- time embedding 负责“当前是在第几步去噪”

### 4.3 输入输出 shape

```text
traj_feature:      [B, G_all, 256]
noisy_traj_points: [B, G_all, 8, 2]
bev_feature:       [B, 64, H_bev, W_bev] 或投影后的 256 通道版本
agents_query:      [B, 30, 256]
ego_query:         [B, 1, 256]
time_embed:        [B, 1, 256]

输出:
poses_reg:         [B, G_all, 8, 3]
poses_cls:         [B, num_groups, 20]
```

### 4.4 学习版代码摘录

```python
def forward(self, traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
            agents_query, ego_query, time_embed, status_encoding, global_img=None):
    # 1. 让“当前 noisy 轨迹上的每个点”去 BEV 地图上采样局部特征
    traj_feature = self.cross_bev_attention(
        traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape
    )

    # 2. 再读周围 agent query
    traj_feature = traj_feature + self.dropout(
        self.cross_agent_attention(traj_feature, agents_query, agents_query)[0]
    )
    traj_feature = self.norm1(traj_feature)

    # 3. 再读 ego query
    traj_feature = traj_feature + self.dropout1(
        self.cross_ego_attention(traj_feature, ego_query, ego_query)[0]
    )
    traj_feature = self.norm2(traj_feature)

    # 4. FFN + 时间步调制
    traj_feature = self.norm3(self.ffn(traj_feature))
    traj_feature = self.time_modulation(traj_feature, time_embed)

    # 5. 输出每个 mode 的轨迹增量
    poses_reg, poses_cls = self.task_decoder(...)
    poses_reg[..., :2] = poses_reg[..., :2] + noisy_traj_points
```

### 4.5 一个容易漏掉的点

`GridSampleCrossBEVAttention` 不是对整张图做全局 attention，而是：

- 把轨迹点坐标归一化后
- 直接去 BEV feature 图上 `grid_sample`

也就是“让轨迹去地图上取值”，这比全局 attention 更贴合规划问题。

---

## 5. `DDIMScheduler_with_logprob`：RL 能成立的关键

源码位置：

- `class DDIMScheduler_with_logprob`

### 5.1 它相比普通 DDIM 多做了什么

多做了两件事：

1. 支持返回 `log_prob`
2. 自定义了乘性噪声采样

### 5.2 为什么 `log_prob` 很重要

因为 RL 训练并不是直接对 reward 反传，而是要先知道：

- 当前策略在每一步产生这次 transition 的概率有多大

这个概率就被写成了：

- `log_prob`

### 5.3 为什么说它学的是 `x_start`

因为代码里写的是：

```python
elif self.config.prediction_type == "sample":
    pred_original_sample = model_output
```

也就是把模型输出直接当成预测的干净样本。

### 5.4 学习版代码摘录

```python
if self.config.prediction_type == "sample":
    # 模型直接输出 x_0，而不是噪声 epsilon
    pred_original_sample = model_output
    pred_epsilon = (sample - alpha_t.sqrt() * pred_original_sample) / beta_t.sqrt()

# 先算无噪声均值
prev_sample_mean = alpha_prev.sqrt() * pred_original_sample + pred_sample_direction

if prev_sample is None:
    # 项目自定义：不是纯加性噪声，而是乘性噪声
    variance_noise_mul = ...
    prev_sample = prev_sample_mean * variance_noise_mul

# 额外返回 log_prob，供 RL loss 使用
log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * std**2) - torch.log(std)
```

### 5.5 论文创新点的落地位置

如果你要找“scale-adaptive multiplicative noise”在代码里真正落在哪里：

- 就看这里的 `variance_noise_mul`

---

## 6. `TrajectoryHead.__init__`：生成器主脑

源码位置：

- `class TrajectoryHead`
- `__init__`

### 6.1 最重要的成员

- `self.plan_anchor`
- `self.diffusion_scheduler`
- `self.diffusionrl_scheduler`
- `self.plan_anchor_encoder`
- `self.time_mlp`
- `self.diff_decoder`
- `self._pdm_pool`

### 6.2 这些成员分别在干什么

- `plan_anchor`
  - 20 个 KMeans 轨迹原型
  - shape 是 `(20, 8, 2)`
- `plan_anchor_encoder`
  - 把轨迹点位置编码后投影到 `d_model=256`
- `time_mlp`
  - 把扩散步 `t` 编码成时间条件
- `diff_decoder`
  - 真正执行去噪 refinement
- `diffusionrl_scheduler`
  - 训练时既采样又输出 `log_prob`
- `_pdm_pool`
  - 并行调用 PDM scorer 计算 reward

### 6.3 这里体现的工程思想

`TrajectoryHead` 不是一个单纯“head”，它其实同时承担了：

- sampler
- denoiser
- reward bridge
- RL objective bridge

---

## 7. `forward_train_rl`：第一遍 rollout

源码位置：

- `TrajectoryHead.forward_train_rl`

这是整份文件最值得逐行精读的函数。

### 7.1 先把 shape 写在纸上

训练时默认：

```text
num_groups = 4
ego_fut_mode = 20
step_num = 10
```

所以：

```text
plan_anchor:                    [20, 8, 2]
repeat 后:                      [B, 4, 20, 8, 2]
flatten 后:                     [B, 80, 8, 2]
all_log_probs 最后 reshape 后:   [B, 4, 20, 10]
```

### 7.2 一次 rollout 的真正流程

```text
anchor
  -> 加截断噪声
  -> 当前 noisy 轨迹编码成 traj_feature
  -> diff_decoder 预测 x_start
  -> scheduler.step 得到 x_{t-1} 和 log_prob
  -> 循环多步
  -> 最终轨迹送进 PDM 得 reward
```

### 7.3 学习版代码摘录

```python
def forward_train_rl(...):
    step_num = 10
    num_groups = self.num_groups   # 默认 4

    # 1. 20 个 anchor 复制成多个 group，形成候选初始族
    plan_anchor = self.plan_anchor.unsqueeze(0).unsqueeze(0).repeat(bs, num_groups, 1, 1, 1)
    plan_anchor = plan_anchor.view(bs, num_groups * self.ego_fut_mode, 8, 2)  # [B, 80, 8, 2]

    # 2. 先把 anchor 变成 truncated noisy sample，而不是从纯高斯噪声开始
    diffusion_output = self.norm_odo(plan_anchor)
    diffusion_output = self.diffusionrl_scheduler.add_noise(
        original_samples=diffusion_output,
        noise=torch.randn_like(diffusion_output),
        timesteps=torch.ones((bs,), device=device, dtype=torch.long) * 8,
    )

    for k in roll_timesteps:
        # 3. 当前 noisy sample -> 轨迹点
        noisy_traj_points = self.denorm_odo(torch.clamp(diffusion_output, min=-1, max=1))

        # 4. 当前几何本身被重新编码成 query feature
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64).flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed).view(bs, -1, 256)

        # 5. 当前时间步编码
        time_embed = self.time_mlp(k.expand(bs)).view(bs, 1, -1)

        # 6. 去噪器预测 x_start
        poses_reg_list, poses_cls_list = self.diff_decoder(...)
        x_start = self.norm_odo(poses_reg_list[-1][..., :2])

        # 7. scheduler 负责从 x_t 走到 x_{t-1}，并记录 log_prob
        diffusion_output, log_prob, _ = self.diffusionrl_scheduler.step(...)
```

### 7.4 reward 是怎么来的

最终轨迹会被：

1. `denorm_odo(...)`
2. `bezier_xyyaw(...)`
3. 与 GT 拼接
4. 送入 `get_pdm_score_para(...)`

得到：

- `reward_group`
- `sub_rewards_group`

### 7.5 advantage 是怎么做的

关键逻辑不是普通的 `reward - baseline`，而是更贴近论文里的“按 anchor/group 做分组归一化”：

1. 先 reshape 成 `(B, num_groups, ego_fut_mode)`
2. 沿 `group` 维度做均值和标准差归一化
3. 只保留“好于 GT”的正样本
4. 若 `no_collision` 或 `drivable_area` 不满足，直接强压成坏样本

这部分就是论文里 RL 训练思想在代码中的主要落点。

### 7.6 一个值得打问号的工程点

代码里：

- 加噪 timestep 用的是 `8`
- rollout timestep 是由 `step_num=10` 推出的序列

这和最教科书式的 DDIM 推导并不完全一致，更像是作者做过工程化试验后的版本。读代码时要接受“它是有效实现，不一定是最标准公式展开”。

---

## 8. `get_rlloss`：第二遍带梯度重放

源码位置：

- `TrajectoryHead.get_rlloss`

### 8.1 它在做什么

第一遍 rollout 只负责：

- 采样
- 打分
- 记下轨迹链

第二遍 `get_rlloss` 才真正负责：

- 重放这条链
- 计算每一步在当前策略下的 `log_prob`
- 按 advantage 做 policy gradient

### 8.2 为什么必须这样做

因为 reward 来自 PDM 外部模拟器，不可微。

所以不能写成：

```text
loss = - reward
```

而要写成：

```text
loss = - log_prob * advantage
```

### 8.3 学习版代码摘录

```python
old_diffusion_output = old_pred["all_diffusion_output"]
advantages = old_pred["advantages"]

# chains[..., i] 是 rollout 时第 i 步的 x_t
# chains_prev[..., i] 是 rollout 时第 i 步对应的 x_{t-1}
chains = old_diffusion_output[..., :-1]
chains_prev = old_diffusion_output[..., 1:]

for i, k in enumerate(roll_timesteps):
    diffusion_output = chains[..., i]
    ...
    x_start = self.norm_odo(poses_reg[..., :2])

    # 这里不再重新采样，而是把 rollout 时真实得到的 prev_sample 喂回来
    _, log_prob, _ = self.diffusionrl_scheduler.step(
        model_output=x_start,
        timestep=k,
        sample=diffusion_output,
        eta=eta,
        prev_sample=chains_prev[..., i],
    )
```

### 8.4 最关键的一行

```python
per_token_loss = -torch.exp(per_token_logps - per_token_logps.detach()) * advantages
```

这行第一次看会很怪，但直觉上它在做的是：

- 前向值看起来像 `-advantages`
- 梯度里保留 `d(log_prob)`

所以本质上还是 REINFORCE / policy gradient 的写法。

### 8.5 为什么还要加 imitation loss

后面代码又加了：

- `IL_loss_b`

作用是：

- 防止 RL 样本太稀或正样本太少时训练完全漂掉
- 给策略一个基本轨迹几何约束

也就是说，这里不是纯 RL，而是：

```text
RL loss + 小权重 IL regularization
```

### 8.6 一个值得注意的实现点

`IL_loss_b` 这个名字看起来像“逐 batch”，但内部 `traj_l1.mean()` 是标量，会被广播到整批。它不一定会坏，但确实值得你后续二次开发时回头仔细检查。

---

## 9. `forward_test_rl` 与 `bezier_xyyaw`

源码位置：

- `TrajectoryHead.forward_test_rl`
- `TrajectoryHead.bezier_xyyaw`

### 9.1 推理时比训练简单很多

推理分支里：

- `step_num = 2`
- `eta = 0.0`

也就是更偏确定性的少步采样。

### 9.2 为什么最后还要做 `bezier_xyyaw`

因为主去噪链学习的是 `xy`，而评测器 / 轨迹对象通常需要 `(x, y, yaw)`。

`bezier_xyyaw` 的做法是：

1. 在最前面补一个原点 `(0, 0)`
2. 把 8 个未来点看成一条 Bézier 曲线控制点序列
3. 用一阶导数方向求切线
4. 用 `atan2(dy, dx)` 得到 `yaw`

它的优点是：

- heading 由轨迹几何导出
- 通常比逐点独立回归的 heading 更平滑

---

## 10. 这份文件里你一定要记住的 8 个点

1. 这个文件里的扩散模型学习的是 `x_start`，不是噪声。
2. 真正被扩散和优化的是未来 `xy`，不是完整 `xyyaw`。
3. `plan_anchor` 是 `(20, 8, 2)`，它把多模态轨迹空间显式离散化了。
4. 条件注入不是一次拼接，而是通过 BEV/agent/ego/time 四条支路分层注入。
5. 第一遍 forward 负责 rollout 和 reward，第二遍 forward 才负责带梯度的 RL loss。
6. reward 不是 learned critic，而是 PDM simulator / scorer 给的闭环分数。
7. `DDIMScheduler_with_logprob` 是 RL 能跑起来的关键桥梁。
8. `bezier_xyyaw` 说明这个项目对 `yaw` 采用了“几何恢复”而不是“主链直接扩散”。

---

## 11. 读完这个文件后，下一步该读什么

建议直接接着读：

1. `navsim/agents/diffusiondrivev2/transfuser_features.py`
2. `navsim/planning/script/run_training.py`
3. `navsim/planning/training/dataset.py`
4. `navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel.py`

原因很简单：

- 先弄清输入是什么
- 再弄清训练入口怎么把它跑起来
- 最后再读 selector，理解 Stage II 如何接在生成器后面
