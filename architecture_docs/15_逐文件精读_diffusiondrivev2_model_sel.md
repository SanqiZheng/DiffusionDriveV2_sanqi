# 15 逐文件精读：`diffusiondrivev2_model_sel.py`

本文对应源码：

- `navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel.py`

如果说 `diffusiondrivev2_model_rl.py` 是“生成器 + RL”，那这份文件就是：

- 冻结生成器
- 在大量候选轨迹上训练 coarse-to-fine 选择器

它对应论文里的 Stage II。

---

## 1. 先给这份文件一个一句话定义

这不是一个纯 scorer 文件，而是：

```text
冻结的候选轨迹生成器
    + 轨迹条件编码器
    + coarse scorer
    + top-k 选择
    + fine scorer
```

也就是说，selector 训练时仍然要先跑生成器，因为它必须先有候选轨迹可选。

---

## 2. 和 RL 版最本质的差别

### 2.1 顶层骨架基本一样

文件开头的：

- `V2TransfuserModel`
- `AgentHead`
- `CustomTransformerDecoderLayer`
- `DDIMScheduler_with_logprob`

这些骨架和 RL 版高度相似。

原因是：

- selector 仍然依赖同一套场景编码器和生成器

### 2.2 真正新增的是“轨迹打分器分支”

最关键的新模块都在 `TrajectoryHead.__init__` 里：

- `plan_anchor_scorer_encoder`
- `scorer_decoder`
- `fine_scorer_decoder`
- `NC_head`
- `EP_head`
- `DAC_head`
- `TTC_head`
- `C_head`
- 对应的一套 `fine_*_head`

直觉上就是：

- 先用生成器造轨迹
- 再把轨迹本身变成 scorer query
- 最后用多个子分数头去预测 PDM 的子指标

---

## 3. 顶层 `V2TransfuserModel`：和 RL 版几乎同构

源码位置：

- `class V2TransfuserModel`

### 3.1 这一层不要读太久

这部分和 RL 版的逻辑几乎一样：

1. backbone 编码相机和 LiDAR
2. 把低分辨率 BEV 变成 token
3. 加入 ego status token
4. query decoder 生成 `trajectory_query` 和 `agents_query`
5. 交给 `_trajectory_head`

所以这里可以把注意力集中在：

- `TrajectoryHead`

---

## 4. `TrajectoryHead.__init__`：Selector 的真正核心

源码位置：

- `class TrajectoryHead`
- `__init__`

### 4.1 先看两类 encoder

这一部分同时保留了两套轨迹编码器：

- `plan_anchor_encoder`
  - 给生成器使用
  - 输出 `d_model=256`
- `plan_anchor_scorer_encoder`
  - 给 scorer 使用
  - 输入维度更大，输出 `512`

为什么 scorer 编码器更重？

因为它不仅看 `xy`，还会看 `heading` 编码，并且打分阶段通常需要更强的判别表达。

### 4.2 coarse scorer 与 fine scorer

你会看到两套完整 scorer：

- `scorer_decoder`
  - 粗筛
- `fine_scorer_decoder`
  - 细排

并且每套 scorer 都不是只输出一个总分，而是分别预测：

- `no_collision`
- `progress`
- `drivable_area`
- `ttc`
- `comfort`

然后再按固定公式拼成最终 reward proxy。

### 4.3 为什么这样设计

因为直接学一个“总分”往往不稳定，也不利于解释。

而拆成多个子指标后：

1. 训练监督更密
2. 每个 head 语义更清楚
3. 更容易复用 PDM 的结构先验

---

## 5. `ScorerTransformerDecoderLayer`：selector 的条件注入位置

源码位置：

- `class ScorerTransformerDecoderLayer`

### 5.1 它和生成器 decoder 的差别

生成器 decoder 更强调“去噪并输出轨迹”。

scorer decoder 更强调“读场景并输出评分特征”。

所以这里：

- 不再输出 `poses_reg`
- 而是输出打分特征 `traj_feature`

### 5.2 条件是怎么注入的

仍然是四类信息，但组织方式略不同：

1. `cross_bev_attention`
2. `cross_agent_attention`
3. `self_attn`
4. `cross_ego_attention`
5. `ffn`

这说明 selector 也不是纯 MLP 打分，而是：

- 让每条候选轨迹再次回到 BEV / agent / ego 条件里重新读上下文

### 5.3 一个实现细节

这里把 scorer 隐层维度直接提到了：

- `tf_d_model = 512`
- `tf_num_head = 16`

也就是判别器比分生成器更“宽”，这通常是为了提升排序能力。

---

## 6. `_get_scorer_inputs`：候选轨迹是如何变成 scorer 输入的

源码位置：

- `TrajectoryHead._get_scorer_inputs`

这是整份 selector 文件里最值得单独读的一个函数。

### 6.1 输入输出语义

输入：

- `diffusion_output: (B, G_all, 8, 3)`

这里的候选轨迹已经是：

- denorm 后
- 并且补齐了 `yaw`

输出：

- `noisy_traj_points_xy`
- `traj_feature`
- `time_embed`（当前实现返回 `None`）

### 6.2 它具体做了什么

1. 先把轨迹重新归一化，再 clamp，再反归一化
2. 把 `xy` 做二维正弦位置编码
3. 把 `heading` 做一维正弦位置编码
4. 拼起来后送入 `plan_anchor_scorer_encoder`

所以 scorer 看见的并不是裸轨迹点，而是：

```text
xy 几何 + heading 几何 的高维位置编码
```

### 6.3 shape 变化

```text
diffusion_output:           [B, G_all, 8, 3]
xy embedding:               [B, G_all, 8 * 64]
heading embedding:          [B, G_all, 8 * 32]
concat 后:                  [B, G_all, 768]
scorer encoder 输出:        [B, G_all, 512]
```

### 6.4 学习版代码摘录

```python
def _get_scorer_inputs(self, diffusion_output, bs, ego_fut_mode):
    # 1. 轨迹重新压回训练时使用的归一化范围
    diffusion_output = self.norm_odo(diffusion_output)
    noisy_traj_points = self.denorm_odo(torch.clamp(diffusion_output, min=-1, max=1))

    # 2. xy 和 heading 分开编码
    noisy_traj_points_xy = noisy_traj_points[..., :2]
    traj_pos_embed = gen_sineembed_for_position(noisy_traj_points_xy, hidden_dim=64).flatten(-2)
    traj_heading_embed = gen_sineembed_for_position_1d(
        noisy_traj_points[..., 2], hidden_dim=32
    ).flatten(-2)

    # 3. 合成 scorer 特征
    traj_feature = torch.cat([traj_pos_embed, traj_heading_embed], dim=-1)
    traj_feature = self.plan_anchor_scorer_encoder(traj_feature).view(bs, ego_fut_mode, -1)
```

---

## 7. `_score_coarse`：粗筛打分器

源码位置：

- `TrajectoryHead._score_coarse`

### 7.1 它预测的不是一个分数，而是 5 类子分数

对应 head：

- `NC_head`
- `EP_head`
- `DAC_head`
- `TTC_head`
- `C_head`

对应语义：

- `no_collision`
- `progress`
- `drivable_area`
- `ttc`
- `comfort`

### 7.2 它的监督来自哪里

不是人工标签，而是：

- `sub_rewards_group`

也就是 PDM scorer 产出的子指标。

所以 coarse scorer 的训练本质上是：

- 用神经网络去拟合 PDM 子指标

### 7.3 loss 组成

- `BCE` 拟合多个子项
- `MarginRankingLoss` 约束 `progress` 排序关系

### 7.4 最终 coarse reward 不是直接监督值，而是一个手工拼接公式

```python
final_coarse_reward =
    sigmoid(NC) * sigmoid(DAC) *
    (5 * sigmoid(TTC) + 5 * sigmoid(EP) + 2 * sigmoid(C)) / 12
```

直觉上这表示：

- 碰撞和可行驶区域是硬门
- 进度和 TTC 权重更高
- comfort 权重稍低

### 7.5 学习版代码摘录

```python
def _score_coarse(self, traj_feature, sub_rewards_group):
    # 1. 各个 head 分别预测一个 PDM 子指标
    NC_score  = self.NC_head(traj_feature).squeeze(-1)
    EP_score  = self.EP_head(traj_feature).squeeze(-1)
    DAC_score = self.DAC_head(traj_feature).squeeze(-1)
    TTC_score = self.TTC_head(traj_feature).squeeze(-1)
    C_score   = self.C_head(traj_feature).squeeze(-1)

    # 2. 用 PDM 子指标做监督
    loss_nc  = self.loss_bce(NC_score,  sub_rewards_group["no_collision"])
    loss_ep  = self.loss_bce(EP_score,  sub_rewards_group["progress"])
    loss_dac = self.loss_bce(DAC_score, sub_rewards_group["drivable_area"])
    loss_ttc = self.loss_bce(TTC_score, sub_rewards_group["ttc"])

    # 3. 额外加排序损失，让 progress 的相对顺序更靠谱
    loss_rank = self.rank_loss(...)

    # 4. 用手工公式合成 coarse 总分
    final_coarse_reward = (
        self.sigmoid(NC_score) * self.sigmoid(DAC_score) *
        (5 * self.sigmoid(TTC_score) + 5 * self.sigmoid(EP_score) + 2 * self.sigmoid(C_score)) / 12
    )
```

---

## 8. `_select_topk`：coarse-to-fine 的连接点

源码位置：

- `TrajectoryHead._select_topk`

这个函数的意义很直接：

- 先用 coarse reward 在大候选集中选 `topk`
- 再只对这批候选做更昂贵的 fine scorer

### 8.1 为什么需要这一层

因为生成器输出候选数非常多。

例如推理时：

- `num_groups = 10`
- `ego_fut_mode = 20`
- 基础候选就是 `200`
- `add_mul_noise()` 默认还会做 3 次增广

也就是说，进入 selector 前的候选规模会非常大。

如果 fine scorer 直接全量跑，成本会明显变高。

### 8.2 它同步裁剪的不只是轨迹

`_select_topk` 会一起 gather：

- `traj_feature`
- `noisy_traj_points_xy`
- `sub_rewards_group`

这样 coarse/fine 之间的监督和输入保持严格对齐。

---

## 9. `_score_fine_multi`：细排打分器

源码位置：

- `TrajectoryHead._score_fine_multi`

### 9.1 为什么叫 multi

因为 `fine_scorer_decoder` 有 3 层，函数会对每一层输出的特征都打一遍分。

所以：

- 它不是只产出一个 fine score
- 而是产出多层细排结果

### 9.2 它和 coarse 的关系

粗筛负责：

- 快速缩小搜索空间

细排负责：

- 在 top-k 里做更精细排序

### 9.3 训练时与测试时的差异

- 训练时：`only_reward=False`
  - 会计算监督 loss
- 测试时：`only_reward=True`
  - 只用预测 reward 做选取

---

## 10. `get_vocab_pdm_subscores`：词表候选增强

源码位置：

- `TrajectoryHead.get_vocab_pdm_subscores`

这部分是 selector 文件里最有“工程味”的一段。

### 10.1 它在做什么

它会从本地 GTRS 词表里读取：

- 预存轨迹 `gtrs_traj/16384.npy`
- 对应的 PDM 分数字典 `gtrs_traj/navtrain_16384.pkl`

再把这些词表候选的子分数拼到当前 batch 的真实生成候选后面。

### 10.2 为什么要这么做

直觉上是：

- 只靠当前生成器采样出来的候选，分布可能不够丰富
- 加入一个大词表，可以让 selector 见到更广的轨迹空间

### 10.3 一个关键细节

它用了：

- `dropout_ratio=0.99`

也就是虽然词表很大，但每个 batch 只随机保留极少部分词表轨迹，以免计算量爆炸。

---

## 11. `add_mul_noise`：候选扩增

源码位置：

- `TrajectoryHead.add_mul_noise`

它会对当前候选再做多次乘性扰动增广。

### 11.1 为什么 selector 还要增广

因为 selector 的目标不是“拟合一个固定候选集”，而是“学会在多种近邻候选里挑出更优轨迹”。

所以在 coarse/fine 之前，作者故意把候选再扩散一层。

---

## 12. `forward_train_rl`：Stage II 的训练主线

源码位置：

- `TrajectoryHead.forward_train_rl`

尽管名字还叫 `forward_train_rl`，但 selector 这一步其实主要在做：

- 冻结生成器
- 训练选择器

### 12.1 训练时执行顺序

```text
with torch.no_grad():
    1. 跑 2 步截断扩散生成候选
    2. 对候选做乘性噪声增广
    3. 用 PDM 给候选打 reward / sub-reward
    4. 拼上 GTRS 词表候选

然后：
    5. 用 coarse scorer 给全体候选打分
    6. 取 top-k
    7. 用 fine scorer 细排
    8. 计算 coarse loss + fine loss
```

### 12.2 一个重要观察

生成器部分被包在：

```python
with torch.no_grad():
```

也就是说 selector 训练时，候选生成链是冻结的。

### 12.3 训练时候选规模

在当前代码里，训练时：

- `num_groups = 2`
- 基础候选数 `2 * 20 = 40`
- `add_mul_noise(..., n_aug=2)` 后变成 `40 * 3 = 120`
- 再拼接经过高 dropout 的词表候选

最终进入 coarse scorer 的候选数会明显大于 120。

### 12.4 学习版代码摘录

```python
with torch.no_grad():
    # 1. 先用冻结生成器生成候选轨迹
    diffusion_output = ...
    diffusion_output = self.add_mul_noise(diffusion_output, n_aug=2, std_min=0.1, std_max=0.2)
    diffusion_output = self.bezier_xyyaw(self.denorm_odo(diffusion_output))

    # 2. 用 PDM 给这些候选打子分数
    reward_group, metric_cache, sub_rewards_group, sim_traj = self.get_pdm_score_para(...)

    # 3. 再混入大词表候选，扩充搜索空间
    sub_rewards_group, keep_idx = self.get_vocab_pdm_subscores(...)
    diffusion_output = torch.cat((diffusion_output, vocab), dim=1)

# 4. coarse scorer 先粗筛
noisy_traj_points_xy, traj_feature, time_embed = self._get_scorer_inputs(...)
traj_feature = self.scorer_decoder(...)[-1]
loss_coarse, final_coarse_reward, coarse_reward, sub_loss_dict = self._score_coarse(...)

# 5. 取 top-k，再交给 fine scorer
traj_feature, noisy_traj_points_xy, sub_rewards_topk, topk_idx, topk_val = self._select_topk(...)
fine_traj_feature_list = self.fine_scorer_decoder(...)
loss_fine, ... = self._score_fine_multi(fine_traj_feature_list, sub_rewards_topk)

loss = loss_coarse + loss_fine
```

---

## 13. `forward_test_rl`：推理阶段的 coarse-to-fine 选择

源码位置：

- `TrajectoryHead.forward_test_rl`

### 13.1 测试时候选更多

测试时：

- `num_groups = 10`
- 基础候选 `10 * 20 = 200`
- `add_mul_noise()` 默认 `n_aug=3`
- 所以增广后候选达到 `200 * 4 = 800`

这说明推理时 selector 面对的是比训练更大的候选池。

### 13.2 测试时不是直接输出 loss，而是输出 reward 统计

它会：

1. coarse scorer 选出一个最好轨迹
2. fine scorer 的每层也各选一个最好轨迹
3. 把这些轨迹一起送去 PDM 做正式评估
4. 返回 `reward_dict`

源码里还有一行被注释掉的：

```python
# return {"trajectory": traj_to_score[:,-1]}
```

这说明作者在实验时可能也用这个分支直接输出最终轨迹做官方评测。

---

## 14. 这份文件最值得你牢记的 7 个点

1. Selector 训练不是替代生成器，而是建立在冻结生成器之上。
2. 打分器输入不是 BEV 全局特征，而是“轨迹编码后的 query”。
3. Coarse scorer 预测的是 PDM 子指标，而不是只预测一个总分。
4. Fine scorer 只在 top-k 候选上工作，降低计算成本。
5. 词表候选增强是这个实现里非常工程化、也非常实用的一步。
6. 测试时候选规模明显大于训练时，说明作者希望 selector 在大搜索空间里发挥作用。
7. 这一阶段的本质不是“再生成”，而是“在大候选集中学会选择”。

---

## 15. 这份文件与论文的关系

论文里的 coarse-to-fine selector，在代码中最直接落在：

- `_get_scorer_inputs`
- `_score_coarse`
- `_select_topk`
- `_score_fine_multi`
- `forward_train_rl`
- `forward_test_rl`

而论文里“为什么 selector 有用”的直觉，在代码里体现为：

- 先用生成器保证多样性
- 再用多子项 scorer 保证质量

这也是这份文件最值得迁移到自己项目里的设计思想。
