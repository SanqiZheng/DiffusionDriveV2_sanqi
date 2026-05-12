# 06. 逐函数精读：`V2TransfuserModel.forward`

主分析对象：

- [diffusiondrivev2_model_rl.py](/home/yihang/Downloads/CodeReference/diff_based/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py)

这篇专门只讲一件事：

`V2TransfuserModel.forward` 这一趟前向，到底是怎么把传感器输入变成“可以送进扩散规划头的条件”的。

## 1. 先说这个函数的职责

`TrajectoryHead` 负责真正的轨迹扩散与 RL，但它自己并不直接处理原始图像和点云。  
`V2TransfuserModel.forward` 的职责就是先把场景压成几类更适合规划的条件：

1. `keyval`
   用于 transformer decoder 做全局记忆读取。
2. `trajectory_query`
   表示 ego 规划查询。
3. `agents_query`
   表示周围目标查询。
4. `cross_bev_feature`
   表示可被轨迹点直接 `grid_sample` 的 BEV 稠密特征图。

所以这个函数本质上是“感知输出 -> 规划条件接口”的桥。

---

## 2. 输入到底是什么

来自 [transfuser_features.py](/home/yihang/Downloads/CodeReference/diff_based/DiffusionDriveV2/navsim/agents/diffusiondrivev2/transfuser_features.py)：

- `camera_feature`: `[B, 3, 256, 1024]`
- `lidar_feature`: `[B, 1, 256, 256]`
- `status_feature`: `[B, 8]`

其中 `status_feature` 的 8 维语义是：

- 4 维驾驶指令
- 2 维速度
- 2 维加速度

直觉上：

- 图像提供语义和远处结构
- LiDAR 提供几何和占据关系
- status 提供 ego 当前控制意图和动态状态

---

## 3. 第一步：backbone 做多模态融合

代码：

```python
bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
```

这里调用的是：

- [transfuser_backbone.py](/home/yihang/Downloads/CodeReference/diff_based/DiffusionDriveV2/navsim/agents/diffusiondrivev2/transfuser_backbone.py)

这个 backbone 的关键输出有两个：

### 3.1 `bev_feature_upscale`

你可以把它理解成：

- 分辨率较高
- 更适合做 BEV 语义图和局部空间采样

它后面会被：

- `self._bev_semantic_head` 用来出 `bev_semantic_map`
- `cross_bev_feature` 用来给轨迹点做局部取值

### 3.2 `bev_feature`

你可以把它理解成：

- 更紧凑的全局 fused feature
- 更适合 flatten 成 token

也就是说，同一个 backbone 同时输出了：

- 一个适合 token 推理的抽象表示
- 一个适合空间采样的稠密表示

这就是为什么后面会同时出现 `keyval` 和 `cross_bev_feature` 两条分支。

---

## 4. 第二步：把 `bev_feature` 变成 token

代码：

```python
bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
bev_feature = bev_feature.permute(0, 2, 1)
status_encoding = self._status_encoding(status_feature)
keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
keyval += self._keyval_embedding.weight[None, ...]
```

我们拆开看。

### 4.1 `self._bev_downscale`

定义：

```python
self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
```

作用：

- 把 backbone 输出的 512 通道压成 `tf_d_model=256`

为什么要这样做：

- transformer token 维度要统一
- 256 比 512 更省算力

### 4.2 `flatten(-2, -1)` + `permute`

假设 `bev_feature` 空间大小是 `8x8`，那么：

- 下采样后是 `[B, 256, 8, 8]`
- `flatten(-2, -1)` 后是 `[B, 256, 64]`
- `permute(0, 2, 1)` 后变成 `[B, 64, 256]`

也就是说：

- 64 个位置 token
- 每个 token 维度 256

### 4.3 `status_encoding`

定义：

```python
self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)
```

输入是 `[B, 8]`，输出是 `[B, 256]`。

后面 `status_encoding[:, None]` 变成 `[B, 1, 256]`，相当于给 transformer 增加一个“ego 状态 token”。

### 4.4 `keyval`

拼接后：

- `[B, 64, 256] + [B, 1, 256] -> [B, 65, 256]`

这 65 个 token 的语义是：

- 前 64 个：来自 BEV 空间网格
- 最后 1 个：来自 ego status

### 4.5 为什么加 `self._keyval_embedding`

定义：

```python
self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)
```

作用：

- 给每个 token 一个可学习的位置/身份 embedding

注意这里很工程化：

- 直接写死成 `8**2 + 1`

这意味着作者默认最后 token 网格就是 `8x8`。  
如果你后面换 backbone 或改输入尺寸，这里很可能要一起改。

---

## 5. 第三步：为什么还要重建一个 `cross_bev_feature`

这是这个函数最容易让人迷惑的地方。

代码：

```python
concat_cross_bev = keyval[:,:-1].permute(0,2,1).contiguous().view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)
cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)
cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2,-1).permute(0,2,1))
cross_bev_feature = cross_bev_feature.permute(0,2,1).contiguous().view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])
```

这个过程可以翻译成一句话：

“把 token 分支的全局抽象信息重新还原到 BEV 空间，再和高分辨率 BEV 特征拼起来，形成适合局部轨迹点采样的稠密地图特征。”

### 5.1 `keyval[:,:-1]`

去掉最后那个 status token，只保留前 64 个空间 token。

为什么：

- status token 没有空间坐标
- 没法还原成 feature map

### 5.2 `view(..., H, W)`

根据 `concat_cross_bev_shape` 把 token 重新变回 2D 网格。

如果原来是 `8x8`：

- `[B, 64, 256] -> [B, 256, 8, 8]`

### 5.3 `F.interpolate`

把这个较粗的空间表示，上采样到 `bev_feature_upscale` 的大小。

为什么：

- 后面轨迹点要去图上采样局部特征
- 分辨率越高，局部几何越细

### 5.4 `torch.cat([concat_cross_bev, cross_bev_feature], dim=1)`

这里拼接了两种信息：

- `concat_cross_bev`
  带有 transformer 全局融合语义
- `cross_bev_feature`
  来自 backbone 的高分辨率稠密 BEV

如果两者通道分别是 `256` 和 `64`，拼完就是 `320` 通道。

### 5.5 `bev_proj`

定义：

```python
self.bev_proj = nn.Sequential(
    *linear_relu_ln(256, 1, 1, 320),
)
```

它不是卷积，而是先把 BEV 图 flatten 成 token 再做 MLP 投影：

- 先 `[B, 320, H, W]`
- 再 `[B, H*W, 320]`
- 经过线性层投影到 `[B, H*W, 256]`
- 再 reshape 回 `[B, 256, H, W]`

为什么要这么做：

- 把拼接后的 320 维统一压回 256
- 保持后续 `GridSampleCrossBEVAttention` 的输入维度一致

---

## 6. 第四步：构造 query，并用 transformer decoder 读 `keyval`

代码：

```python
query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
query_out = self._tf_decoder(query, keyval)
```

### 6.1 `self._query_splits`

在初始化里：

```python
self._query_splits = [
    1,
    config.num_bounding_boxes,
]
```

如果 `num_bounding_boxes = 30`，那 query 总数就是：

- `1 + 30 = 31`

也就是说，这里有两类 query：

- 1 个 ego trajectory query
- 30 个 agent query

### 6.2 `query`

它是纯可学习参数，不依赖输入。

语义上可以理解成：

- 一组固定的“我要问场景什么”的槽位

### 6.3 `self._tf_decoder(query, keyval)`

作用：

- 让 query 去读场景 memory

输出 `query_out` 后，再 split 成：

```python
trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)
```

所以 shape 通常是：

- `trajectory_query`: `[B, 1, 256]`
- `agents_query`: `[B, 30, 256]`

---

## 7. 第五步：BEV 语义头为什么还在这里

代码：

```python
bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}
```

这说明主模型仍然保留了多任务结构：

- 一边做规划
- 一边做 BEV 语义图

虽然 RL 版训练主要集中在 `_trajectory_head`，但这个输出接口还在。

工程意义：

- 兼容原本的多任务训练结构
- 也方便后续用语义分支做辅助监督或调试

---

## 8. 第六步：为什么 `_trajectory_head` 要调用两次

代码：

```python
with torch.no_grad():
    old_pred = self._trajectory_head(...)
pred = self._trajectory_head(..., old_pred=old_pred, ...)
```

这是整个 RL 设计的关键。

### 第一次调用：采样旧轨迹链

第一次在 `torch.no_grad()` 里做的事，本质上是：

1. 从 anchor + 噪声出发跑一遍扩散采样
2. 得到一整条 `all_diffusion_output`
3. 调 PDM scorer 算 reward
4. 构造 advantages

这一步像 RL 里的 rollout / collect trajectories。

### 第二次调用：用旧轨迹链算当前损失

第二次调用时传入 `old_pred`，于是 `_trajectory_head` 不再重新采样，而是进入 `get_rlloss(...)`：

1. 读取旧轨迹链
2. 重新评估当前策略对这些 transition 的 log-prob
3. 用 `advantages` 组装策略梯度损失

这一步像 RL 里的 policy update。

### 为什么不合并成一次

因为这两步的角色不同：

- 第一次需要“采样”
- 第二次需要“算梯度”

如果混在一起，计算图会更乱，也更难控制旧策略/当前策略的边界。

---

## 9. 第七步：`AgentHead` 的位置意义

代码最后：

```python
agents = self._agent_head(agents_query)
output.update(agents)
```

说明：

- `agents_query` 不只是给轨迹头当条件
- 也真的接了一个周围车预测头

这和端到端自动驾驶里常见的“共享场景 token，再接多任务 heads”是同一思路。

---

## 10. 用一张顺序图记住这个 forward

你可以把 `V2TransfuserModel.forward` 记成下面这 10 步：

1. 读入 `camera_feature / lidar_feature / status_feature`
2. backbone 做图像-激光融合
3. `bev_feature` 压成 256 通道 token
4. `status_feature` 编成 status token
5. 拼成 `keyval`
6. 把空间 token 重新还原为 BEV 稠密图，并与高分辨率 BEV 拼接，得到 `cross_bev_feature`
7. 用 learnable query 去读 `keyval`
8. 拆出 `trajectory_query` 和 `agents_query`
9. 第一次调用 `_trajectory_head` 收集轨迹链和 reward
10. 第二次调用 `_trajectory_head` 计算 RL loss，并再接 `AgentHead`

如果你能把这 10 步在脑子里复述出来，这个函数就真的读明白了。

---

## 11. 这个函数最核心的设计思想

不是“把 backbone 输出直接丢给轨迹头”，而是刻意构造了两套互补条件：

- `keyval`
  负责全局语义与 token 级推理
- `cross_bev_feature`
  负责轨迹点到地图的局部几何读取

这正体现了开发者的正向思维：

- 全局 token 很强，但不适合直接做每个轨迹点的局部空间采样
- 稠密特征很细，但缺乏 transformer 融合后的全局语义
- 两者都要，规划才稳

这也是这份代码里最值得迁移到你自己项目里的一个思路。
