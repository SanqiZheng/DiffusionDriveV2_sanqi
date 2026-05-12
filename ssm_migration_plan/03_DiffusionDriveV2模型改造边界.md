# DiffusionDriveV2 模型改造边界

## 主定义

DiffusionDriveV2 模型改造的核心，是只移植 anchor-conditioned diffusion `TrajectoryHead`，抛弃原始 Transfuser sensor backbone，并用新增 SSMConditionEncoder 把 J6 SSM 输入编码为该规划头需要的条件张量。

## 最终记忆版本

`不接 DiffusionDriveV2 全网络；只接 TrajectoryHead，SSMConditionEncoder 负责生成 ego_query、agents_query、bev_feature 和 status 条件，wrapper 暴露原始轨迹输出，端侧不跑 PDM scorer。`

## 当前共识约束

```text
不改变原始 DiffusionDriveV2 网络语义。
不新增 LocalRoad/Decision 语义预测头。
不要求 DiffusionDriveV2 直接输出 centerline / boundary / stop fence / nudge boundary。
只把 DiffusionDriveV2 的原始轨迹类输出作为 adapter 的后处理输入。
不要求 J6 SSM module 输入三前向相机图像、点云或 NAVSIM BEV raster。
不移植 DiffusionDriveV2 原始感知 backbone。
```

## 当前原始模型链路

当前 `V2TransfuserModel.forward` 的主链路是：

```text
camera_feature
lidar_feature
status_feature
  -> TransfuserBackbone
  -> bev_feature_upscale / bev_feature
  -> keyval + transformer decoder
  -> trajectory_query / agents_query
  -> TrajectoryHead
  -> diffusion candidates + selector
```

这条链路适合 NAVSIM 研究环境，但不适合直接接入 SSM 第一阶段。

## 规划头真实输入契约

源码中的 `TrajectoryHead.forward()` 入口是：

```text
ego_query
agents_query
bev_feature
bev_spatial_shape
status_encoding
status_feature
camera_feature
```

其中第一阶段移植真正需要重建的是：

| 条件张量 | 推荐形状 | 来源 |
|---|---:|---|
| `ego_query` | `[B,1,256]` | 自车状态、泊车目标、导航意图编码 |
| `agents_query` | `[B,30,256]` | 动态障碍物列表 padding 后编码 |
| `bev_feature` | `[B,256,H,W]` | freespace、参考线、静态语义、车道/边界线 raster 后编码 |
| `bev_spatial_shape` | `[2]` | 固定部署 BEV 特征高宽 |
| `status_encoding` | `[B,1,256]` | 8 维状态向量线性/MLP 编码 |
| `status_feature` | `[B,8]` | 车速、加速度、命令/状态等工程状态 |

`camera_feature` 在当前 `TrajectoryHead` 实现中只是透传参数，第一阶段 deployment wrapper 可以传空占位或删除该参数依赖；不能因此把原始相机输入重新引入 SSM module。

## 新 deployment 模型链路

建议新增：

```text
class DiffusionDriveV2DeploymentCore(nn.Module):
    def forward(
        self,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        status_feature,
        valid_masks=None,
    ):
        ...
```

这个 wrapper 的权限很窄：

```text
可以绕开离线 reward_dict 返回路径。
可以暴露 forward_test_rl 内部已经算出的 traj_to_score / traj_to_score[:,-1]。
不可以新增 LocalRoad/Decision 预测分支。
不可以改变 diffusion head 的数学语义。
deployment wrapper 本身不改变 checkpoint 权重。
```

内部保留：

| 模块 | 是否保留 | 原因 |
|---|---|---|
| `plan_anchor` | 保留 | 提供轨迹意图原型 |
| `time_mlp` | 保留 | diffusion timestep 条件 |
| `plan_anchor_encoder` | 保留 | noisy trajectory 转 token |
| `diff_decoder` | 保留 | 核心去噪轨迹生成 |
| `scorer_decoder` | 保留 | selector 选轨迹 |
| `fine_scorer_decoder` | 可保留 | 质量更高但更重 |
| `bezier_xyyaw` | 保留 | 从 xy 恢复 yaw |

需要替换：

| 模块 | 处理 |
|---|---|
| `TransfuserBackbone` | 抛弃，不进入 SSM module |
| `_tf_decoder` | 可替换为条件 adapter 输出，不必保留 |
| `_agent_head` | 第一阶段不需要 |
| `_bev_semantic_head` | 第一阶段不需要 |
| 在线 `PDMScorer` | 端侧去掉 |
| `metric_cache` | 端侧去掉 |

## 推理输出接口要改

当前 selector 版本默认返回 `reward_dict`，不是工程 adapter 需要的轨迹。

deployment wrapper 应返回：

```text
selected_trajectory: float32 [B,8,3]
candidate_trajectories: float32 [B,G,8,3]
candidate_scores: float32 [B,G]
valid_flag: bool [B]
```

其中 `selected_trajectory` 来自原代码中已经存在的：

```text
traj_to_score[:,-1]
```

后续是否使用这些轨迹由 shadow / switch / fusion 策略决定；不能新增到 DiffusionDriveV2 网络头里，也不能让规划头直接预测 `LocalRoad + Decision` 字段。

## 训练策略

不能假设原始 checkpoint 直接可用。

原因：

```text
原始 checkpoint 的 planning head 学到的是 Transfuser backbone 产生的条件分布。
新方案条件来自 SSM structured/raster encoder。
因此 SSMConditionEncoder 必须训练，不能手写规则直接拼 token。
```

建议训练顺序：

1. 冻结 diffusion head，先训练 `SSMConditionEncoder` 对齐原模型中间特征或轨迹监督。
2. 解冻 diffusion head 做监督微调。
3. 用回放场景做闭环指标筛选。
4. 再考虑 RL/selector 微调。

如果没有原始中间特征监督，也可以直接训练：

```text
SSM condition -> selected_trajectory
```

但这要求有足够的泊车数据标注或可用规则 planner 伪标签。

## 导出约束

为了 ONNX/HBM，deployment forward 必须满足：

| 约束 | 要求 |
|---|---|
| batch | 第一版固定 `B=1` |
| agent 数 | 固定 `A`，padding + mask |
| BEV shape | 固定 `H,W` |
| diffusion steps | 固定，例如 `2` |
| random | 端侧关闭或固定 |
| PDM scorer | 不进入导出图 |
| Python list/dict 动态逻辑 | 导出前改为固定 tensor 输出 |

## 第一版不做的事

第一版不建议做：

1. 完整端到端从 camera/lidar 原始输入重新训练。
2. 直接把旧 5 个 HBM 拼成新 HBM。
3. 把 `temp_instance_feature` 强行接入 diffusion head。
4. 端侧运行 PDM scorer。
5. 让模型直接写 C++ `LocalRoad` 或 `Decision`。
6. 新增 `centerline/boundary/stop fence/nudge boundary` 预测头。
