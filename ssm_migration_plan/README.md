# DiffusionDriveV2 接入 SSM 替换计划

## 主定义

DiffusionDriveV2 接入 SSM 的合理路线，是在 J6 SSM module 输入链路不变的前提下，只移植 DiffusionDriveV2 的 `TrajectoryHead`，用新增 `SSMConditionEncoder` 把现有 SSM 输入编码成规划头条件张量，再暴露规划头的原始多模态轨迹输出。

## 最终记忆版本

`SSM 原始输入不变 -> SSMConditionEncoder -> DiffusionDriveV2 TrajectoryHead -> 多模态轨迹输出`

## 文档索引

0. [00_需求共识.md](./00_需求共识.md)
1. [01_行动计划.md](./01_行动计划.md)
2. [02_SSM输入到Diffusion条件映射.md](./02_SSM输入到Diffusion条件映射.md)
3. [03_DiffusionDriveV2模型改造边界.md](./03_DiffusionDriveV2模型改造边界.md)
4. [04_输出适配_验证_部署路线.md](./04_输出适配_验证_部署路线.md)

## 核心判断

不要把迁移工作限定在旧 `det_adapter / map_adapter / ego_adapter` 框架内。旧 adapter 的输出是 SparseDrive motion head 的输入契约，不是 DiffusionDriveV2 的天然输入契约。

DiffusionDriveV2 轨迹头真正需要的是：

```text
ego_query
agents_query
bev_feature
bev_spatial_shape
status_encoding
status_feature
```

新增硬约束：

```text
不改变原始 DiffusionDriveV2 网络语义。
不新增 LocalRoad/Decision 语义预测头。
不要求 DiffusionDriveV2 直接输出 centerline / boundary / stop fence / nudge boundary。
只把 DiffusionDriveV2 的原始轨迹类输出作为 adapter 的后处理输入。
不接入 DiffusionDriveV2 原始 camera/lidar 感知 backbone。
不要求 SSM 模块输入三前向相机图像、点云或 NAVSIM BEV raster。
允许写 deployment wrapper，但它只暴露内部已存在的多模态轨迹和 selected trajectory，不改 TrajectoryHead 语义、不新增预测头。
```

因此，新的工程边界应是：

```text
现有 SSM 原始输入
  -> SSMConditionEncoder
  -> DiffusionDriveV2 TrajectoryHead deployment forward
  -> candidate_trajectories / candidate_scores / selected_trajectory
```

## 主要参考文档

- `/home/yihang/Documents/j5code/端到端SSM替换文档/00_项目目标.md`
- `/home/yihang/Documents/j5code/端到端SSM替换文档/02_现有输入契约.md`
- `/home/yihang/Documents/j5code/端到端SSM替换文档/04_端到端输出到SSM字段映射.md`
- `/home/yihang/Documents/j5code/端到端SSM替换文档/05_适配层设计.md`
- `navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel.py`
- `navsim/agents/diffusiondrivev2/modules/blocks.py`
