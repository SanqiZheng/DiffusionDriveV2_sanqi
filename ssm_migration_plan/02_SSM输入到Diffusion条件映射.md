# SSM 输入到 Diffusion 条件映射

## 主定义

SSM 输入到 Diffusion 条件映射，是把当前工程的结构化消息集合转换成 DiffusionDriveV2 `TrajectoryHead` 需要的 `ego_query / agents_query / bev_feature / status_encoding` 等条件张量。

## 最终记忆版本

`动态障碍物变 agents_query，freespace/参考线/静态语义变 bev_feature，自车和目标变 ego_query/status_encoding。`

## DiffusionDriveV2 轨迹头需要什么

当前代码中的轨迹头核心调用可抽象为：

```text
diff_decoder(
  traj_feature,
  noisy_traj_points,
  bev_feature,
  bev_spatial_shape,
  agents_query,
  ego_query,
  time_embed,
  status_encoding
)
```

其中：

| 条件 | 含义 | 是否必须 |
|---|---|---|
| `ego_query` | 自车与规划意图 token | 必须 |
| `agents_query` | 动态障碍物 token | 必须 |
| `bev_feature` | 稠密 BEV 场景特征 | 必须 |
| `bev_spatial_shape` | BEV 特征图尺寸 | 必须 |
| `status_encoding` | 自车状态编码 | 必须 |
| `status_feature` | 8 维工程状态向量 | 必须 |
| `time_embed` | diffusion timestep 编码 | 模型内部生成 |
| `noisy_traj_points` | 当前扩散步候选轨迹 | 模型内部生成 |

## 输入映射表

| SSM 输入 | Diffusion 条件 | 处理方式 |
|---|---|---|
| `VSTATUS_DATA` | `ego_query`, `status_encoding`, `status_feature` | 速度、加速度、yaw rate、档位、方向盘角、驾驶模式编码 |
| `ODOMETRY` | `ego_query`, 坐标转换参数 | 提供 ego pose、yaw、坐标系对齐 |
| `NDM_SP_LOCATION` | `ego_query`, `route_condition` | 提供 submap、定位点、全局到局部转换基准 |
| `NDM_SP_NAVIGATION` | `route_condition`, `bev_feature` | 参考线、目标车位、停车目标 rasterize 或 token 化 |
| `LIDAR_TRAJ` / `ADAS_OBS` | `agents_query` | 目标 id、类别、位置、速度、尺寸、置信度编码 |
| `ADAS_STATIC_OBJECT` | `bev_feature` | 车位、锥桶、停止线、静态边界 rasterize |
| `ADAS_LANE` / `BEV_LINE` | `bev_feature` | 车道线、边界线、路沿等 rasterize |
| `APA_FREESPACE_GRIDX` | `bev_feature` | 可行驶区域主输入 |
| `BEVSR_FREESPACE` | `bev_feature` | freespace 备份或增强 |
| `PLANNING_TRAJ` | optional history | 第一版可不用 |
| `STATE_MANAGER` / `PNC_SM_STATE` | `ego_query` 或 gating | 场景状态、是否允许 E2E 运行 |

## `agents_query` 设计

建议每个动态目标构造一个基础向量：

```text
agent_raw =
  id_embed
  type_onehot
  confidence
  x, y, yaw
  vx, vy
  ax, ay
  length, width, height
  valid_mask
```

再经过 MLP：

```text
agent_raw -> AgentConditionEncoder -> agents_query [B,30,256]
```

第一版建议：

| 参数 | 建议 |
|---|---|
| `A` | 固定为 30，匹配 DiffusionDriveV2 `num_bounding_boxes` 默认配置 |
| 排序 | 按距离 ego 或风险优先 |
| 坐标系 | 车辆局部坐标 |
| padding | `valid_mask=false` |
| 历史 | 暂不接，后续扩展 |

## `bev_feature` 设计

DiffusionDriveV2 的 BEV 条件必须是稠密图，因为轨迹点会在 BEV feature 上做局部采样。

建议先构造多通道 raster：

```text
bev_raster [B,C0,H,W]
  channel 0: freespace valid
  channel 1: occupied/static obstacle
  channel 2: dynamic obstacle footprint
  channel 3: reference line
  channel 4: left boundary
  channel 5: right boundary
  channel 6: target parking slot
  channel 7: stop line / stop fence
```

再经过轻量 CNN：

```text
bev_raster -> BEVConditionEncoder -> bev_feature [B,256,H,W]
```

第一版建议分辨率：

```text
H = 128 或 160
W = 128 或 160
range = 车辆周围固定局部窗口
```

实际分辨率要以 SSM freespace 的 `width/height/resolution/origin` 能稳定提供为准。

## `ego_query` 和 `status_encoding`

建议输入：

```text
ego_raw =
  speed
  acceleration
  yaw_rate
  steering_angle
  gear
  brake
  throttle
  drive_mode
  command
  target_slot_relative_pose
  route_progress
```

编码：

```text
ego_raw -> EgoConditionEncoder -> ego_query [B,1,256]
ego_raw -> StatusEncoder -> status_encoding [B,1,256]
```

## `route_condition`

泊车参考线和目标车位非常关键，不应只作为普通 map channel。

第一版可采用两种方式之一：

1. rasterize 到 `bev_feature`。
2. 额外输出 route token，再和 `ego_query` 融合。

建议第一版优先 rasterize，原因是更容易导出 ONNX/HBM。

## 不建议第一版使用的输入

| 输入 | 原因 |
|---|---|
| `lane_index / bind_flag / dists` | 这些是 SSM 后处理语义，不是稳定原始输入 |
| `temp_instance_feature` | 属于 SparseDrive instance queue，不是 DiffusionDriveV2 主链天然输入 |
| 上一帧规划轨迹 | 可提升稳定性，但第一版会增加耦合 |
| BEV3D | 依赖编译开关和时间同步，先作为增强项 |

## 明确不作为输入的内容

第一阶段不要求输入：

```text
三前向相机图像
点云
NAVSIM 原始 lidar_feature
DiffusionDriveV2 TransfuserBackbone 输出
```

原因是当前移植目标是 `TrajectoryHead`，不是完整 DiffusionDriveV2 感知-规划全网络。
