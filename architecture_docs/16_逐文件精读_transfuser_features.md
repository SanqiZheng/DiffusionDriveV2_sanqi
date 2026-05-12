# 16 逐文件精读：`transfuser_features.py`

本文对应源码：

- `navsim/agents/diffusiondrivev2/transfuser_features.py`

这份文件不负责模型计算，但它决定了一个更基础的问题：

- 模型真正吃进去的输入到底是什么
- 监督目标到底是什么

如果这份文件没读透，你后面看 `forward(features, targets)` 时就很容易“看得懂代码，但不知道张量在表达什么”。

---

## 1. 这份文件回答哪 4 个问题

1. 相机输入是如何组织的
2. LiDAR 输入是如何变成网络张量的
3. ego 状态是如何编码成 `status_feature` 的
4. target 里的 `trajectory` / `agent_states` / `bev_semantic_map` 分别来自哪里

---

## 2. `TransfuserFeatureBuilder`：输入侧构造器

源码位置：

- `class TransfuserFeatureBuilder`
- `compute_features`
- `_get_camera_feature`
- `_get_lidar_feature`

### 2.1 `compute_features` 输出了什么

这个函数最终返回一个 `dict`，包含 3 个键：

- `camera_feature`
- `lidar_feature`
- `status_feature`

这 3 个键会直接成为后面模型 forward 的输入字典键名。

### 2.2 学习版代码摘录

```python
def compute_features(self, agent_input):
    features = {}

    # 1. 视觉输入：拼接后的前向三相机图像
    features["camera_feature"] = self._get_camera_feature(agent_input)

    # 2. LiDAR 输入：BEV histogram
    features["lidar_feature"] = self._get_lidar_feature(agent_input)

    # 3. ego 状态输入：驾驶命令 + 速度 + 加速度
    features["status_feature"] = torch.concatenate([
        torch.tensor(agent_input.ego_statuses[-1].driving_command, dtype=torch.float32),  # 4 维
        torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32),      # 2 维
        torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32),  # 2 维
    ])

    return features
```

### 2.3 `status_feature` 的语义

`status_feature` 不是位姿，而是：

- `driving_command`
- `ego_velocity`
- `ego_acceleration`

总计 8 维：

```text
4 + 2 + 2 = 8
```

后面在主模型里会被投影成一个额外的 token。

---

## 3. `_get_camera_feature`：相机输入怎么组织

源码位置：

- `TransfuserFeatureBuilder._get_camera_feature`

### 3.1 它没有用全 8 路相机

这里只取了最后一个历史时刻的 3 路相机：

- `cam_l0`
- `cam_f0`
- `cam_r0`

也就是：

- 左前
- 正前
- 右前

### 3.2 处理步骤

1. 对左右相机做横向裁切
2. 对三张图做上下裁切，保证可拼接
3. 横向拼接成一张超宽图
4. resize 到 `(1024, 256)`
5. 转成 tensor

### 3.3 最终 shape

```text
camera_feature: [3, 256, 1024]
```

### 3.4 学习版代码摘录

```python
def _get_camera_feature(self, agent_input):
    cameras = agent_input.cameras[-1]  # 只取当前时刻

    # 1. 裁成可拼接的视角区域
    l0 = cameras.cam_l0.image[28:-28, 416:-416]
    f0 = cameras.cam_f0.image[28:-28]
    r0 = cameras.cam_r0.image[28:-28, 416:-416]

    # 2. 左-中-右横向拼接，形成一个超宽前视图
    stitched_image = np.concatenate([l0, f0, r0], axis=1)

    # 3. 统一 resize，便于 backbone 处理
    resized_image = cv2.resize(stitched_image, (1024, 256))
    tensor_image = transforms.ToTensor()(resized_image)

    return tensor_image
```

### 3.5 为什么这样设计

它是一种很典型的工程折中：

- 不用全相机 surround-view 网络
- 也不只看正前一张图
- 而是用三路前向视角扩展可见范围

好处是：

- 实现简单
- 输入固定
- 能复用 TransFuser 风格 backbone

---

## 4. `_get_lidar_feature`：LiDAR 怎么变成网络输入

源码位置：

- `TransfuserFeatureBuilder._get_lidar_feature`

### 4.1 它不是点云 Transformer，也不是 voxel encoder

这个项目里的 LiDAR 处理非常“经典工程化”：

- 把点云投成 2D histogram
- 形成 BEV raster

### 4.2 处理步骤

1. 只取当前时刻最后一帧 LiDAR
2. 从点云里取 `(x, y, z)`
3. 过滤掉过高的点
4. 按 `lidar_split_height` 分成上下两层
5. 对 `(x, y)` 做 `histogramdd`
6. 每个格子计数截断到 `hist_max_per_pixel`
7. 归一化到 `[0, 1]`

### 4.3 默认输出 shape

默认 `use_ground_plane=False`，所以只有一层：

```text
lidar_feature: [1, 256, 256]
```

如果开启地面层，会变成：

```text
lidar_feature: [2, 256, 256]
```

### 4.4 学习版代码摘录

```python
def _get_lidar_feature(self, agent_input):
    # 1. 当前时刻点云，只取 x/y/z
    lidar_pc = agent_input.lidars[-1].lidar_pc[LidarIndex.POSITION].T  # [N, 3]

    # 2. 先过滤过高点
    lidar_pc = lidar_pc[lidar_pc[..., 2] < self._config.max_height_lidar]

    # 3. 按高度拆成两层
    below = lidar_pc[lidar_pc[..., 2] <= self._config.lidar_split_height]
    above = lidar_pc[lidar_pc[..., 2] > self._config.lidar_split_height]

    # 4. 把 x/y 统计成 256 x 256 的栅格计数图
    above_features = splat_points(above)

    if self._config.use_ground_plane:
        below_features = splat_points(below)
        features = np.stack([below_features, above_features], axis=-1)
    else:
        features = np.stack([above_features], axis=-1)

    # 5. 最终转成 [C, H, W]
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    return torch.tensor(features)
```

### 4.5 为什么这样设计

这说明作者更关心：

- 稳定
- 快速
- 和 TransFuser 原始范式兼容

而不是把 LiDAR 编码做得特别重。

---

## 5. `TransfuserTargetBuilder`：监督目标构造器

源码位置：

- `class TransfuserTargetBuilder`
- `compute_targets`

### 5.1 它输出什么

target 字典包含：

- `trajectory`
- `agent_states`
- `agent_labels`
- `bev_semantic_map`

### 5.2 `trajectory` 来自哪里

```python
scene.get_future_trajectory(
    num_trajectory_frames=self._config.trajectory_sampling.num_poses
).poses
```

默认配置里：

- `time_horizon = 4`
- `interval_length = 0.5`

所以未来轨迹点数是：

```text
num_poses = 8
```

最终：

```text
trajectory: [8, 3]
```

语义是：

- 未来 8 个时刻的 `(x, y, heading)`
- 相对于当前 ego 的局部坐标

### 5.3 学习版代码摘录

```python
def compute_targets(self, scene):
    trajectory = torch.tensor(
        scene.get_future_trajectory(
            num_trajectory_frames=self._config.trajectory_sampling.num_poses
        ).poses
    )  # [8, 3]

    frame_idx = scene.scene_metadata.num_history_frames - 1
    annotations = scene.frames[frame_idx].annotations
    ego_pose = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)

    agent_states, agent_labels = self._compute_agent_targets(annotations)
    bev_semantic_map = self._compute_bev_semantic_map(annotations, scene.map_api, ego_pose)

    return {
        "trajectory": trajectory,
        "agent_states": agent_states,
        "agent_labels": agent_labels,
        "bev_semantic_map": bev_semantic_map,
    }
```

---

## 6. `_compute_agent_targets`：目标框监督是怎么来的

源码位置：

- `TransfuserTargetBuilder._compute_agent_targets`

### 6.1 它只保留车辆

代码里显式判断：

```python
if name == "vehicle" and _xy_in_lidar(...):
```

所以这一分支的 agent 框监督只针对：

- 在 LiDAR 范围内的车辆

### 6.2 处理步骤

1. 遍历当前帧 annotation box
2. 只取车类目标
3. 过滤出 LiDAR 范围内的框
4. 记录 `(x, y, heading, length, width)`
5. 按距离排序，保留最近的 `max_agents`
6. 不足部分补零

### 6.3 最终 shape

默认 `num_bounding_boxes=30`：

```text
agent_states: [30, 5]
agent_labels: [30]
```

其中 `agent_labels[i] = True` 表示该位置有真实目标。

---

## 7. `_compute_bev_semantic_map`：BEV 语义监督怎么做

源码位置：

- `_compute_bev_semantic_map`
- `_compute_map_polygon_mask`
- `_compute_map_linestring_mask`
- `_compute_box_mask`
- `_coords_to_pixel`

### 7.1 它融合了两类来源

1. 地图 API
2. 当前帧 annotation

### 7.2 语义类别来自配置

配置里定义了：

- road
- walkways
- centerline
- static_objects
- vehicles
- pedestrians

### 7.3 处理流程

对于每个类别：

1. 如果是 polygon 类地图要素，就画 polygon mask
2. 如果是 linestring 类地图要素，就画 polyline mask
3. 如果是 box 类实体，就根据 annotation 画旋转框 mask

最后得到：

```text
bev_semantic_map: [128, 256]
```

其中：

- `bev_pixel_height = 128`
- `bev_pixel_width = 256`

### 7.4 一个值得注意的小细节

代码里最后写的是：

```python
return torch.Tensor(bev_semantic_map)
```

这会把原本的整数类别图转成 float tensor。训练时如果后续 loss 期望 long type，就要在别处再处理。

这不一定有问题，但值得你读 loss 时留意。

---

## 8. 坐标系统：为什么这份文件很重要

这份文件把很多“默认假设”都固化进来了：

### 8.1 相机

- 只取当前时刻
- 只取前向三相机
- resize 成固定输入大小

### 8.2 LiDAR

- 只取当前时刻
- 投成 BEV raster
- 坐标范围由 config 决定

### 8.3 轨迹目标

- 用局部 ego 坐标表示
- 默认 8 个未来点

### 8.4 周围车目标

- 只保留最近 30 个 vehicle

也就是说，你在后面模型文件里看到的一切 shape，几乎都能追溯回这份文件。

---

## 9. 这份文件最该记住的 6 个结论

1. 模型输入不是原始多相机序列，而是三路前向相机拼接图。
2. LiDAR 输入不是点级表示，而是二维 BEV histogram。
3. `status_feature` 是 8 维 ego 动态状态，不是位姿序列。
4. 轨迹监督 `trajectory` 的 shape 是 `[8, 3]`。
5. 目标框监督只保留最近的 vehicle，shape 是 `[30, 5]`。
6. 这份文件决定了后面模型里所有输入字典键名和基础 shape。

---

## 10. 读完它之后建议立刻跳到哪

建议下一步读：

1. `navsim/planning/training/dataset.py`
2. `navsim/planning/script/run_training.py`

因为这两份文件会告诉你：

- 这些 feature/target 是怎么被缓存和加载的
- 训练入口到底怎么把它们喂给 agent 和模型
