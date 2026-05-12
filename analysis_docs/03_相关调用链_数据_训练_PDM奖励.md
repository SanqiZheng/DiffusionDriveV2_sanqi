# 03. 相关调用链：数据、训练、PDM 奖励

这一篇负责回答一个问题：

“主文件里的输入是从哪来的，loss 又是怎么被外层训练框架消费掉的？”

## 1. 训练入口

训练入口在：

- [run_training.py](../navsim/planning/script/run_training.py)

关键函数：

- `build_datasets` 第 23 行
- `main` 第 80 行

主链如下：

1. Hydra 配置实例化 agent
2. 用 agent 提供的 feature builder 和 target builder 构建 dataset
3. DataLoader 取 batch
4. `AgentLightningModule.training_step`
5. `agent.forward(...)`
6. `agent.compute_loss(...)`
7. Lightning 用返回的 `loss` 反传

## 2. agent 层：真正被训练的是谁

文件：

- [diffusiondrivev2_rl_agent.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_rl_agent.py)

### 2.1 `__init__`

最重要的是这几行逻辑：

1. 创建 `self._transfuser_model`
2. 把不属于 `_trajectory_head` 的参数全部 `requires_grad=False`
3. 把非 `_trajectory_head` 模块切到 `eval()`
4. 只让 `_trajectory_head.train()`

这意味着：

- RL 训练不是全模型 end-to-end 微调
- 而是冻结感知与大部分表示层，只优化轨迹生成头

开发者为什么这样做：

- 感知 backbone 训练成本高
- RL 信号噪声大
- 如果全模型一起抖，容易训练崩掉

### 2.2 `forward`

位置：

- 第 116 行

它只是把 batch 转发给：

- `self._transfuser_model(features, targets=targets, eta=1.0, metric_cache=metric_cache, token=token)`

注意这里固定传了：

- `eta = 1.0`

这意味着训练时更接近带随机性的 DDPM 风格采样。

### 2.3 `compute_loss`

位置：

- 第 120 行

这里非常薄：

- 不自己重新算 loss
- 直接读取 `predictions["loss"]`
- 再把 `reward`、`sub_rewards` 打到日志里

也就是说，真正的 loss 逻辑完全在 `TrajectoryHead.get_rlloss`。

## 3. 配置系统怎么把这些东西组起来

关键文件：

- [diffusiondrivev2_rl_agent.yaml](../navsim/planning/script/config/common/agent/diffusiondrivev2_rl_agent.yaml)
- [diffusiondrivev2_rl_config.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_rl_config.py)

### 3.1 Hydra 实例化 agent

YAML 里指定：

- `_target_: navsim.agents.diffusiondrivev2.diffusiondrivev2_rl_agent.Diffusiondrivev2_Rl_Agent`

其中 `config` 会实例化成：

- `navsim.agents.diffusiondrivev2.diffusiondrivev2_rl_config.TransfuserConfig`

### 3.2 一个容易踩坑的点

主模型文件里 import 的类型注解来自：

- [navsim/agents/diffusiondrive/transfuser_config.py](../navsim/agents/diffusiondrive/transfuser_config.py)

但真正运行时传进去的 config 实例来自：

- [navsim/agents/diffusiondrivev2/diffusiondrivev2_rl_config.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_rl_config.py)

因为后者多了：

- `num_groups`

而 `TrajectoryHead` 在第 730 行附近明确依赖它。

所以这是一种“类型注解来自老配置，运行配置来自新配置”的写法，静态阅读时需要特别注意。

## 4. 数据如何进入模型

### 4.1 `TransfuserFeatureBuilder`

文件：

- [transfuser_features.py](../navsim/agents/diffusiondrivev2/transfuser_features.py)

关键函数：

- `compute_features` 第 39 行
- `_get_camera_feature` 第 55 行
- `_get_lidar_feature` 第 77 行

#### 相机特征

做法：

1. 取 `cam_l0 / cam_f0 / cam_r0`
2. 左右图裁切，中间图保留
3. 横向拼接
4. resize 到 `1024 x 256`
5. `ToTensor`

结果：

- 单帧、三通道、前视拼接图

#### LiDAR 特征

做法：

1. 取最后一帧激光点云
2. 只保留 `(x, y, z)`
3. 按高度切成 above / below
4. 在 BEV 平面做 histogram splat
5. 归一化到 `[0, 1]`

若 `use_ground_plane=False`：

- 输出 shape 约为 `[1, 256, 256]`

这说明 LiDAR 输入不是点级 transformer，而是经典 BEV occupancy / density 图。

#### ego 状态

直接拼：

- `driving_command`
- `ego_velocity`
- `ego_acceleration`

结果是 `8` 维向量。

### 4.2 `TransfuserTargetBuilder`

关键函数：

- `compute_targets` 第 135 行

输出包括：

- `trajectory`
- `agent_states`
- `agent_labels`
- `bev_semantic_map`

其中：

- `trajectory` 是未来 `8` 个 pose，来自 `scene.get_future_trajectory(...)`
- 这条轨迹主要用于 IL 辅助损失与日志

## 5. dataset 怎么把 `metric_cache` 送进 batch

文件：

- [dataset.py](../navsim/planning/training/dataset.py)

关键类：

- `CacheOnlyDataset`
- `CacheOnlyDatasetTest`

### 5.1 训练 batch 不只有 features 和 targets

`CacheOnlyDataset._load_scene_with_token` 在第 115 行开始返回：

- `(features, targets, pdm_token_path, token)`

也就是说，训练 batch 里除了普通监督信号，还带着：

- 当前样本对应的 `metric_cache.pkl` 路径
- `token`

### 5.2 为什么不直接把 metric_cache 对象放进 batch

因为：

- `metric_cache` 对象很重
- DataLoader 多进程传输会更慢
- 这里把路径传进去，真正算 reward 时再在子进程里懒加载

这是很典型的工程优化。

## 6. Lightning 层怎么消费这个 batch

文件：

- [agent_lightning_module.py](../navsim/planning/training/agent_lightning_module.py)

关键函数：

- `_step` 第 20 行

逻辑：

1. 如果 batch 长度为 4，就拆成 `features, targets, pdm_token_path, token`
2. 调 `self.agent.forward(features, targets, pdm_token_path, token)`
3. 调 `self.agent.compute_loss(...)`
4. 把每个 loss/reward 指标都 log 出去

所以从训练框架视角看，它并不知道 diffusion、PDM、GRPO 的细节；这些都被 agent/model 封装起来了。

## 7. backbone 到 query 的调用链

文件：

- [transfuser_backbone.py](../navsim/agents/diffusiondrivev2/transfuser_backbone.py)

### 7.1 backbone 的角色

它做的是：

- 图像编码
- LiDAR 编码
- 多尺度 image-lidar fusion

不是直接出轨迹，而是出场景表征。

### 7.2 `fuse_features`

位置：

- 第 221 行

做法：

1. 对图像特征和 LiDAR 特征分别做自适应池化
2. 送进 `GPT` 模块做 token 级交互
3. 再把融合结果插值回原尺度
4. 残差加回原 feature

直觉：

- 相机擅长语义
- LiDAR 擅长几何
- 先在压缩 token 上融合，再回写到 dense feature map

## 8. PDM reward 的完整链路

### 8.1 `get_pdm_score_para`

位置：

- [diffusiondrivev2_model_rl.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py) 第 820 行

它把每个 batch 样本提交到多进程池：

- 子进程运行 `_pdm_worker`

### 8.2 `_pdm_worker`

做三件事：

1. 从 `metric_cache_path` 反序列化 `MetricCache`
2. 调 `pdm_score_para(...)`
3. 再从 scorer 内部提取 pairwise score 和 subscore

### 8.3 `pdm_score_para`

文件：

- [pdm_score.py](../navsim/evaluate/pdm_score.py) 第 205 行

流程：

1. 把模型轨迹转成 `Trajectory` / `InterpolatedTrajectory`
2. 转成统一 state array
3. 调 `PDMSimulator.simulate_proposals`
4. 调 `PDMScorer.score_proposals`

### 8.4 `PDMSimulator`

文件：

- [pdm_simulator.py](../navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py)

作用：

- 用 `BatchLQRTracker + BatchKinematicBicycleModel` 模拟 proposal 真正执行后的状态

物理意义：

- 不是只看你“想去哪”
- 而是看这条轨迹在车辆动力学下“真的怎么走”

### 8.5 `PDMScorer`

文件：

- [pdm_scorer.py](../navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py)

评分分两层：

第一层，乘法型硬约束：

- `NO_COLLISION`
- `DRIVABLE_AREA`

第二层，加权型软指标：

- `PROGRESS`
- `TTC`
- `COMFORTABLE`
- `DRIVING_DIRECTION`

聚合时先做：

`multi_metrics.prod(axis=0)`

再做 weighted average。

这意味着：

- 只要硬安全指标挂了，最终分会被直接压低

## 9. 训练链总结

把外层所有文件拼起来，一条训练样本真正经过的路径是：

`run_training.py`

-> `CacheOnlyDataset.__getitem__`

-> `AgentLightningModule._step`

-> `Diffusiondrivev2_Rl_Agent.forward`

-> `V2TransfuserModel.forward`

-> `TrajectoryHead.forward_train_rl`

-> `get_pdm_score_para`

-> `_pdm_worker`

-> `pdm_score_para`

-> `PDMSimulator.simulate_proposals`

-> `PDMScorer.score_proposals`

-> `TrajectoryHead.get_rlloss`

-> 返回 `loss`

这条链吃透之后，你就知道 reward 是怎么从外部 simulator 一层层传回模型的。
