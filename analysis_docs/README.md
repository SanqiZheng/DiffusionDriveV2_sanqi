# DiffusionDriveV2 源码精读文档导航

这套文档围绕主文件 [navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py) 展开，目标不是只“总结论文”，也不是只“翻译代码”，而是把下面三件事接起来：

1. 扩散模型在这个项目里到底学什么。
2. 训练时 reward 是怎么一步步变成 loss 的。
3. 代码里每个关键函数为什么要这么写。

## 推荐阅读顺序

1. [01_先建立整体直觉.md](./01_先建立整体直觉.md)
2. [02_diffusiondrivev2_model_rl_主文件精读.md](./02_diffusiondrivev2_model_rl_主文件精读.md)
3. [03_相关调用链_数据_训练_PDM奖励.md](./03_相关调用链_数据_训练_PDM奖励.md)
4. [04_数学原理_扩散_RL_与代码对应.md](./04_数学原理_扩散_RL_与代码对应.md)
5. [05_实现观察_潜在坑点_与二次开发建议.md](./05_实现观察_潜在坑点_与二次开发建议.md)

## 一句话概括这个 RL 版本

它本质上是一个“冻结大部分感知 backbone，只训练轨迹扩散头”的端到端自动驾驶规划器：先从多组 anchor 轨迹出发做截断扩散采样，再用 PDM closed-loop scorer 给采样轨迹打 reward，最后用类似 policy gradient 的方式更新去噪策略。

## 读源码前先记住 5 个结论

1. 这不是最常见的“预测噪声 epsilon”的 diffusion，而是直接预测 `x_start`。源码证据在 [diffusiondrivev2_model_rl.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py) 第 716-728 行附近，`prediction_type="sample"`。
2. 它不是从纯高斯白噪声开始采样，而是从 `20` 条 KMeans 轨迹 anchor 的邻域开始采样，再乘以 `num_groups` 做每个 anchor 的多次探索。
3. 训练的 reward 不是手工分类标签，而是由 PDM simulator + PDM scorer 跑出来的闭环指标。
4. 训练时真正更新的几乎只有 `_trajectory_head`，其余 TransFuser 主干都被冻结。源码证据在 [diffusiondrivev2_rl_agent.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_rl_agent.py) 第 59-68 行附近。
5. 代码里虽然 import 了 `ConditionalUnet1D`、`LossComputer`、`transfuser_loss`，但 RL 主链路并不靠这些完成训练，而是靠 `CustomTransformerDecoder + DDIMScheduler_with_logprob + PDM reward + get_rlloss`。

## 文档覆盖范围

重点覆盖这些文件：

- [navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py)
- [navsim/agents/diffusiondrivev2/diffusiondrivev2_rl_agent.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_rl_agent.py)
- [navsim/agents/diffusiondrivev2/diffusiondrivev2_rl_config.py](../navsim/agents/diffusiondrivev2/diffusiondrivev2_rl_config.py)
- [navsim/agents/diffusiondrivev2/transfuser_backbone.py](../navsim/agents/diffusiondrivev2/transfuser_backbone.py)
- [navsim/agents/diffusiondrivev2/transfuser_features.py](../navsim/agents/diffusiondrivev2/transfuser_features.py)
- [navsim/agents/diffusiondrive/modules/blocks.py](../navsim/agents/diffusiondrive/modules/blocks.py)
- [navsim/evaluate/pdm_score.py](../navsim/evaluate/pdm_score.py)
- [navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py](../navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py)
- [navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py](../navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py)
- [navsim/planning/training/dataset.py](../navsim/planning/training/dataset.py)
- [navsim/planning/training/agent_lightning_module.py](../navsim/planning/training/agent_lightning_module.py)
- [navsim/planning/script/run_training.py](../navsim/planning/script/run_training.py)

## 你应该优先盯住的 5 个函数

1. `V2TransfuserModel.forward`
2. `TrajectoryHead.forward_train_rl`
3. `DDIMScheduler_with_logprob.step`
4. `TrajectoryHead.get_rlloss`
5. `pdm_score_para`

## 最重要的阅读提醒

代码里的注释经常把 `metric_cache` 里的参考轨迹叫成“GT”，但从 [metric_cache_processor.py](../navsim/planning/metric_caching/metric_cache_processor.py) 第 217-229 行附近看，它实际来自 `PDM-Closed` planner 轨迹，不是简单意义上的人工标注 future trajectory。读 reward 逻辑时一定要区分：

- `targets["trajectory"]`: 监督学习用未来轨迹
- `metric_cache.trajectory`: PDM scorer 的参考轨迹
- `diffusion_output`: 模型当前采样出来的候选轨迹
