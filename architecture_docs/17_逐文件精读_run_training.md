# 17 逐文件精读：`run_training.py`

本文对应源码：

- `navsim/planning/script/run_training.py`

相关联文件建议同时打开：

- `navsim/planning/training/agent_lightning_module.py`
- `navsim/planning/training/dataset.py`
- `navsim/planning/script/config/training/default_training.yaml`
- `navsim/planning/script/config/common/agent/diffusiondrivev2_rl_agent.yaml`
- `navsim/planning/script/config/common/agent/diffusiondrivev2_sel_agent.yaml`

这份文件很短，但它是训练入口主开关。你要读懂的不是代码难度，而是：

- 谁实例化谁
- 数据从哪里来
- batch 如何送进 agent
- Lightning 负责什么，agent 又负责什么

---

## 1. 这份文件只做两件事

1. `build_datasets(cfg, agent)`
2. `main(cfg)`

所以读它时不要陷入“只有几十行，好像没什么内容”的错觉。

它真正的作用是：

- 把 Hydra 配置树翻译成一套可运行的训练对象图

---

## 2. `build_datasets(cfg, agent)`：把配置变成数据集

源码位置：

- `def build_datasets(cfg, agent)`

### 2.1 先实例化两个 `SceneFilter`

函数一开始会各自构造：

- `train_scene_filter`
- `val_scene_filter`

然后根据：

- `cfg.train_logs`
- `cfg.val_logs`

重新筛一遍 `log_names`。

这说明：

- 同一个基础 filter 模板
- 会被分别约束成 train / val 两份

### 2.2 再构造两个 `SceneLoader`

接下来用：

- `cfg.navsim_log_path`
- `cfg.sensor_blobs_path`
- `agent.get_sensor_config()`

分别构造：

- `train_scene_loader`
- `val_scene_loader`

这里有一个很重要的设计点：

- 入口脚本并不知道模型到底用哪些传感器
- 传感器配置由 agent 决定

### 2.3 最后构造两个 `Dataset`

真正决定 feature / target 内容的，不是 `run_training.py`，而是：

- `agent.get_feature_builders()`
- `agent.get_target_builders()`

所以 `run_training.py` 的角色更像：

- 通用训练组装器

### 2.4 学习版代码摘录

```python
def build_datasets(cfg, agent):
    # 1. 先从 Hydra 配置里实例化场景过滤器
    train_scene_filter = instantiate(cfg.train_test_split.scene_filter)
    val_scene_filter = instantiate(cfg.train_test_split.scene_filter)

    # 2. 再用 train_logs / val_logs 限定真正使用哪些日志
    train_scene_filter.log_names = cfg.train_logs
    val_scene_filter.log_names = cfg.val_logs

    # 3. SceneLoader 负责把原始日志组织成 scene / token
    train_scene_loader = SceneLoader(..., scene_filter=train_scene_filter,
                                     sensor_config=agent.get_sensor_config())
    val_scene_loader = SceneLoader(..., scene_filter=val_scene_filter,
                                   sensor_config=agent.get_sensor_config())

    # 4. 真正的 feature / target 由 agent 提供的 builders 决定
    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )
```

---

## 3. `main(cfg)`：训练总控流程

源码位置：

- `def main(cfg)`

### 3.1 执行顺序一眼看懂版

```text
Hydra 读配置
  -> seed_everything
  -> instantiate(cfg.agent)
  -> AgentLightningModule(agent)
  -> Dataset 或 CacheOnlyDataset
  -> DataLoader
  -> pl.Trainer(...)
  -> trainer.fit(...)
```

### 3.2 学习版代码摘录

```python
def main(cfg):
    # 1. 先固定随机种子，保证 worker 和训练过程可复现
    pl.seed_everything(cfg.seed, workers=True)

    # 2. Hydra 根据 cfg.agent._target_ 实例化具体 agent
    agent = instantiate(cfg.agent)

    # 3. LightningModule 只是训练外壳，真正 forward/loss 在 agent 内部
    lightning_module = AgentLightningModule(agent=agent)

    # 4. 根据配置决定：在线构建 SceneLoader，还是直接吃缓存
    if cfg.use_cache_without_dataset:
        train_data = CacheOnlyDataset(...)
        val_data = CacheOnlyDataset(...)
    else:
        train_data, val_data = build_datasets(cfg, agent)

    # 5. 用统一的 DataLoader 参数包起来
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)

    # 6. Trainer 负责 epoch/ckpt/logging，agent 负责模型和优化器
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())
    trainer.fit(model=lightning_module,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
```

---

## 4. `use_cache_without_dataset` 这个分支非常关键

你一定要注意这段逻辑：

```python
if cfg.use_cache_without_dataset:
    ...
    train_data = CacheOnlyDataset(...)
else:
    train_data, val_data = build_datasets(cfg, agent)
```

### 4.1 它意味着什么

训练可以有两种模式：

#### 模式 A：在线构建场景

- 走 `SceneLoader`
- 走 `Dataset`
- 可以从原始日志动态生成 feature/target

#### 模式 B：直接吃缓存

- 不再创建 `SceneLoader`
- 直接走 `CacheOnlyDataset`
- 适合已经提前缓存好 feature/target 的训练

### 4.2 为什么 DiffusionDriveV2 更依赖缓存模式

因为 RL / selector 阶段除了要 feature/target，还要：

- `metric_cache_path`
- `token`

这些信息在 `CacheOnlyDataset` 里会被一并返回。

---

## 5. 这份文件不负责 loss，也不负责 optimizer

这是很多初学者第一次看入口脚本最容易混淆的地方。

### 5.1 `run_training.py` 负责什么

- 实例化对象
- 连接对象
- 调 `trainer.fit`

### 5.2 不负责什么

- 不负责模型 forward
- 不负责 loss 公式
- 不负责 optimizer 具体参数组
- 不负责 scheduler 细节

这些分别在：

- `AgentLightningModule`
- `Diffusiondrivev2_Rl_Agent`
- `Diffusiondrivev2_Sel_Agent`
- 具体模型文件

里实现。

---

## 6. 真正的 batch 是怎么进入 agent 的

这部分虽然不在 `run_training.py` 里，但你读它时必须顺手跳到：

- `navsim/planning/training/agent_lightning_module.py`

### 6.1 Lightning 里的关键分叉

`AgentLightningModule._step(...)` 里有这样一段逻辑：

```python
if len(batch) == 2:
    features, targets = batch
    prediction = self.agent.forward(features, targets)
else:
    features, targets, pdm_token_path, token = batch
    prediction = self.agent.forward(features, targets, pdm_token_path, token)
```

这意味着：

- 普通监督训练 batch 是 2 元组
- DiffusionDriveV2 RL / selector 训练 batch 是 4 元组

也就是说，`run_training.py` 只是把 DataLoader 建起来，真正区分“普通训练”和“带 reward cache 的训练”是在 dataset + lightning wrapper 里。

---

## 7. Hydra 在这里扮演什么角色

入口函数上面有：

```python
@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
```

这意味着：

- 你运行的不是一个“普通 Python 脚本 + argparse”
- 而是一个 Hydra 配置系统入口

所以命令行里写：

```bash
python navsim/planning/script/run_training.py agent=diffusiondrivev2_rl_agent
```

本质上是在覆写配置树中的：

- `cfg.agent`

### 7.1 对你读代码最重要的结论

训练到底跑 RL agent 还是 selector agent，不是由 `if` 写死的，而是由 Hydra 配置决定的。

---

## 8. “谁决定什么”的职责分配表

### 8.1 `run_training.py`

- 训练入口
- 实例化 agent / dataset / trainer

### 8.2 `Dataset` / `CacheOnlyDataset`

- 返回样本
- 决定 batch 结构

### 8.3 `AgentLightningModule`

- 调 `agent.forward`
- 调 `agent.compute_loss`
- 负责日志记录

### 8.4 `Diffusiondrivev2_Rl_Agent` / `Diffusiondrivev2_Sel_Agent`

- 持有模型
- 定义 optimizer / scheduler
- 决定 feature builder / target builder

### 8.5 具体模型文件

- 真正执行 forward
- 实现采样、reward、loss

---

## 9. 这份文件最该记住的 5 个点

1. 这份文件是训练入口，不是模型逻辑本体。
2. 入口脚本并不知道 feature/target 细节，这些由 agent 提供。
3. 是否直接用缓存，由 `use_cache_without_dataset` 控制。
4. RL / selector 训练之所以能拿到 `metric_cache` 和 `token`，是因为 `CacheOnlyDataset` 返回了 4 元组 batch。
5. Hydra 配置决定了实例化哪个 agent，也就决定了后面整套训练链跑哪一版模型。

---

## 10. 读完它后下一步建议

建议立刻接着读：

1. `navsim/planning/training/dataset.py`
2. `navsim/agents/diffusiondrivev2/transfuser_features.py`

这样你就能把：

```text
run_training.py
  -> Dataset
  -> features / targets
  -> AgentLightningModule
  -> agent.forward
```

这条链完整串起来。
