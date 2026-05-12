# 18 逐文件精读：`dataset.py`

本文对应源码：

- `navsim/planning/training/dataset.py`

这份文件的价值非常高，因为它解释了：

- 样本是如何从 cache 或原始 scene 变成 batch 的
- 为什么 DiffusionDriveV2 的 batch 会比普通监督训练多两个字段
- feature cache 和 metric cache 之间是怎么接上的

---

## 1. 这份文件里有三种 dataset

### 1.1 `Dataset`

作用：

- 支持在线从 `SceneLoader` 计算 feature/target
- 也支持把结果写到 cache 后再加载

### 1.2 `CacheOnlyDataset`

作用：

- 只从 feature/target cache 读数据
- 额外返回 `pdm_token_path` 和 `token`

这是 DiffusionDriveV2 RL / selector 训练最关键的 dataset。

### 1.3 `CacheOnlyDatasetTest`

作用：

- 评测时同时读取 feature cache 和 metric cache
- 返回 `(features, targets, metric_cache_path, token)`

---

## 2. 两个 gzip helper：cache 文件真正怎么存

源码位置：

- `load_feature_target_from_pickle`
- `dump_feature_target_to_pickle`

### 2.1 文件格式

缓存文件不是 `.pt`，而是：

- `gzip + pickle`

对应文件扩展名：

- `*.gz`

### 2.2 它缓存的是什么

每个 builder 各自缓存一个字典：

- `transfuser_feature.gz`
- `transfuser_target.gz`

这些文件被放在：

```text
cache_path / log_name / token /
```

下面。

---

## 3. `Dataset`：既能在线算，也能读缓存

源码位置：

- `class Dataset`

### 3.1 初始化时做了什么

`Dataset.__init__` 会保存：

- `scene_loader`
- `feature_builders`
- `target_builders`
- `cache_path`
- `force_cache_computation`

然后立刻执行：

- `_load_valid_caches(...)`

如果配置了 `cache_path`，还会继续：

- `cache_dataset()`

### 3.2 `_load_valid_caches` 的语义

它会遍历 `cache_path` 下所有：

- `log/token/`

目录，并检查每个 token 是否已经存在所有 builder 对应的 `.gz` 文件。

只有 feature builders 和 target builders 对应文件都齐全，这个 token 才会被认为是“有效缓存”。

### 3.3 `_cache_scene_with_token`：在线计算并落盘

这个函数的主线很重要：

```text
token
  -> scene_loader.get_scene_from_token(token)
  -> scene.get_agent_input()
  -> feature builders 逐个计算
  -> target builders 逐个计算
  -> 分别写成 .gz
```

### 3.4 学习版代码摘录

```python
def _cache_scene_with_token(self, token):
    # 1. 先从 SceneLoader 取出完整 scene
    scene = self._scene_loader.get_scene_from_token(token)
    agent_input = scene.get_agent_input()

    # 2. 以 log_name / initial_token 为目录写缓存
    token_path = self._cache_path / metadata.log_name / metadata.initial_token
    os.makedirs(token_path, exist_ok=True)

    # 3. 每个 feature builder 各写一个 .gz
    for builder in self._feature_builders:
        data_dict = builder.compute_features(agent_input)
        dump_feature_target_to_pickle(token_path / (builder.get_unique_name() + ".gz"), data_dict)

    # 4. 每个 target builder 也各写一个 .gz
    for builder in self._target_builders:
        data_dict = builder.compute_targets(scene)
        dump_feature_target_to_pickle(token_path / (builder.get_unique_name() + ".gz"), data_dict)
```

### 3.5 `__getitem__` 的双模式逻辑

如果有 `cache_path`：

- 直接从缓存读

如果没有 `cache_path`：

- 现场调用 builder 计算

所以 `Dataset` 是一个“可缓存的通用 dataset”。

---

## 4. `CacheOnlyDataset`：DiffusionDriveV2 训练真正依赖的版本

源码位置：

- `class CacheOnlyDataset`

### 4.1 它为什么重要

因为 RL / selector 训练不只需要：

- `features`
- `targets`

还需要：

- `pdm_token_path`
- `token`

而这些就是 `CacheOnlyDataset.__getitem__` 返回的额外内容。

### 4.2 返回值不是 2 元组，而是 4 元组

```python
return (features, targets, pdm_token_path, token)
```

这就是为什么后面的 `AgentLightningModule` 要写：

```python
if len(batch) == 2:
    ...
else:
    features, targets, pdm_token_path, token = batch
```

### 4.3 `pdm_token_path` 是怎么推出来的

如果当前 cache 路径里含有：

- `"training_cache"`

代码会把它替换成：

- `"train_pdm_cache"`

然后插入一层：

- `"unknown"`

最后拼成：

```text
... / train_pdm_cache / log_name / unknown / token / metric_cache.pkl
```

这说明 feature cache 和 metric cache 在磁盘上是两套平行目录。

### 4.4 学习版代码摘录

```python
def _load_scene_with_token(self, idx):
    token = self.tokens[idx]
    token_path = self._valid_cache_paths[token]

    if "training_cache" in str(token_path):
        # 训练时从 feature cache 路径推导出对应的 PDM metric cache 路径
        pdm_token_path = str(token_path).replace("training_cache", "train_pdm_cache")
        pdm_token_path_parts = pdm_token_path.split("/")
        pdm_token_path_parts.insert(-1, "unknown")
        pdm_token_path = "/".join(pdm_token_path_parts) + "/metric_cache.pkl"
    else:
        pdm_token_path = token_path

    # 再分别读 feature / target 的 .gz
    ...
    return (features, targets, pdm_token_path, token)
```

### 4.5 一个很重要的结论

从这里你能直接看出：

- RL/selector 的 reward 并不是 dataset 现场算的
- 而是依赖预先准备好的 `metric_cache.pkl`

---

## 5. `CacheOnlyDatasetTest`：评测时为什么又单独写一个类

源码位置：

- `class CacheOnlyDatasetTest`

### 5.1 它和 `CacheOnlyDataset` 的差别

它不再从 feature cache 路径“推导” metric cache 路径，而是显式传入：

- `feature_cache_path`
- `metric_cache_path`

### 5.2 为什么这样更合理

评测时：

- feature cache 和 metric cache 的目录未必和训练时完全同构
- 分开传路径更安全，也更容易做离线验证

### 5.3 返回值

```python
return (features, targets, metric_cache_path, token)
```

和训练版很像，只是路径来源不同。

---

## 6. 这份文件决定了 batch 长什么样

### 6.1 普通监督训练

如果 DataLoader 用的是 `Dataset`，则 batch 语义是：

```text
(features, targets)
```

### 6.2 DiffusionDriveV2 RL / selector 训练

如果 DataLoader 用的是 `CacheOnlyDataset`，则 batch 语义是：

```text
(features, targets, pdm_token_path, token)
```

### 6.3 为什么这点非常重要

因为后面：

- `AgentLightningModule`
- `Diffusiondrivev2_Rl_Agent`
- `Diffusiondrivev2_Sel_Agent`

都默认建立在这个 batch 约定上。

如果你自己改 dataset 而没同步改这些模块，训练会立刻断。

---

## 7. cache 目录结构应该怎么理解

### 7.1 feature/target cache

典型结构：

```text
training_cache/
  log_name/
    token/
      transfuser_feature.gz
      transfuser_target.gz
```

### 7.2 metric cache

典型结构：

```text
train_pdm_cache/
  log_name/
    unknown/
      token/
        metric_cache.pkl
```

这个目录结构解释了为什么 `CacheOnlyDataset` 里有一段字符串替换逻辑。

---

## 8. 这份文件里值得特别注意的 3 个实现观察

### 8.1 token 键只用 `token_path.name`

无论 `Dataset` 还是 `CacheOnlyDataset`，内部 `_valid_cache_paths` 的 key 都是：

- `token_path.name`

也就是只用 token 字符串本身，不带 log 名。

如果不同 log 之间存在重名 token，就可能产生覆盖风险。

### 8.2 `CacheOnlyDataset` 里 `metric_cache_loader` 基本没被主链使用

代码里如果 `cache_path` 包含 `"metric"`，会创建：

- `self.metric_cache_loader = MetricCacheLoader(...)`

但这条对象并没有成为主训练链的关键输入，主逻辑仍然是通过 `pdm_token_path` 传文件路径。

### 8.3 类型注解和真实返回值不完全一致

一些 docstring / type hint 看起来仍像“返回 feature 和 target 二元组”，但 `CacheOnlyDataset` 实际返回的是四元组。

读代码时要以真实返回值为准。

---

## 9. 这份文件最该记住的 6 个结论

1. `Dataset` 是通用版，支持在线计算和缓存。
2. `CacheOnlyDataset` 是 DiffusionDriveV2 RL / selector 训练真正依赖的版本。
3. RL / selector 训练之所以能拿到 reward cache，是因为 dataset 额外返回了 `pdm_token_path`。
4. feature cache 与 metric cache 是两套平行目录。
5. DataLoader 输出几元组，不是由模型决定，而是由 dataset 决定。
6. 如果你要二次开发训练链，这份文件通常是最早要改的地方之一。

---

## 10. 推荐怎么和其他文件联动着读

建议联动阅读顺序：

1. `navsim/agents/diffusiondrivev2/transfuser_features.py`
2. `navsim/planning/training/dataset.py`
3. `navsim/planning/script/run_training.py`
4. `navsim/planning/training/agent_lightning_module.py`

这样你会得到一条很清楚的数据主线：

```text
Scene / AgentInput
  -> feature builder / target builder
  -> cache 文件
  -> Dataset / CacheOnlyDataset
  -> DataLoader batch
  -> LightningModule
  -> agent.forward(...)
```
