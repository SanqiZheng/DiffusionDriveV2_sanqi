# DiffusionDriveV2 架构学习文档

## 工程一句话概括

DiffusionDriveV2 是一个面向端到端自动驾驶规划的“锚点约束截断扩散生成器 + 强化学习约束 + 两阶段轨迹选择器”工程，目标是在保留多模态轨迹生成能力的同时，提高生成结果的整体质量与安全性。

## 文档导航

1. [00_代码阅读路线.md](./00_代码阅读路线.md)
2. [01_论文主线与核心思想.md](./01_论文主线与核心思想.md)
3. [02_论文到代码的映射关系.md](./02_论文到代码的映射关系.md)
4. [03_工程总览与架构地图.md](./03_工程总览与架构地图.md)
5. [04_数据流与特征表示.md](./04_数据流与特征表示.md)
6. [05_模型架构与关键组件.md](./05_模型架构与关键组件.md)
7. [06_训练流程与损失函数.md](./06_训练流程与损失函数.md)
8. [07_推理_采样_生成流程.md](./07_推理_采样_生成流程.md)
9. [08_从基础理论到项目算法的数学推导.md](./08_从基础理论到项目算法的数学推导.md)
10. [09_逐文件精读_训练入口与主链路.md](./09_逐文件精读_训练入口与主链路.md)
11. [10_逐文件精读_核心模型模块.md](./10_逐文件精读_核心模型模块.md)
12. [11_逐文件精读_数据处理链路.md](./11_逐文件精读_数据处理链路.md)
13. [12_从输入到loss_或从噪声到样本的端到端总图.md](./12_从输入到loss_或从噪声到样本的端到端总图.md)
14. [13_实现观察_潜在问题与二次开发建议.md](./13_实现观察_潜在问题与二次开发建议.md)
15. [14_逐文件精读_diffusiondrivev2_model_rl.md](./14_逐文件精读_diffusiondrivev2_model_rl.md)
16. [15_逐文件精读_diffusiondrivev2_model_sel.md](./15_逐文件精读_diffusiondrivev2_model_sel.md)
17. [16_逐文件精读_transfuser_features.md](./16_逐文件精读_transfuser_features.md)
18. [17_逐文件精读_run_training.md](./17_逐文件精读_run_training.md)
19. [18_逐文件精读_dataset.md](./18_逐文件精读_dataset.md)

## 推荐阅读顺序

如果你是“Python 新手 + 生成模型初学者”，建议按下面顺序读：

1. 先读 `00_代码阅读路线.md`
2. 再读 `01_论文主线与核心思想.md`
3. 接着读 `03_工程总览与架构地图.md`
4. 然后读 `04_数据流与特征表示.md`
5. 再读 `05_模型架构与关键组件.md`
6. 然后读 `06_训练流程与损失函数.md`
7. 再读 `07_推理_采样_生成流程.md`
8. 对数学还不熟时，再配合 `08_从基础理论到项目算法的数学推导.md`
9. 进入源码精读时，优先读 `17 -> 18 -> 16 -> 14 -> 15`
10. `09` 到 `13` 适合作为宏观地图，`14` 到 `18` 适合作为逐文件对照源码
11. `13_实现观察_潜在问题与二次开发建议.md` 适合作为复盘和二次开发入口

## 论文与代码关系说明

本套文档同时参考了三类材料：

1. 本地论文 PDF `2512.07745v1.pdf`
2. 仓库 `README.md` 与 `docs/train_eval.md`
3. 当前工程真实源码

需要特别注意：

- 文档中的“理论主线”以论文为主。
- 文档中的“实现细节”以当前仓库真实代码为准。
- 若论文表述、README 说明和源码实现不一致，文档会明确写出“论文说了什么，代码实际做了什么”。
- 本项目里实际上有三套相关实现并存：
  - `navsim/agents/diffusiondrive/`：原版 DiffusionDrive
  - `navsim/agents/diffusiondrivev2/`：DiffusionDriveV2 的 RL 版与 selector 版
  - `navsim/planning/...`：NAVSIM 训练、缓存、评测主干

## 学习这套工程时最重要的认识

这不是一个“只有一个模型 forward 就结束”的项目，而是三段式系统：

1. 感知编码与条件构造：相机、LiDAR、ego status 变成 BEV 特征与查询。
2. 轨迹生成器：基于 anchor 的截断扩散，在轨迹空间里生成多模态候选。
3. 训练与评测外环：PDM 模拟器与 scorer 给出闭环指标，RL 和 selector 都围绕这个闭环指标训练。

所以真正要吃透它，必须同时看：

- 论文的动机与训练目标
- 主模型里的采样与损失
- 数据与缓存系统
- PDM 评测接口如何变成 reward

## 你会在源码里反复看到的关键词

- `plan_anchor`
- `DDIMScheduler`
- `DDIMScheduler_with_logprob`
- `TrajectoryHead`
- `forward_train_rl`
- `get_rlloss`
- `PDMScorer`
- `pdm_score_para`
- `coarse scorer`
- `fine scorer`
- `topk`
- `metric_cache`
- `TransfuserBackbone`
- `GridSampleCrossBEVAttention`

## 建议的源码起点

如果你现在就想打开代码，最值得优先读的是这 5 个文件：

1. `navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py`
2. `navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel.py`
3. `navsim/agents/diffusiondrivev2/transfuser_features.py`
4. `navsim/planning/script/run_training.py`
5. `navsim/planning/training/dataset.py`

对应的逐文件精读文档就是：

1. `14_逐文件精读_diffusiondrivev2_model_rl.md`
2. `15_逐文件精读_diffusiondrivev2_model_sel.md`
3. `16_逐文件精读_transfuser_features.md`
4. `17_逐文件精读_run_training.md`
5. `18_逐文件精读_dataset.md`
