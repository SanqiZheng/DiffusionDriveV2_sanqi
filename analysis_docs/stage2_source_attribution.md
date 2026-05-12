# DiffusionDriveV2 第二阶段代码来源归因（本地严格比对）

## 0. 比对方法与判定规则

本报告基于**本地仓库逐文件比对**，不是记忆判断。主要方法：

1. `git diff --no-index` 同路径/同职责文件对比。
2. 类名、函数名、接口签名对齐。
3. import 链与调用链对齐。
4. 稀有标识符反查（如 `gen_sineembed_for_position`、`DDIMScheduler`、`BatchLQRTracker`）。
5. 目录组织与配置结构对齐。

说明：
- 多个仓库存在行尾差异（CRLF/LF）。本报告在关键比对中统一使用 `--ignore-cr-at-eol`，避免把行尾格式误判成代码改动。
- 判断类型定义：`直接继承` / `轻改` / `深改` / `原创` / `第三方 vendored` / `推断`。

---

## 1. 参考工程总表

| 工程 | 本地路径 | 仓库链接 | 在当前工程中的作用 |
|---|---|---|---|
| DiffusionDriveV2 | `/home/yihang/Downloads/CodeReference/diff_based/DiffusionDriveV2` | https://github.com/hustvl/DiffusionDriveV2 | 当前目标工程主体 |
| DiffusionDrive | `/home/yihang/Downloads/CodeReference/diff_based/DiffusionDrive` | https://github.com/hustvl/DiffusionDrive | 当前 `diffusiondrivev2` 目录最直接上游（大量同文件直接继承） |
| NAVSIM | `/home/yihang/Downloads/CodeReference/E2E_classic/navsim` | https://github.com/autonomousvision/navsim | 训练/评测/规划框架祖系，上游基线接口来源 |
| nuPlan-devkit | `/home/yihang/Downloads/CodeReference/E2E_classic/nuplan-devkit` | https://github.com/motional/nuplan-devkit | LQR/控制器算法来源（经 PDM 适配） |
| diffusion_policy | `/home/yihang/Downloads/CodeReference/flow_based/generation/diffusion_policy` | https://github.com/real-stanford/diffusion_policy | `ConditionalUnet1D` 祖系实现来源 |
| carla_garage | `/home/yihang/Downloads/CodeReference/E2E_classic/carla_garage` | https://github.com/autonomousvision/carla_garage | TransFuser LiDAR histogram 特征实现来源 |
| DAB-DETR | `/home/yihang/Downloads/CodeReference/E2E_classic/DAB-DETR` | https://github.com/IDEA-Research/DAB-DETR | sine positional embedding 及注意力相关实现来源 |
| diffusers | `/home/yihang/Downloads/CodeReference/diff_based/diffusers` | https://github.com/huggingface/diffusers | `DDIMScheduler` 基类与 step 推导来源 |

---

## 2. 逐文件来源归因（核心文件）

> 说明：以下 `+x/-y` 为相对“主对照上游”的净改动规模（忽略 CRLF 差异）。

| 当前文件 | 主对照上游文件 | 次级来源链 | 改动规模 | 判断类型 | 结论 |
|---|---|---|---|---|---|
| `navsim/agents/diffusiondrivev2/transfuser_backbone.py` | `DiffusionDrive/navsim/agents/diffusiondrive/transfuser_backbone.py` | NAVSIM `transfuser_backbone.py` <- carla_garage TransFuser链 | `+1/-1` | 轻改 | 仅 import 路径切到 `diffusiondrivev2` 命名空间，主体逻辑可按上游原样理解。 |
| `navsim/agents/diffusiondrivev2/transfuser_features.py` | `DiffusionDrive/.../transfuser_features.py` | NAVSIM transfuser + carla_garage data.py | `IDENTICAL` | 直接继承 | 本地几乎无需看，直接看 DiffusionDrive 同文件；注释中保留 carla_garage 来源。 |
| `navsim/agents/diffusiondrivev2/transfuser_loss.py` | `DiffusionDrive/.../transfuser_loss.py` | NAVSIM transfuser_loss | `IDENTICAL` | 直接继承 | 当前工程相对 DiffusionDrive 无新增。 |
| `navsim/agents/diffusiondrivev2/transfuser_callback.py` | `DiffusionDrive/.../transfuser_callback.py` | NAVSIM transfuser_callback | `IDENTICAL` | 直接继承 | 当前工程相对 DiffusionDrive 无新增。 |
| `navsim/agents/diffusiondrivev2/transfuser_config.py` | `DiffusionDrive/.../transfuser_config.py` | NAVSIM transfuser_config | `IDENTICAL` | 直接继承 | 当前工程相对 DiffusionDrive 无新增。 |
| `navsim/agents/diffusiondrivev2/modules/conditional_unet1d.py` | `DiffusionDrive/.../modules/conditional_unet1d.py` | diffusion_policy `conditional_unet1d.py` | `IDENTICAL`（对 DiffusionDrive）; 相对 diffusion_policy `+48/-9` | 第三方 vendored（已在上游完成改写） | 当前工程本身没改；若追根因应对照 diffusion_policy。 |
| `navsim/agents/diffusiondrivev2/modules/multimodal_loss.py` | `DiffusionDrive/.../modules/multimodal_loss.py` | 多模态损失链 | `IDENTICAL` | 直接继承 | 本地无新增。 |
| `navsim/agents/diffusiondrivev2/modules/scheduler.py` | `DiffusionDrive/.../modules/scheduler.py` | 自定义 LR scheduler | `IDENTICAL` | 直接继承 | 本地无新增。 |
| `navsim/agents/diffusiondrivev2/modules/blocks.py` | `DiffusionDrive/.../modules/blocks.py` | DAB-DETR `gen_sineembed_for_position` | `+77/-0` | 深改 + 第三方 vendored | 在原 blocks 基础上新增 `gen_sineembed_for_position_1d` 与 `GridSampleCrossBEVAttentionScorer` 整段（110-186），为 scorer 分支服务。 |
| `navsim/agents/diffusiondrivev2/diffusiondrivev2_sel_agent.py` | `DiffusionDrive/.../transfuser_agent.py` | DiffusionDrive agent 训练接口 | `+57/-24` | 深改 | 主要改点：训练参数冻结策略、loss 汇总口径、WarmupCosLR 训练轮次、模型入口切到 `model_sel`。 |
| `navsim/agents/diffusiondrivev2/diffusiondrivev2_sel_config.py` | `DiffusionDrive/.../transfuser_config.py` | DiffusionDrive config | `+3/-2` | 轻改 | 改动仅资源路径和 `num_groups` 参数。 |
| `navsim/agents/diffusiondrivev2/diffusiondrivev2_rl_agent.py` | `diffusiondrivev2_sel_agent.py` | SEL agent 分支 | `+24/-41` | 深改 | 从 SEL 分支切成 RL 训练范式：冻结策略更激进、loss 字段改为 `loss/reward/sub_rewards`。 |
| `navsim/agents/diffusiondrivev2/diffusiondrivev2_rl_config.py` | `diffusiondrivev2_sel_config.py` | SEL config 分支 | `+1/-1` | 轻改 | 仅 `num_groups: 8 -> 4`。 |
| `navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel.py` | `DiffusionDrive/.../transfuser_model_v2.py` | diffusers + DAB-DETR + PDM链 | `+1143/-95` | 深改（核心创新承载） | 从单纯 diffusion 轨迹头扩展为：并行 PDM 打分、coarse/fine scorer、多头子奖励学习、自定义 DDIM logprob 采样、vocab 拼接与筛选。 |
| `navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py` | `diffusiondrivev2_model_sel.py` | SEL -> RL 策略优化 | `+540/-729` | 深改（核心创新承载） | 重点改成“采样轨迹链 + 优势估计 + 复算 logprob 的策略梯度损失”，形成两阶段 RL 更新。 |
| `navsim/evaluate/pdm_score.py` | `DiffusionDrive/navsim/evaluate/pdm_score.py` | NAVSIM `pdm_score.py` | `+147/-0` | 深改 | 在原 `pdm_score` 上新增批量入口 `pdm_score_para` 与 `_pdm_score_single`，支持一次仿真/打分多轨迹。 |
| `navsim/planning/.../scoring/pdm_scorer.py` | `DiffusionDrive/.../pdm_scorer.py` | NAVSIM PDM scorer | `IDENTICAL` | 直接继承（第三方基座） | 当前工程本身无变更，按上游理解。 |
| `navsim/planning/.../simulation/batch_lqr.py` | `DiffusionDrive/.../batch_lqr.py` | nuplan-devkit `controller/tracker/lqr.py` | `IDENTICAL`（对 DiffusionDrive） | 第三方 vendored（上游已改写） | 本地无新增；来源注释明确基于 nuplan-devkit。 |
| `navsim/planning/.../simulation/batch_lqr_utils.py` | `DiffusionDrive/.../batch_lqr_utils.py` | nuplan-devkit `tracker_utils.py` | `IDENTICAL`（对 DiffusionDrive） | 第三方 vendored（上游已改写） | 本地无新增；来源注释明确基于 nuplan-devkit。 |

---

## 3. 大文件细化（>500行，类级/函数级）

## 3.1 `diffusiondrivev2_model_sel.py`（1606行）

对照上游：`DiffusionDrive/navsim/agents/diffusiondrive/transfuser_model_v2.py`（558行）

类级映射：

| 当前类（行段） | 上游对应 | 归因 | 关键变化 |
|---|---|---|---|
| `V2TransfuserModel` (143-266) | `V2TransfuserModel` (17-139) | 深改 | 接口扩展出 `eta/metric_cache/token` 与 RL/评分链调用入口。 |
| `AgentHead` (267-308) | `AgentHead` (140-181) | 轻改 | 主体保持一致。 |
| `DiffMotionPlanningRefinementModule` (309-354) | 同名类 (182-228) | 轻改 | 保持任务头结构。 |
| `ModulationLayer` (355-395) | 同名类 (229-269) | 轻改 | 主体一致。 |
| `ScorerTransformerDecoderLayer` (396-481) | 无 | 原创/拼接改写 | 新增 scorer cross-bev/cross-agent/cross-ego/self-attn + FFN。 |
| `ScorerTransformerDecoder` (482-511) | 无 | 原创/拼接改写 | 新增 scorer 层堆叠。 |
| `CustomTransformerDecoderLayer` (512-596) | 同名类 (270-349) | 深改 | 继续保留 diffusion decoder，同时与 scorer 体系协同。 |
| `CustomTransformerDecoder` (597-628) | 同名类 (350-381) | 深改 | 输出与上层调用契合 scorer/RL。 |
| `DDIMScheduler_with_logprob` (629-784) | diffusers `scheduling_ddim.py::DDIMScheduler.step` | 深改 + 第三方 vendored | 在 DDIM step 上加入 logprob、乘性噪声采样、`prev_sample`回放支持。 |
| `TrajectoryHead` (785-1606) | `TrajectoryHead` (382-558) | 深改（核心） | 新增 coarse/fine scorer 多头损失、并行 PDM 打分池、vocab 轨迹融合、train/test RL 分支。 |

函数级关键补充：
- 新增 PDM 并行打分辅助：`_pairwise_subscores`(38)、`_pairwise_scores`(75)、`_pdm_worker`(120)、`_init_pool`(138)。
- 新增 scorer 训练核心：`_score_coarse`(978)、`_select_topk`链、`_score_fine_multi`调用链。
- 新增轨迹后处理：`bezier_xyyaw`(1558)。

## 3.2 `diffusiondrivev2_model_rl.py`（1417行）

对照上游：`diffusiondrivev2_model_sel.py`（同仓库）

类级映射：

| 当前类（行段） | 对应来源 | 归因 | 关键变化 |
|---|---|---|---|
| `V2TransfuserModel` (245-433) | SEL 同名类 | 深改 | 前向流程改为“双调用轨迹头”：先 `old_pred` 采样，再 `get_rlloss` 策略更新。 |
| `AgentHead` (434-475) | SEL 同名类 | 轻改 | 检测头基本保持。 |
| `DiffMotionPlanningRefinementModule` (476-521) | SEL 同名类 | 轻改 | 主体保持。 |
| `ModulationLayer` (522-562) | SEL 同名类 | 轻改 | 主体保持。 |
| `CustomTransformerDecoderLayer/Decoder` (563-688) | SEL 同名类 | 轻改 | 扩散解码主干保留。 |
| `DDIMScheduler_with_logprob` (689-893) | SEL 同名类 + diffusers | 深改 | 保留 DDIM 推导并强调 logprob 与乘性噪声机制。 |
| `TrajectoryHead` (894-1417) | SEL `TrajectoryHead` | 深改（核心） | 重写 RL 训练闭环：`forward_train_rl`、`get_rlloss`、优势筛选、IL/RL 混合损失。 |

函数级关键补充：
- `_pairwise_subscores`(64)、`_pairwise_scores`(111) 与 PDM 子指标联动。
- `forward_train_rl`(1039)：采样链 + PDM 奖励 + advantage。
- `get_rlloss`(1276)：固定旧轨迹链回放，复算当前 logprob，形成 policy gradient。

## 3.3 `transfuser_backbone.py`（513行）

对照上游：`DiffusionDrive/navsim/agents/diffusiondrive/transfuser_backbone.py`

类级：
- `TransfuserBackbone` / `GPT` / `SelfAttention` / `Block` / `MultiheadAttentionWithAttention` / `TransformerDecoderLayerWithAttention` / `TransformerDecoderWithAttention` 全部同源。
- 本地唯一有效差异：`TransfuserConfig` import 路径切换。

结论：可按上游原样读。

## 3.4 `pdm_scorer.py`（509行）

对照上游：`DiffusionDrive/.../pdm_scorer.py`

类级：
- `PDMScorerConfig`、`PDMScorer` 本地与上游一致（忽略 CRLF）。

结论：当前工程无本地创新点，可跳过本地文件，直接读上游。

---

## 4. 跳读地图（最重要）

> 目标：尽量用最少行数理解“当前工程相对上游改了什么”。

## A. 核心必读文件

### A1. `navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel.py`
- 对照上游：`DiffusionDrive/.../transfuser_model_v2.py`
- 建议首轮精读：
  - `38-142`：PDM 并行打分辅助函数（分数重算与子分项拆解）
  - `396-509`：新增 scorer decoder 结构
  - `629-783`：`DDIMScheduler_with_logprob`（基于 diffusers 的关键改写）
  - `844-923`：TrajectoryHead 中 coarse/fine scorer 与 PDM pool 初始化
  - `954-977`：forward 路由 + 并行打分入口
  - `1288-1387`：训练主流程（扩散采样 -> PDM打分 -> coarse/fine loss）
  - `1390-1493`：测试主流程（生成 + scorer 选优）
- 可先跳过：
  - 与上游同构的 `AgentHead`/基础 decoder 常规层细节（267-395、512-628）
  - 纯数学工具实现细节（如 `compute_diversity`、`bezier_xyyaw` 可二读）
- 为什么最值得看：
  - 这里是从 DiffusionDrive 单纯 diffusion 规划，升级到“可打分可筛选可并行评测”的核心创新区域。

### A2. `navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py`
- 对照上游：`diffusiondrivev2_model_sel.py`
- 建议首轮精读：
  - `64-181`：PDM score/子指标与 worker 管线
  - `402-425`：两阶段调用 `_trajectory_head` 的 RL 更新框架
  - `689-888`：DDIM step + logprob 计算
  - `1039-1181`：`forward_train_rl`（优势构造、子指标过滤）
  - `1276-1368`：`get_rlloss`（回放轨迹链、复算 logprob、IL/RL 混合）
- 可先跳过：
  - 与 SEL 相同的基础 backbone/query 构造（260-399）
  - `bezier_xyyaw` 几何后处理（1371-1417）
- 为什么最值得看：
  - 这是“方法变更”最密集处，直接决定 RL 训练目标、稳定性和收益来源。

### A3. `navsim/evaluate/pdm_score.py`
- 对照上游：`DiffusionDrive/navsim/evaluate/pdm_score.py`
- 建议首轮精读：
  - `147-202`：`_pdm_score_single`
  - `205-287`：`pdm_score_para`（批量入口）
- 可先跳过：
  - `24-140`（与 DiffusionDrive 旧接口主体一致）
- 为什么最值得看：
  - 该文件把单轨迹评分改成了批量评分入口，直接服务于 RL 训练效率。

### A4. `navsim/agents/diffusiondrivev2/modules/blocks.py`
- 对照上游：`DiffusionDrive/.../modules/blocks.py`
- 建议首轮精读：
  - `20-35`：`gen_sineembed_for_position`（DAB-DETR 改写版）
  - `110-186`：新增 `gen_sineembed_for_position_1d` + `GridSampleCrossBEVAttentionScorer`
- 可先跳过：
  - `42-109`（原有 cross-bev attention 基本继承）
- 为什么最值得看：
  - 这是 scorer 分支的 attention 基建新增点。

## B. 可直接看上游、当前可跳过的文件

这些文件相对 DiffusionDrive 为 `IDENTICAL`（忽略 CRLF），当前工程本地无需精读：

- `navsim/agents/diffusiondrivev2/transfuser_features.py`
- `navsim/agents/diffusiondrivev2/transfuser_loss.py`
- `navsim/agents/diffusiondrivev2/transfuser_callback.py`
- `navsim/agents/diffusiondrivev2/transfuser_config.py`
- `navsim/agents/diffusiondrivev2/modules/conditional_unet1d.py`
- `navsim/agents/diffusiondrivev2/modules/multimodal_loss.py`
- `navsim/agents/diffusiondrivev2/modules/scheduler.py`
- `navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py`
- `navsim/planning/simulation/planner/pdm_planner/simulation/batch_lqr.py`
- `navsim/planning/simulation/planner/pdm_planner/simulation/batch_lqr_utils.py`

建议：直接切到对应上游文件读，不必在当前工程重复读。

## C. 轻改文件（只看几行）

- `transfuser_backbone.py`：仅 `line 13` import 路径改动。
- `diffusiondrivev2_sel_config.py`：`18-19` 路径参数，`37` 新增 `num_groups`。
- `diffusiondrivev2_rl_config.py`：`37` `num_groups` 从 8 改 4。

## D. agent 封装层（读 30~80 行即可）

- `diffusiondrivev2_sel_agent.py`：
  - 重点看 `61-82`（trainable_prefixes 冻结策略）
  - 看 `136-149`（loss 字段聚合）
- `diffusiondrivev2_rl_agent.py`：
  - 重点看 `59-74`（仅 trajectory_head 可训练）
  - 看 `124-134`（loss/reward/sub_rewards 输出口径）

---

## 5. 跨仓库函数级强证据（用于快速溯源）

1. DAB-DETR -> `blocks.py`
- 现文件注释明确 `Mostly copy-paste from ... DAB-DETR`。
- `gen_sineembed_for_position` 与 `DAB_DETR/transformer.py` 同源，当前版改成 `...` 广播写法并保留二维位置主分支。

2. diffusers -> `DDIMScheduler_with_logprob`
- `diffusiondrivev2_model_{sel,rl}.py` 明确 `from diffusers.schedulers import DDIMScheduler`。
- `step()` 结构与 `diffusers/scheduling_ddim.py::DDIMScheduler.step` 同骨架，新增：
  - `prev_sample` 回放输入
  - 乘性噪声采样
  - transition `log_prob` 输出

3. diffusion_policy -> `conditional_unet1d.py`
- 与 `diffusion_policy/model/diffusion/conditional_unet1d.py` 结构同源。
- 主要是把 `conv1d_components` / `positional_embedding` 内联到单文件，便于工程内 vendoring。

4. nuplan-devkit -> `batch_lqr*.py`
- 文件注释显式声明 based on nuplan-devkit。
- 关键函数名与 `nuplan/planning/simulation/controller/tracker/{lqr.py,tracker_utils.py}` 对齐。

5. carla_garage -> `transfuser_features.py`
- 文件内有行级注释引用 `carla_garage/team_code/data.py#L873`。
- `splat_points` + split-height 直方图逻辑同源。

---

## 6. 最终阅读建议（按投入产出比）

1. 先读（核心创新）：
- `diffusiondrivev2_model_rl.py`
- `diffusiondrivev2_model_sel.py`
- `modules/blocks.py`
- `evaluate/pdm_score.py`

2. 次读（训练封装）：
- `diffusiondrivev2_sel_agent.py`
- `diffusiondrivev2_rl_agent.py`
- `diffusiondrivev2_*_config.py`

3. 可跳过本地、直接看上游：
- `transfuser_features/loss/callback/config`
- `modules/conditional_unet1d/multimodal_loss/scheduler`
- `pdm_scorer.py`
- `batch_lqr.py` / `batch_lqr_utils.py`

4. 若只想抓“当前工程相对参考工程真正改了什么”
- 优先看 `model_rl.py` 的 `1039-1181` 与 `1276-1368`
- 再看 `model_sel.py` 的 `629-783` 与 `1288-1387`
- 最后看 `pdm_score.py` 的 `205-287`

