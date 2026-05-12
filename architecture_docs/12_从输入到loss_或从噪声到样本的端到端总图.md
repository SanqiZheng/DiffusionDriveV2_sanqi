# 12 从输入到 loss 或从噪声到样本的端到端总图

这一篇给你两张总图：

1. 从输入到 loss：Stage I RL 训练主链
2. 从输入到最终样本：Stage II selector 推理主链

如果前面的细节已经读得差不多了，这一篇适合用来收束。

## 1. 总图 A：从输入到 loss（RL 训练）

```text
原始场景
  -> SceneLoader
  -> AgentInput / Scene
  -> TransfuserFeatureBuilder
       camera_feature: (B,3,256,1024)
       lidar_feature:  (B,1,256,256)
       status_feature: (B,8)
  -> TransfuserTargetBuilder
       trajectory:     (B,8,3)
       agent_states:   (B,30,5)
       agent_labels:   (B,30)
       bev_map:        (B,128,256)
  -> CacheOnlyDataset
       + metric_cache_path
       + token
  -> AgentLightningModule._step
  -> Diffusiondrivev2_Rl_Agent.forward
  -> V2TransfuserModel.forward
       -> TransfuserBackbone
           bev_feature_upscale: (B,64,64,64)
           bev_feature:         (B,512,8,8)
       -> status_encoding:      (B,256)
       -> keyval:               (B,65,256)
       -> query_out:            (B,31,256)
       -> trajectory_query:     (B,1,256)
       -> agents_query:         (B,30,256)
       -> with no_grad:
            TrajectoryHead.forward_train_rl
             -> anchor repeat
             -> truncated noisy trajectories
             -> 10-step rollout
             -> PDM reward
             -> advantages
       -> TrajectoryHead.get_rlloss
             -> replay old chain
             -> recompute log_prob
             -> RL loss
             -> IL trajectory L1
             -> final loss
  -> agent.compute_loss
  -> Lightning log + backward
```

## 2. 把总图 A 再拆成 8 个你应该真正脑补的阶段

### 阶段 1：场景切片

单位不是单帧，而是：

- 4 帧历史
- 10 帧未来

### 阶段 2：特征构造

模型真正吃的是：

- 三路前向相机拼图
- 单通道 LiDAR BEV histogram
- 8 维 ego status

### 阶段 3：场景编码

backbone 负责把：

- image
- lidar

融合成：

- 低分辨率 token memory
- 高分辨率 BEV feature map

### 阶段 4：query 解码

learnable query 被分成：

- 1 个 ego query
- 30 个 agent query

### 阶段 5：anchor-conditioned diffusion rollout

从每个 anchor 的多个采样链出发，生成候选轨迹。

### 阶段 6：闭环 reward

候选轨迹不是直接拿来和 GT 点对点算分，而是：

- 先变成 `Trajectory`
- 经 `PDM simulator`
- 再经 `PDM scorer`

### 阶段 7：advantage

先在同 anchor 内部做 group-relative standardization，  
再做 positive truncation 和 failure penalty。

### 阶段 8：RL + IL 反向传播

最终更新的不是整个网络，而是：

- `_trajectory_head`

---

## 3. 总图 B：从输入到最终轨迹（selector 推理）

```text
scene feature
  -> V2TransfuserModel.forward
  -> backbone + query decoder
  -> TrajectoryHead.forward_test_rl
       -> plan_anchor repeat
       -> truncated noise
       -> 2-step diffusion denoise
       -> candidate trajectories
       -> add_mul_noise augmentation
       -> bezier_xyyaw
       -> _get_scorer_inputs
            noisy_traj_points_xy
            traj_feature
       -> coarse scorer
            final_coarse_reward
            best_coarse_flat
       -> top-k = 32
       -> fine scorer (3 layers)
            best fine traj per layer
       -> traj_to_score = [coarse_best, fine_best_0, fine_best_1, fine_best_2]
       -> offline eval only:
            get_pdm_score_para(traj_to_score, metric_cache)
            reward_dict
```

## 4. 从噪声到样本的 shape 演化

下面这一段最适合你对着 `TrajectoryHead.forward_test_rl` 看。

### 4.1 初始 anchor

```text
plan_anchor: (20, 8, 2)
```

### 4.2 batch 扩展

```text
plan_anchor -> (B, num_groups * 20, 8, 2)
```

### 4.3 归一化并加噪

```text
diffusion_output: (B, G_all, 8, 2)
```

### 4.4 位置编码

```text
noisy_traj_points -> sine embedding -> traj_feature: (B, G_all, 256)
```

### 4.5 decoder 输出

```text
poses_reg: (B, G_all, 8, 3)
poses_cls: (B, num_groups, 20)
```

注意：

- `poses_reg[..., :2]` 才是 diffusion 主预测对象
- `heading` 后面主要靠 `bezier_xyyaw` 补全

### 4.6 final candidates

```text
diffusion_output -> denorm -> bezier_xyyaw -> (B, G_all, 8, 3)
```

---

## 5. 条件信息在端到端图里怎么穿行

## 5.1 相机与 LiDAR

在 backbone 里融合，变成：

- `bev_feature`
- `cross_bev_feature`

## 5.2 ego status

先编码成：

- `status_encoding`

再作为额外 token 拼进 memory。

## 5.3 轨迹条件

当前 noisy trajectory 自己也会反过来变成 query feature，这一点非常重要：

- 轨迹不是只被预测
- 它还是每一步 decoder 的输入条件

所以这个系统更像：

- “轨迹迭代 refinement”

而不是：

- 一次性回归未来轨迹

---

## 6. 两种“loss”其实作用在不同层级

## 6.1 RL 阶段的 loss

作用对象：

- 生成器

学习目标：

- 让 diffusion rollout 的最终轨迹更高质量

## 6.2 selector 阶段的 loss

作用对象：

- 轨迹打分器

学习目标：

- 从候选集中挑出最好的

换句话说：

- Stage I 学“怎么造”
- Stage II 学“怎么选”

---

## 7. 一个最容易混淆但必须记住的事实

在这个项目里：

- reward 不是训练时额外打印的监控项
- reward 本身就是算法闭环的一部分

所以整个系统不能只用“监督学习 pipeline”的视角理解，  
而要用“生成 + 环境评测 + 策略更新”的视角理解。

---

## 8. 你现在应该能回答的 5 个问题

读完这篇后，你最好能自己回答：

1. 为什么 `TrajectoryHead` 里既有 diffusion scheduler，又有 PDM pool
2. 为什么 RL 版训练要先无梯度跑一次再有梯度跑一次
3. 为什么 selector 训练前半段是 `with torch.no_grad()`
4. 为什么这个项目必须额外维护 `metric_cache_path`
5. 为什么最后部署时真正需要返回的是 `trajectory`，但快速评测默认返回的是 `reward_dict`

如果这 5 个问题都能答顺，这个项目的端到端总图就已经在你脑子里成型了。
