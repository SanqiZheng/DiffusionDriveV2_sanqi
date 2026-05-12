# DiffusionDriveV2 代码来源树状图（紧凑版）

图例：
- `[DD 直继]`：同路径来自 DiffusionDrive，基本未改
- `[DD 轻改]`：同路径来自 DiffusionDrive，小改（导入/配置/少量逻辑）
- `[DD 深改]`：同路径来自 DiffusionDrive，核心逻辑改写明显
- `[NEW 路径新增]`：在本仓新增目录/文件，但通常可追溯到其他上游文件拼接/迁移
- `{上游链: ...}`：跨仓库来源线索

```text
DiffusionDriveV2/
├── setup.py                                                                 [DD 直继]
│
└── navsim/
    ├── __init__.py                                                          [DD 直继]
    │
    ├── common/
    │   └── __init__.py/dataclasses.py/dataloader.py/enums.py               [DD 直继]
    │
    ├── visualization/
    │   └── __init__.py/bev.py/camera.py/config.py/lidar.py/plots.py        [DD 直继]
    │
    ├── evaluate/
    │   ├── __init__.py                                                      [DD 直继]
    │   └── pdm_score.py                                                     [DD 深改 +147/-0]
    │
    ├── agents/
    │   ├── __init__.py/abstract_agent.py/constant_velocity_agent.py         [DD 直继]
    │   ├── human_agent.py/ego_status_mlp_agent.py                           [DD 直继]
    │   │
    │   ├── transfuser/                                                      [DD 直继; NAVSIM TransFuser链]
    │   │   └── transfuser_agent.py/transfuser_backbone.py/
    │   │      transfuser_callback.py/transfuser_config.py/
    │   │      transfuser_features.py/transfuser_loss.py/transfuser_model.py
    │   │      {上游链: carla_garage(特征构建)}
    │   │
    │   ├── diffusiondrive/                                                  [DD 直继; DiffusionDrive 主链]
    │   │   ├── transfuser_agent.py/transfuser_backbone.py/
    │   │   │   transfuser_callback.py/transfuser_config.py/
    │   │   │   transfuser_features.py/transfuser_loss.py/transfuser_model_v2.py
    │   │   │   {上游链: carla_garage(特征构建)}
    │   │   └── modules/
    │   │       ├── conditional_unet1d.py                                   [DD 直继]
    │   │       ├── multimodal_loss.py/scheduler.py                          [DD 直继]
    │   │       └── blocks.py                                                [DD 直继]
    │   │           {上游链: DAB-DETR(gen_sineembed_for_position)}
    │   │
    │   └── diffusiondrivev2/                                                [NEW 路径新增; 当前主创新链]
    │       ├── diffusiondrivev2_model_sel.py                                [DD 深改(基于 diffusiondrive/transfuser_model_v2)]
    │       │   {上游链: diffusers + DAB-DETR + PDM scorer}
    │       ├── diffusiondrivev2_model_rl.py                                 [DD 深改(基于 model_sel 再改写)]
    │       │   {上游链: diffusers + RL policy gradient 回放链}
    │       ├── diffusiondrivev2_sel_agent.py                                [DD 深改(基于 diffusiondrive/transfuser_agent)]
    │       ├── diffusiondrivev2_rl_agent.py                                 [DD 深改(基于 diffusiondrivev2_sel_agent)]
    │       ├── diffusiondrivev2_sel_config.py/diffusiondrivev2_rl_config.py [DD 轻改(基于 diffusiondrive/transfuser_config)]
    │       ├── transfuser_backbone.py                                       [DD 轻改(仅导入路径/命名空间变更)]
    │       ├── transfuser_features.py/transfuser_loss.py/
    │       │   transfuser_callback.py/transfuser_config.py                  [DD 直继(迁移到 v2 路径)]
    │       │   {上游链: carla_garage(特征构建)}
    │       └── modules/
    │           ├── conditional_unet1d.py/multimodal_loss.py/scheduler.py    [DD 直继(迁移到 v2 路径)]
    │           │   {上游链: diffusion_policy(conditional_unet1d)}
    │           └── blocks.py                                                [DD 深改(+77/-0)]
    │               {上游链: DAB-DETR + scorer attention 新增}
    │
    └── planning/
        ├── __init__.py                                                      [DD 直继]
        │
        ├── metric_caching/
        │   └── __init__.py/caching.py/metric_cache.py/
        │      metric_cache_processor.py/metric_caching_utils.py             [DD 直继]
        │
        ├── scenario_builder/
        │   └── __init__.py/navsim_scenario.py/navsim_scenario_utils.py      [DD 直继]
        │
        ├── script/
        │   ├── __init__.py                                                   [DD 直继]
        │   ├── builders/
        │   │   └── __init__.py/observation_builder.py/planner_builder.py/
        │   │      simulation_builder.py/worker_pool_builder.py              [DD 直继]
        │   ├── config/
        │   │   ├── __init__.py                                               [DD 直继]
        │   │   ├── common/__init__.py                                        [DD 直继]
        │   │   ├── common/train_test_split/scene_filter/__init__.py          [DD 直继]
        │   │   ├── common/worker/__init__.py                                 [DD 直继]
        │   │   ├── metric_caching/__init__.py                                [DD 直继]
        │   │   ├── pdm_scoring/__init__.py                                   [DD 直继]
        │   │   └── training/__init__.py                                      [DD 直继]
        │   ├── run_create_submission_pickle.py/run_dataset_caching.py/
        │   │   run_merge_submission_pickles.py/run_metric_caching.py/
        │   │   run_pdm_score.py/run_pdm_score_from_submission.py/
        │   │   run_training.py/utils.py                                      [DD 直继]
        │   └── run_pdm_score_fast.py                                         [NEW 路径新增]
        │
        ├── simulation/
        │   ├── __init__.py                                                   [DD 直继]
        │   └── planner/
        │       ├── __init__.py                                               [DD 直继]
        │       └── pdm_planner/
        │           ├── __init__.py                                           [DD 直继]
        │           ├── abstract_pdm_closed_planner.py/
        │           │   abstract_pdm_planner.py/pdm_closed_planner.py         [DD 直继]
        │           ├── observation/__init__.py/pdm_object_manager.py/
        │           │   pdm_observation.py/pdm_occupancy_map.py               [DD 直继]
        │           ├── proposal/__init__.py/batch_idm_policy.py/
        │           │   pdm_generator.py/pdm_proposal.py                      [DD 直继]
        │           ├── scoring/__init__.py/pdm_comfort_metrics.py/
        │           │   pdm_scorer.py/pdm_scorer_utils.py                     [DD 直继]
        │           ├── simulation/__init__.py/batch_kinematic_bicycle.py/
        │           │   batch_lqr.py/batch_lqr_utils.py/pdm_simulator.py      [DD 直继]
        │           │   {上游链: nuplan-devkit(batch_lqr*)}
        │           └── utils/__init__.py/pdm_array_representation.py/
        │               pdm_emergency_brake.py/pdm_enums.py/pdm_geometry_utils.py/
        │               pdm_path.py/route_utils.py/
        │               graph_search/__init__.py/bfs_roadblock.py/dijkstra.py [DD 直继]
        │
        ├── training/
        │   ├── __init__.py/abstract_feature_target_builder.py                [DD 直继]
        │   ├── callbacks/time_logging_callback.py                             [DD 直继]
        │   ├── agent_lightning_module.py                                     [DD 轻改 +11/-3]
        │   └── dataset.py                                                    [DD 深改 +85/-4]
        │
        └── utils/
            └── multithreading/__init__.py/worker_ray_no_torch.py             [DD 直继]
```

最值得优先精读：
1. `navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel.py`
2. `navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py`
3. `navsim/agents/diffusiondrivev2/modules/blocks.py`
4. `navsim/evaluate/pdm_score.py`
