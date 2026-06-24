# MARVEL 多智能体强化学习探索系统 —— 完整技术文档

## 目录
1. [项目概述](#1-项目概述)
2. [项目文件结构](#2-项目文件结构)
3. [环境配置与安装](#3-环境配置与安装)
4. [全局配置参数详解](#4-全局配置参数详解)
5. [核心模块详解](#5-核心模块详解)
   - [5.1 仿真环境 (env.py / env_test.py)](#51-仿真环境)
   - [5.2 传感器模拟 (sensor.py)](#52-传感器模拟)
   - [5.3 运动模型 (motion_model.py)](#53-运动模型)
   - [5.4 图节点管理 (node_manager.py)](#54-图节点管理)
   - [5.5 真值图节点管理 (ground_truth_node_manager.py)](#55-真值图节点管理)
   - [5.6 四叉树空间索引 (quads.py)](#56-四叉树空间索引)
   - [5.7 智能体 (agent.py)](#57-智能体)
   - [5.8 神经网络模型 (model.py)](#58-神经网络模型)
   - [5.9 工具函数 (utils.py)](#59-工具函数)
6. [训练系统详解](#6-训练系统详解)
   - [6.1 训练主循环 (driver.py)](#61-训练主循环)
   - [6.2 分布式训练工作器 (runner.py)](#62-分布式训练工作器)
   - [6.3 多智能体训练工作器 (multi_agent_worker.py)](#63-多智能体训练工作器)
7. [测试与评估系统](#7-测试与评估系统)
8. [算法详解](#8-算法详解)
   - [8.1 SAC 算法实现](#81-sac-算法实现)
   - [8.2 四种训练模式](#82-四种训练模式)
   - [8.3 奖励函数设计](#83-奖励函数设计)
9. [网络架构详解](#9-网络架构详解)
   - [9.1 PolicyNet（策略网络）](#91-policynet策略网络)
   - [9.2 QNet（Q值网络）](#92-qnetq值网络)
10. [完整运行流程](#10-完整运行流程)
    - [10.1 训练流程](#101-训练流程)
    - [10.2 测试/评估流程](#102-测试评估流程)
11. [关键设计决策与创新点](#11-关键设计决策与创新点)

---

## 1. 项目概述

**MARVEL**（**M**ulti-**A**gent **R**einforcement Learning for Constrained Field-of-**V**iew Multi-Robot **E**xploration in **L**arge-Scale Environments）是一个面向大规模环境中受限视野多机器人协同探索的神经框架。

- **发表会议**：ICRA 2025
- **论文链接**：[arXiv:2502.20217](https://arxiv.org/abs/2502.20217)
- **许可证**：MIT
- **核心思想**：利用图注意力网络（Graph Attention Networks）结合新颖的前沿分布与朝向特征融合技术，通过多智能体强化学习（MARL）开发协作式分布式策略，用于具有受限视场角（FoV）的机器人探索任务。

### 1.1 系统能力一览

| 能力 | 描述 |
|------|------|
| 多机器人协同探索 | 支持 4+ 个机器人同时探索未知室内环境 |
| 受限视野模拟 | 可配置视场角（默认 120°）的视线传感器 |
| 图基状态表示 | 使用规则网格节点（4m 间距）+ 四叉树空间索引 |
| Transformer 策略 | 6 层编码器 + 1 层解码器 + 指针网络进行目标节点选择 |
| 分布式训练 | 基于 Ray 的并行经验收集（18 个 meta-agent） |
| 多算法支持 | SAC / MAAC / Ground Truth / 混合模式 |
| 硬件验证 | 提供真实无人机实验的硬件验证 |

---

## 2. 项目文件结构

```
MARVEL-main/
├── driver.py                        # 主训练脚本（SAC 分布式训练）
├── parameter.py                     # 训练全局配置文件
├── test_driver.py                   # 分布式测试/评估脚本
├── test_parameter.py                # 测试全局配置文件
├── marvel.yml                       # Conda 环境配置
├── README.md                        # 项目说明
├── LICENSE                          # MIT 许可证
│
├── utils/                           # 核心源代码目录
│   ├── model.py                     # 神经网络架构（PolicyNet + QNet + Transformer）
│   ├── agent.py                     # 单智能体逻辑（478 行）
│   ├── env.py                       # 训练环境（188 行）
│   ├── env_test.py                  # 测试环境（204 行）
│   ├── multi_agent_worker.py        # 多智能体训练工作器（351 行）
│   ├── test_worker.py               # 测试工作器（351 行）
│   ├── runner.py                    # Ray 分布式 Runner（76 行）
│   ├── node_manager.py             # 图节点管理（含 Dijkstra/A*）（431 行）
│   ├── ground_truth_node_manager.py # 真值图节点管理（433 行）
│   ├── quads.py                     # 四叉树空间索引（781 行）
│   ├── sensor.py                    # 传感器模拟（123 行）
│   ├── motion_model.py              # 运动学模型（73 行）
│   ├── utils.py                     # 工具函数集（237 行）
│   └── media/                       # 演示素材
│       ├── MARVEL.gif               # 仿真探索动画
│       └── Hardware_validation.gif  # 硬件验证动画
│
├── maps_medium/                     # 训练地图集（4000 张 PNG）
├── maps_test/                       # 测试地图集（100 张 PNG）
├── load_model/MARVEL/               # 预训练模型
│   └── checkpoint.pth               # 61 MB PyTorch 检查点
│
├── model/                           # 训练输出（自动生成）
├── train/                           # TensorBoard 日志（自动生成）
└── gifs/                            # 可视化 GIF（自动生成）
```

---

## 3. 环境配置与安装

### 3.1 系统依赖

- **Python**：3.11.8
- **PyTorch**：2.5.1（CUDA 12.4，cuDNN 9.1）
- **Ray**：2.44.0（分布式训练框架）
- **关键包**：gymnasium 0.28.1, wandb 0.17.5, tensorboard, matplotlib, scikit-image, shapely, imageio, numpy, scipy, pandas

### 3.2 安装步骤

```bash
# 创建 conda 环境
conda env create -f marvel.yml
conda activate marvel
```

### 3.3 硬件建议

- 训练：推荐使用 GPU（`USE_GPU_GLOBAL = True`）
- 数据收集：可使用 CPU（`USE_GPU = False`）
- 地图数据：约 4000 张训练地图 + 100 张测试地图

---

## 4. 全局配置参数详解

### 4.1 训练配置 (`parameter.py`)

#### 路径与存储配置
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `FOLDER_NAME` | `'marvel'` | 实验文件夹名称 |
| `model_path` | `'model/marvel'` | 模型保存路径 |
| `load_path` | `'load_model/marvel'` | 预训练模型加载路径 |
| `train_path` | `'train/marvel'` | TensorBoard 日志路径 |
| `gifs_path` | `'gifs/marvel'` | 可视化输出路径 |
| `SUMMARY_WINDOW` | 32 | TensorBoard 记录间隔（episode 数） |
| `LOAD_MODEL` | `True` | 是否从 checkpoint 恢复训练 |
| `SAVE_IMG_GAP` | 1000 | 每隔多少 episode 保存可视化 |
| `NUM_EPISODE_BUFFER` | 40 | 每个 episode buffer 的槽位数（对应 buffer 中的不同字段） |

#### 仿真参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `N_AGENTS` | 4 | 每 episode 的智能体数量 |
| `USE_CONTINUOUS_SIM` | `True` | 是否使用连续运动模拟 |
| `NUM_SIM_STEPS` | 6 | 每次决策之间的模拟步数 |
| `VELOCITY` | 1 | 机器人线速度（m/s） |
| `YAW_RATE` | 35 | 最大偏航角速率（度/秒） |

#### 传感器与朝向参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `FOV` | 120 | 视场角（度） |
| `V_FOV` | 60 | 垂直视场角（度） |
| `MOUNTING_ANGLE` | 15 | 传感器下倾角（度） |
| `NUM_ANGLES_BIN` | 36 | 朝向离散化桶数（每 10° 一桶） |
| `NUM_HEADING_CANDIDATES` | 3 | 每个邻居节点的候选朝向数 |
| `DRONE_HEIGHT` | 2 | 无人机飞行高度（m） |

#### 地图与规划分辨率
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CELL_SIZE` | 0.4 | 栅格单元尺寸（m） |
| `NODE_RESOLUTION` | 4.0 | 图节点间距（m） |
| `FRONTIER_CELL_SIZE` | 0.8 | 前沿下采样分辨率 |
| `FREE` | 255 | 空闲单元像素值 |
| `OCCUPIED` | 1 | 占用单元像素值 |
| `UNKNOWN` | 127 | 未知单元像素值 |

#### 传感器与效用范围
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SENSOR_RANGE` | 10 | 传感器感知范围（m） |
| `UTILITY_RANGE` | 9 | 效用计算范围 = 0.9 × SENSOR_RANGE |
| `MIN_UTILITY` | 1 | 最小效用阈值（低于此值认为节点无效） |
| `UPDATING_MAP_SIZE` | 56 | 局部地图更新范围 = 4×SENSOR_RANGE + 4×NODE_RESOLUTION |

#### 训练超参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_EPISODE_STEP` | 128 | 每 episode 最大步数 |
| `REPLAY_SIZE` | 10000 | 经验回放缓冲区大小 |
| `MINIMUM_BUFFER_SIZE` | 2000 | 开始训练的最小经验量 |
| `BATCH_SIZE` | 256 | 每次训练的批量大小 |
| `LR` | 1e-5 | 学习率 |
| `GAMMA` | 1 | 折扣因子（无折扣） |
| `NUM_META_AGENT` | 18 | 并行 Ray worker 数量 |

#### 网络参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NODE_INPUT_DIM` | 6 | 节点输入维度（x, y, utility, guidepost, occupancy, highest_angle） |
| `EMBEDDING_DIM` | 128 | Transformer 嵌入维度 |
| `NUM_NODE_NEIGHBORS` | 5 | 邻居矩阵大小（5×5=25 候选邻居） |
| `K_SIZE` | 25 | 相邻节点数量 = NUM_NODE_NEIGHBORS² |
| `NODE_PADDING_SIZE` | 360 | 每 batch 填充到的最大节点数 |

#### GPU 与日志配置
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `USE_GPU` | `False` | Worker 端是否使用 GPU |
| `USE_GPU_GLOBAL` | `True` | 训练端是否使用 GPU |
| `NUM_GPU` | 1 | GPU 数量 |
| `USE_WANDB` | `False` | 是否使用 Weights & Biases 记录 |
| `TRAIN_ALGO` | 3 | 训练算法：0=SAC, 1=MAAC, 2=Ground Truth, 3=MAAC+GT |

### 4.2 测试配置 (`test_parameter.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TEST_SET` | `'maps_test'` | 测试地图目录 |
| `NUM_TEST` | 100 | 测试 episode 数量 |
| `NUM_META_AGENT` | 10 | 并行测试 worker 数量 |
| `GREEDY` | `True` | 是否使用贪婪策略（确定性） |
| `SAVE_GIFS` | `True` | 是否保存可视化 GIF |
| `INITIAL_EXPLORED_RATE` | 0.90 | 评估 "达到 90% 探索的距离" |

---

## 5. 核心模块详解

### 5.1 仿真环境

#### 训练环境 (`utils/env.py`)

`Env` 类负责管理多机器人探索的仿真环境。

**核心属性**：
- `ground_truth`：真实占据栅格地图（二值数组）
- `robot_belief`：所有机器人共享的信念地图（已知/未知表示）
- `robot_locations`：所有机器人的当前位置坐标
- `angles`：所有机器人的初始朝向（随机分布在 0-360°）
- `global_frontiers`：当前信念地图上的所有前沿点集合

**地图加载流程 (`import_ground_truth`)**：
1. 从 `maps_medium/` 读取 PNG 地图
2. 执行 2 倍下采样（`block_reduce`，取 min）
3. 从像素值 208 处提取机器人初始位置
4. 二值化：像素值 > 150 或 50-80 → 空闲（255）；其他 → 占用（1）
5. 初始信念全部设为未知（127）

**核心方法**：

| 方法 | 功能 |
|------|------|
| `update_robot_belief(robot_cell, heading)` | 根据机器人位置和朝向，调用传感器模型更新信念地图 |
| `calculate_team_reward()` | 计算团队奖励：基于减少的前沿点数量 |
| `check_done()` | 判断探索是否完成：未探索单元 ≤ 250 |
| `evaluate_exploration_rate()` | 计算探索率 = 已探索自由单元 / 总自由单元 |
| `create_sensing_mask(location, heading)` | 创建扇形感知掩码（基于 shapely 多边形），用于计算感知重叠 |
| `calculate_overlap_reward(robot_locations, headings)` | 计算多机器人感知区域的重叠奖励 |

#### 测试环境 (`utils/env_test.py`)

与训练环境的区别：
- 从 `maps_test/` 加载地图
- 支持可变数量的智能体（不限于 `N_AGENTS`）
- 初始角度固定为 270°（面向上方），而非随机
- 新增 `calculate_overlap_reward` 用于评估重叠率

### 5.2 传感器模拟

#### 文件：`utils/sensor.py`

模拟受限视野的视线传感器。

**核心函数 `sensor_work_heading`**：
1. 计算 FoV 的起始和结束角度
2. 以 0.5° 为增量遍历 FoV 内的所有角度
3. 对每个角度，使用 Bresenham 光线投射算法
4. 将光线上每个单元的 ground truth 值写入 belief map
5. 遭遇超过 2 个连续占用单元时停止该光线

**`collision_check` 函数（Bresenham 算法）**：
- 沿光线逐像素检查 ground truth
- 如遇到占用单元（像素值=1），碰撞计数 +1
- 连续 2 个占用单元后终止，表示被遮挡
- 光线上测量值直接写入 belief 地图

**`fov_sweep` 函数**：
处理跨越 0° 边界的情况（如 start_angle=330°, end_angle=90°），分别生成 [330°, 360°) 和 [0°, 90°] 的角度序列。

### 5.3 运动模型

#### 文件：`utils/motion_model.py`

**`compute_allowable_heading` 函数**：
- **目的**：考虑偏航角速率约束，计算实际可达的朝向
- **逻辑**：
  1. 计算从当前位置到目标位置的轨迹朝向 `theta_target`
  2. 计算期望朝向变化 `delta_theta_desired`（归一化到 [-180°, 180°]）
  3. 计算到达目标位置所需时间 `t_travel = distance / velocity`
  4. 计算完成期望转向所需时间 `t_desired_yaw = |delta_theta_desired| / omega_max`
  5. 如果 `t_desired_yaw ≤ t_travel`：可以达到期望朝向 → 返回 `theta_desired`
  6. 否则：只能达到部分转向 → 返回 `theta_current + copysign(delta_theta_achievable, delta_theta_desired)`

此机制确保机器人在短距离移动时不会做出不切实际的快速转向。

### 5.4 图节点管理

#### 文件：`utils/node_manager.py`

##### `NodeManager` 类

负责管理和更新探索图中的所有空间节点。

**核心属性**：
- `nodes_dict`：四叉树（Quadtree）数据结构，存储所有节点
- `utility_range`：效用计算范围（默认 9m）
- `fov` / `sensor_range`：传感器参数

**关键方法**：

| 方法 | 功能 |
|------|------|
| `update_graph(robot_location, frontiers, updating_map_info, map_info)` | 主图更新入口：在机器人周围生成/更新节点，更新邻居关系 |
| `get_all_node_graph(robot_location, robot_locations)` | 获取完整图状态：所有节点坐标、效用、前沿分布、朝向访问矩阵、邻接矩阵、当前节点索引、邻居索引、路标、占用信息、A* 路径 |
| `Dijkstra(start)` | 经典 Dijkstra 最短路径算法（遍历所有节点） |
| `a_star(start, destination)` | A* 路径规划算法，使用欧几里得距离作为启发式函数 |
| `add_node_to_dict(coords, frontiers, updating_map_info)` | 创建新节点并插入四叉树 |
| `remove_node_from_dict(node)` | 删除节点并更新邻居关系 |

##### `Node` 类

表示图中的一个空间节点。

**核心属性**：
| 属性 | 类型 | 说明 |
|------|------|------|
| `coords` | ndarray | 节点世界坐标 |
| `utility` | int | 节点效用值（可观察到的前沿点数量） |
| `observable_frontiers` | set | 节点可视范围内的前沿点集合 |
| `frontiers_distribution` | ndarray(36,) | 前沿点在 36 个角度桶中的分布直方图 |
| `heading_visited` | ndarray(36,) | 每个角度桶已被访问过的标记 |
| `highest_utility_angle` | float | 在 FoV 窗口内前沿密度最高的角度 |
| `neighbor_matrix` | ndarray(5×5) | 邻居矩阵（以节点为中心的 5×5 网格） |
| `neighbor_list` | list | 已确认邻居节点坐标列表（含自身） |
| `visited` | int | 节点是否被访问过 |

**效用计算 (`initialize_observable_frontiers`)**：
1. 筛选在 `utility_range` 内的所有前沿点
2. 对每个前沿点做碰撞检测（Bresenham），确认视线无遮挡
3. 计算效用 = 可观察到的前沿点数量
4. 若效用 ≤ `MIN_UTILITY`（1），则效用归零，节点视为无效
5. 计算前沿分布直方图（36 个桶）
6. 使用滑动窗口（窗宽 = FoV）找到累积前沿密度最高的角度方向

**邻居更新 (`update_neighbor_nodes`)**：
- 检查 5×5 网格中的相邻位置是否存在节点
- 进行碰撞检测确认两个节点之间可达
- 双向更新邻居关系

### 5.5 真值图节点管理

#### 文件：`utils/ground_truth_node_manager.py`

`GroundTruthNodeManager` 维护一个基于真实地图（ground truth）的"全知"图，用于提供集中式评论家（critic）的全局信息。

**与 NodeManager 的关键区别**：
1. **初始化时构建完整图**：`initialize_graph()` 在真实地图的自由空间网格上预先创建所有节点
2. **跟踪探索状态**：`Node.explored` 标记节点是否已被探索（初始化为 0）
3. **动态同步**：`update_graph()` 从 `NodeManager` 同步已探索节点的前沿分布和效用信息
4. **提供真值观测**：`get_ground_truth_observation()` 生成包含 `explored` 标志的观测（7 维 vs 6 维），专供 QNet 使用
5. **节点数更大**：包含所有未探索节点，需要填充到 `NODE_PADDING_SIZE=360`

### 5.6 四叉树空间索引

#### 文件：`utils/quads.py`

完整的四叉树（Quadtree）实现（781 行），来自 Daniel Lindsley 的 `quads` 库。

**核心类**：

| 类 | 功能 |
|------|------|
| `Point` | 二维点（x, y） |
| `BoundingBox` | 轴对齐边界框 |
| `QuadNode` | 四叉树节点（包含点列表、边界、子节点指针） |
| `QuadTree` | 四叉树主类 |

**核心操作**：
- `insert(point, data)`：插入点及其关联数据
- `find(point)`：精确查找点
- `remove(point)`：删除点
- `nearest_neighbors(point, count)`：半径递增的最近邻搜索
- `within_bb(bounding_box)`：边界框内范围查询
- `__iter__()`：深度优先遍历所有节点
- `__len__()`：树中点的总数

四叉树用于 NodeManager 和 GroundTruthNodeManager 中的所有空间节点查找和邻居搜索。

### 5.7 智能体

#### 文件：`utils/agent.py`

`Agent` 类（478 行）是单个机器人的完整逻辑实现。

**核心属性**：

| 属性 | 说明 |
|------|------|
| `location` | 当前世界坐标 |
| `heading` | 当前朝向（度，0-360） |
| `travel_dist` | 累计行进距离 |
| `map_info` | 全局信念地图信息 |
| `updating_map_info` | 局部更新窗口内的地图信息 |
| `frontier` | 局部地图中的前沿点集合 |
| `node_manager` | 图节点管理器引用 |
| `ground_truth_node_manager` | 真值图节点管理器引用 |
| `policy_net` | 策略网络模型 |

**图状态属性（每次调用 `update_planning_state` 更新）**：
- `node_coords`：所有节点坐标
- `utility`：各节点效用值
- `guidepost`：A* 路标（最优路径上的节点标记为 1）
- `occupancy`：节点占用情况（-1=自身, 0=空闲, 1=其他机器人）
- `adjacent_matrix`：节点邻接矩阵
- `current_index`：当前所在节点的索引
- `neighbor_indices`：邻居节点索引列表
- `highest_utility_angles`：最高效用朝向角
- `frontier_distribution`：前沿角度分布
- `heading_visited`：已访问朝向记录

**关键方法流程**：

##### `update_graph(map_info, location)`
```
1. 更新全局地图引用
2. 更新位置（含行进距离计算）
3. 提取局部更新窗口地图
4. 检测前沿点
5. 调用 NodeManager.update_graph() 更新图节点
6. 标记当前节点的 visited 状态
```

##### `update_planning_state(robot_locations)`
```
调用 NodeManager.get_all_node_graph()
→ 返回节点坐标、效用、路标、占用、邻接矩阵、当前索引、邻居索引、
   最优朝向、前沿分布、朝向访问记录、A* 路径
```

##### `get_observation(pad=True)`
构建策略网络的标准输入（9 个张量）：
1. `node_inputs`：[N, 6] — 每个节点的 (相对坐标 x, y, 效用, 路标, 占用, 最高效用角)
2. `node_padding_mask`：[1, 1, N_padded] — 填充掩码
3. `edge_mask`：[1, N_padded, N_padded] — 邻接矩阵掩码
4. `current_index`：[1, 1, 1] — 当前节点索引
5. `current_edge`：[1, K, 1] — 当前节点的邻居边（含自身）
6. `edge_padding_mask`：[1, 1, K_padded] — 边的填充掩码
7. `frontier_distribution`：[1, N_padded, 36] — 每个节点的前沿角度分布
8. `heading_visited`：[1, N_padded, 36] — 每个节点的已访问朝向
9. `neighbor_best_headings`：[1, K_padded, 3, 36] — 每个邻居节点的最优候选朝向

所有特征都经过归一化处理（坐标除以更新范围，效用除以理论最大值等）。

##### `select_next_waypoint(observation, greedy=False)`
1. 通过策略网络计算 log-probabilities
2. 贪婪模式：`argmax(logp)` 选择最高概率动作
3. 随机模式：`multinomial(logp.exp())` 按概率采样
4. 解析动作索引 → 目标节点索引 + 朝向索引
5. 返回目标位置、节点索引、动作索引、朝向索引

##### `compute_best_heading(node_coords, frontier_distribution, neighbor_nodes)`
为每个邻居节点计算 NUM_HEADING_CANDIDATES（3）个最优候选朝向：
1. 如果节点有正效用：
   - 从 frontier_distribution 中提取节点的前沿角度分布
   - 用滑动窗口（窗宽=FoV）计算每个角度的累积前沿密度
   - 选择 top-3 累积密度最高的角度
   - 将 FoV 范围内的所有角度桶标记为候选（one-hot 编码）
2. 如果节点效用为 0：
   - 若存在 A* 路径：朝向沿路径的下一个节点
   - 否则：朝向邻居节点的方向
   - 若都没有：朝向默认值

##### episode_buffer 存储
Agent 维护 40 个槽位的 episode buffer，存储以下字段：

| 槽位 | 内容 |
|------|------|
| 0-9 | 标准观测字段 |
| 8 | 动作索引 |
| 9 | 奖励 |
| 10 | 完成标志 |
| 11-18 | 下一观测 |
| 19-26 | 真值观测（Ground Truth） |
| 27-34 | 下一真值观测 |
| 35-37 | 所有智能体的节点索引（MAAC） |
| 38-39 | 邻居最佳朝向（当前和下一） |

### 5.8 神经网络模型

#### 文件：`utils/model.py`（详见[第 9 节](#9-网络架构详解)）

模型文件实现了完整的 Transformer 架构，包括：
- `SingleHeadAttention`：带 tanh 裁剪的指针网络注意力
- `MultiHeadAttention`：标准 8 头自注意力
- `EncoderLayer` / `DecoderLayer`：带残差连接和 LayerNorm
- `Encoder`（6 层）/ `Decoder`（1 层）
- `PolicyNet`：策略网络
- `QNet`：Q 值网络

### 5.9 工具函数

#### 文件：`utils/utils.py`

| 函数 | 功能 |
|------|------|
| `get_cell_position_from_coords(coords, map_info)` | 世界坐标 → 栅格单元索引 |
| `get_coords_from_cell_position(cell_position, map_info)` | 栅格单元索引 → 世界坐标 |
| `get_free_and_connected_map(location, map_info)` | 连通分量标记：提取机器人所在连通区域 |
| `get_updating_node_coords(location, updating_map_info)` | 在局部地图范围内生成规则网格节点坐标，过滤连通性 |
| `get_frontier_in_map(map_info)` | 8 连通前沿检测：自由单元且 2-7 个邻居为未知 |
| `frontier_down_sample(data, voxel_size)` | 体素下采样前沿点 |
| `is_frontier(location, map_info)` | 判断单个位置是否为前沿点 |
| `check_collision(start, end, map_info)` | Bresenham 碰撞检测（占用和未知都算碰撞） |
| `make_gif(path, n, frame_files, rate)` | 将帧序列合成为 GIF 并清理中间文件 |
| `MapInfo` 类 | 封装地图数组 + 原点 + 单元尺寸 |

---

## 6. 训练系统详解

### 6.1 训练主循环

#### 文件：`driver.py`（385 行）

##### 初始化阶段
```
1. ray.init() — 初始化 Ray 分布式框架
2. 创建 TensorBoard SummaryWriter
3. 创建模型保存目录
4. 实例化神经网络：
   - PolicyNet(NODE_INPUT_DIM=6, EMBEDDING_DIM=128, NUM_ANGLES_BIN=36)
   - QNet × 2（双 Q 网络减少过估计偏差）
   - log_alpha（可学习温度参数，初始值 -2）
   - Target QNet × 2（目标网络用于稳定训练）
5. 创建优化器：
   - Adam (lr=1e-5) × 3（策略 + Q1 + Q2）
   - Adam (lr=1e-4) × 1（温度参数）
6. 计算目标熵：entropy_target = 0.05 * (-ln(1/K_SIZE)) = 0.05 * (-ln(1/25))
7. 如果 LOAD_MODEL：从 checkpoint 恢复所有模型和优化器状态
8. 用 DataParallel 包裹网络以支持多 GPU
9. 启动 NUM_META_AGENT(18) 个 Ray RLRunner worker
10. 为每个 worker 提交第一个 episode 任务
```

##### 主训练循环
```
while True:
    1. ray.wait(job_list) — 等待任意 worker 完成
    2. ray.get(done_id) — 获取完成的 episode 数据
    3. 将经验（观测、动作、奖励等）存入 experience_buffer
    4. 记录性能指标（travel_dist, success_rate, explored_rate）
    5. 为完成的 worker 提交新的 episode 任务
    6. 如果 buffer ≥ MINIMUM_BUFFER_SIZE(2000)：
       a. 限制 buffer 大小 ≤ REPLAY_SIZE(10000)
       b. 随机采样 BATCH_SIZE(256) 条经验
       c. 每个训练步骤执行 4 次更新（j in range(4)）
       d. 每次更新执行完整的 SAC 算法步骤（见 8.1 节）
    7. 每 SUMMARY_WINDOW(32) 个 episode 写 TensorBoard 日志
    8. 每 64 次 Q 网络更新后软更新目标网络（完整复制）
    9. 每 32 个 episode 保存一次 checkpoint
```

##### 四个训练步骤的详细流程

每个步骤中，根据 `TRAIN_ALGO` 参数构建不同的 state/next_state：

**TRAIN_ALGO = 0（纯 SAC）**：
- state = next_state = 标准 9 元组观测
- 策略仅使用局部观测做决策

**TRAIN_ALGO = 1（MAAC）**：
- state 扩展为包含 `all_agent_indices` 和 `all_agent_next_indices`
- QNet 可以通过 agent_decoder 处理其他智能体选择的动作

**TRAIN_ALGO = 2（Ground Truth）**：
- state 使用 GroundTruth 观测（来自 `ground_truth_node_manager`）
- 节点维度为 NODE_INPUT_DIM + 1（多一个 explored 标志）
- 提供全局信息给 Q 网络

**TRAIN_ALGO = 3（MAAC + GT）**：
- state 同时包含 GT 观测和其他智能体动作信息
- 最大程度利用全局信息进行集中式训练

### 6.2 分布式训练工作器

#### 文件：`utils/runner.py`

##### `Runner` 基类
```
class Runner:
    - __init__(meta_agent_id)：创建本地 PolicyNet
    - get_weights()：获取当前网络权重
    - set_policy_net_weights(weights)：加载全局权重
    - do_job(episode_number)：
      1. 如果 episode_number % SAVE_IMG_GAP == 0，启用可视化
      2. 创建 MultiAgentWorker 并运行完整 episode
      3. 返回 episode_buffer 和 perf_metrics
    - job(weights_set, episode_number)：Ray 调用的入口
```

##### `RLRunner`（Ray Remote 类）
```python
@ray.remote(num_cpus=1, num_gpus=gpu)
class RLRunner(Runner):
    # 继承 Runner，通过 Ray 远程执行
    # GPU 分配：NUM_GPU / NUM_META_AGENT
```

每个 RLRunner：
- 在自己的进程/GPU 上运行
- 接收全局策略权重 → 用本地网络执行 episode → 返回经验数据
- 不执行训练（训练集中由 driver 完成）

### 6.3 多智能体训练工作器

#### 文件：`utils/multi_agent_worker.py`

`MultiAgentWorker` 是训练中每个 episode 的核心执行器。

##### 初始化
```
1. 创建 Env（环境仿真，含地图加载）
2. 创建 NodeManager（局部图管理）
3. 创建 GroundTruthNodeManager（真值图管理）
4. 创建 N_AGENTS 个 Agent 实例，共享同一个 PolicyNet
```

##### `run_episode()` 完整流程

**Phase 1：初始化图状态**
```
for robot in robot_list:
    robot.update_graph(belief_info, robot_locations[robot.id])
for robot in robot_list:
    robot.update_planning_state(robot_locations)
```

**Phase 2：主循环（最多 MAX_EPISODE_STEP=128 步）**

每一步：

1. **决策收集**：每个机器人：
   - `get_observation()`：获取局部观测
   - `get_ground_truth_observation()`：获取真值观测
   - `save_observation()` / `save_ground_truth_observation()`：存储
   - `select_next_waypoint(observation)`：通过策略网络选择下一个目标节点 + 朝向
   - `save_action(action_index)`：存储动作

2. **冲突解决**：
   - 按距离排序（优先到达的优先选择）
   - 如果多个机器人选择同一节点，通过最近邻搜索分配替代节点

3. **运动模拟**（`NUM_SIM_STEPS=6` 个子步）：
   - 对每个机器人：
     - 计算从当前位置到目标位置的中间路径点
     - 使用 `compute_allowable_heading` 计算受偏航速率约束的实际朝向
     - 使用 `smooth_heading_change` 在子步之间平滑过渡朝向
   - 在每个子步：
     - 使用中间路径的位置和朝向来更新传感器的信念地图
     - 可选：生成可视化帧

4. **奖励计算**：
   - **效用奖励（utility_reward）**：目标节点在当前 FoV 内可观察到的前沿数量（归一化）
   - **朝向奖励（angle_reward）**：机器人朝向与节点最优朝向的余弦相似度
   - **轨迹奖励（trajectory_reward）**：机器人朝向与运动方向的余弦相似度
   - **团队奖励（team_reward）**：减少的全局前沿数量（归一化）− 0.5 基线偏移
   - **完成奖励**：episode 成功完成时 +10

   每个机器人获得 `utility_reward + trajectory_reward + team_reward`。

5. **状态更新**：
   - `env.final_sim_step()`：最终位置的传感器更新
   - 每个机器人 `update_graph()` 和 `update_planning_state()`
   - 检查完成条件（所有节点效用为 0 或探索率 > 99%）

**Phase 3：Episode 结束**
- 保存 `next_observations`（用于 TD 目标计算）
- 合并所有机器人的 episode_buffer → worker 的 episode_buffer
- 可选：保存可视化 GIF

---

## 7. 测试与评估系统

### 文件：`test_driver.py` + `utils/test_worker.py`

#### 测试架构

```
test_driver.py
  ├── ray.init()
  ├── 加载预训练 PolicyNet
  ├── 创建 10 个 Ray TestRunner
  ├── 遍历配置组合：
  │     n_agent ∈ [6]
  │     fov ∈ [120]
  │     sensor_range ∈ [10]
  └── 收集和打印测试指标
```

#### 测试配置组合
默认测试参数：6 个智能体，120° FoV，10m 传感器范围。可通过修改 `all_fov`、`all_n_agent`、`all_sensor_range` 列表来测试不同配置。

#### 评估指标

| 指标 | 说明 |
|------|------|
| Average/Max/Min/Std Travel Distance | 所有测试 episode 中最长机器人行进距离的统计值 |
| Average Explored Rate | 平均最终探索率 |
| Average Success Rate | 探索成功的 episode 比例 |
| Average Distance to 90% Explored | 达到 90% 探索率时的平均行进距离 |
| Average Overlap Ratio | 多机器人感知区域的平均重叠比例 |

#### `TestWorker` 与 `MultiAgentWorker` 的区别
- 不使用 GroundTruthNodeManager（不需要训练信息）
- 执行确定性的贪婪动作选择（`greedy=True`）
- 支持可变数量的智能体
- 记录更详细的历史数据：`length_history`、`explored_rate_history`、`overlap_ratio_history`
- 使用 `compute_overlap_rate` 跟踪重叠率
- 完成条件更宽松（探索率 > 99%）
- 支持 `setpoints` 记录（用于可能的硬件部署）

---

## 8. 算法详解

### 8.1 SAC 算法实现

MARVEL 使用 **Soft Actor-Critic（SAC）** 作为基础 RL 算法。SAC 是最大熵强化学习算法，平衡探索与利用。

#### 核心组件

1. **策略网络（Actor）π(a|s)**：输出动作的对数概率 log π(a|s)
2. **两个 Q 网络（双 Critic）Q1(s,a), Q2(s,a)**：估计状态-动作值，取最小值减少过估计
3. **两个目标 Q 网络**：Q1_target, Q2_target（硬更新，每 64 步同步一次）
4. **可学习温度参数 α**：自动调节熵正则化强度

#### 损失函数

**策略损失（Policy Loss）**：
```
L_π = E[∑_a π(a|s) * (α * log π(a|s) - min(Q1(s,a), Q2(s,a)))]
```
即：最大化 Q 值的同时保持策略的熵。

**Q 网络损失（Q Loss）**：
```
y = r + γ * (1 - done) * ∑_{a'} π(a'|s') * (min(Q1_target(s',a'), Q2_target(s',a')) - α * log π(a'|s'))
L_Q = MSE(Q(s,a), y)
```
SAC 特有的 soft TD 目标，其中值函数包含熵奖励项。

**温度损失（Temperature Loss）**：
```
L_α = -α * (H(π) + H_target)
```
其中目标熵 `H_target = 0.05 * (-ln(1/25))`，鼓励策略维持最小熵水平。初始 `log α = -2`。

#### 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 更新频率 | 每个新 episode，执行 4 次参数更新 | 保证样本效率 |
| 目标网络更新 | 每 64 次 Q 更新 | 硬更新（完全复制） |
| 折扣因子 γ | 1.0 | 无限视界任务，不使用折扣 |
| 学习率 | 1e-5 | 策略和 Q 网络 |
| 温度学习率 | 1e-4 | 温度参数 |
| 梯度裁剪 | max_norm=100 (策略), max_norm=20000 (Q) | 防止梯度爆炸 |

### 8.2 四种训练模式

通过 `TRAIN_ALGO` 参数（0-3）切换：

| TRAIN_ALGO | 模式 | 说明 |
|------|------|------|
| 0 | **SAC** | 纯分布式 SAC。每个机器人独立使用局部观测做决策，Q 网络也仅使用局部信息。完全的 CTDE（集中训练、分散执行）实现。 |
| 1 | **MAAC** | Multi-Agent Actor-Critic。Q 网络额外接收所有智能体选择的动作（节点索引），通过 `agent_decoder` 对所有智能体进行注意力计算，实现多智能体信用分配。 |
| 2 | **Ground Truth** | 使用真值图信息训练。策略使用局部观测，但 Q 网络接收全局真实状态（包括所有未探索节点的信息），提供完美的集中式评估。 |
| 3 | **MAAC + Ground Truth** | 结合 MAAC 和 GT。Q 网络同时接收真值状态和其他智能体的动作。最完整的集中式训练信息。**默认模式**。 |

### 8.3 奖励函数设计

每个时间步，机器人获得三个分量的奖励：

#### 1. 效用奖励（Utility Reward）
```
utility_reward = len(current_observable_frontiers) / normalization_factor
normalization_factor = (2 * sensor_range * π / frontier_cell_size) / (360 / fov)
```
- 当前节点在机器人 FoV 内可观察到的前沿点数量
- 归一化到 [0, 1] 范围
- 鼓励机器人选择信息量大的目标节点

#### 2. 轨迹奖励（Trajectory Reward）
```
trajectory_reward = cos(heading - trajectory_angle)
```
- 机器人朝向与实际运动方向之间的余弦相似度
- 范围 [-1, 1]
- 鼓励机器人朝目标方向直线前进

#### 3. 团队奖励（Team Reward）（所有机器人共享）
```
team_reward = (delta_num_frontiers / normalization_factor) - 0.5
delta_num_frontiers = len(global_frontiers_before) - len(global_frontiers_after)
```
- 团队共同减少的全局前沿数量
- -0.5 基线惩罚（每一步都有微小惩罚，推动机器人高效探索）
- 完成奖励：episode 成功完成时 +10

**总奖励** = `utility_reward + trajectory_reward + team_reward`

---

## 9. 网络架构详解

### 9.1 PolicyNet（策略网络）

```
输入: (node_inputs, node_padding_mask, edge_mask, current_index,
       current_edge, edge_padding_mask, frontier_distribution,
       heading_visited, neighbor_best_headings)
输出: log_probabilities [batch, K_SIZE * NUM_HEADING_CANDIDATES]
```

#### 前向传播流程：

```
┌─────────────────────────────────────────────────────┐
│ Step 1: encode_graph (图编码)                       │
│                                                     │
│ node_inputs [B, N, 6]                               │
│   │                                                 │
│   ▼                                                 │
│ Linear(6 → 128)           frontier_distribution     │
│   │                       [B, N, 36]                │
│   ▼                             │                   │
│ Encoder (6层, 4头)              │                   │
│   │                              │                   │
│   ▼                             ▼                   │
│ enhanced_node_feature    Conv1d(36→128, k=3,p=1)   │
│   [B, N, 128]              │                        │
│   │                        ▼                        │
│   │                  frontiers_feature [B, N, 128]  │
│   │                        │                        │
│   └────────┬───────────────┘                        │
│            ▼                                        │
│   Concat → Linear(256 → 128)                       │
│            │                                        │
│            ▼                                        │
│   enhanced_node_feature [B, N, 128]                │
└─────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│ Step 2: decode_state (解码当前状态)                 │
│                                                     │
│   gather(current_index) → current_node_feature      │
│   [B, 1, 128]                                       │
│            │                                        │
│            ▼                                        │
│   Decoder (1层, 4头) cross-attention                │
│   with enhanced_node_feature                        │
│            │                                        │
│            ▼                                        │
│   enhanced_current_node_feature [B, 1, 128]         │
└─────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│ Step 3: output_policy (输出策略)                    │
│                                                     │
│   current_state = Linear(256→128)(                  │
│       concat(enhanced_current, current_node)        │
│   )                                                 │
│            │                                        │
│            ▼                                        │
│   邻居节点特征聚合：                                 │
│   - gather(current_edge) → neighboring_feature      │
│   - best_headings_embedding (Linear 36→128)         │
│   - visited_headings_embedding (Linear 36→128)      │
│   - concat + Linear(384→128) → 邻居综合特征         │
│            │                                        │
│            ▼                                        │
│   SingleHeadAttention (指针网络)                    │
│   query = current_state_feature                     │
│   key = enhanced_neighbor_features                  │
│   mask = edge_padding_mask                          │
│            │                                        │
│            ▼                                        │
│   log_softmax → logp [B, K*3]                       │
│   (K个邻居 × 3个候选朝向)                           │
└─────────────────────────────────────────────────────┘
```

**关键设计细节**：
- **6 层 Transformer 编码器**：充分建模节点间的关系依赖
- **1D 卷积前沿特征融合**：将 36 维角度分布压缩为 128 维嵌入，与编码器输出拼接
- **指针网络（SingleHeadAttention）**：带 tanh 裁剪（scale=10）的注意力机制，输出 log-softmax，自然支持可变数量的候选动作
- **朝向候选机制**：每个邻居节点的 3 个最优朝向作为独立候选动作，将联合选择解耦为 (目标节点, 朝向) 的乘积空间

### 9.2 QNet（Q 值网络）

QNet 的结构与 PolicyNet 高度相似，关键区别：

1. **输入维度不同**：GT 模式下 `node_input_dim = 7`（多了 explored 标志）
2. **输出结构不同**：输出 `[B, K*3, 1]` 的标量 Q 值（而非 log-probabilities）
3. **MAAC 模块**（TRAIN_ALGO=1 或 3 时启用）：
   ```
   all_agent_node_feature = gather(all_agent_indices)         # 每个智能体当前节点
   all_agent_selected_feature = gather(all_agent_next_indices) # 每个智能体目标节点
   concat + Linear(256→128) → all_agent_action_features
   
   agent_decoder (1层 Decoder, 4头) cross-attention:
     query = current_state_feature
     key/value = all_agent_action_features
     mask = agent_mask (剔除自己)
   → global_state_action_feature
   
   Q = Linear(384→1)(
       concat(current_state, neighbor_feature, global_state_action_feature)
   )
   ```

**双 Q 网络**：Q1 和 Q2 结构完全相同但独立初始化，取 min(Q1, Q2) 减少过估计偏差。

---

## 10. 完整运行流程

### 10.1 训练流程

#### 启动训练

```bash
conda activate marvel
python driver.py
```

#### 训练流程详解

```
                           ┌──────────────────┐
                           │   driver.py      │
                           │   (主训练进程)    │
                           │                  │
                           │  全局模型：       │
                           │  - PolicyNet     │
                           │  - QNet1, QNet2  │
                           │  - TargetQ1, Q2  │
                           │  - log_alpha     │
                           └──────┬───────────┘
                                  │ 分发权重
                    ┌─────────────┼─────────────┐
                    │             │             │
              ┌─────▼─────┐ ┌───▼───┐  ┌─────▼─────┐
              │ RLRunner 0│ │  ...  │  │RLRunner 17│
              │ (Ray)     │ │       │  │(Ray)      │
              │           │ │       │  │           │
              │ MultiAgent│ │       │  │ MultiAgent│
              │ Worker    │ │       │  │ Worker    │
              └─────┬─────┘ └───┬───┘  └─────┬─────┘
                    │            │            │
                    │ 经验数据    │            │
                    └────────────┼────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  driver.py (训练端)      │
                    │                          │
                    │  1. ray.wait(job_list)   │
                    │  2. ray.get(done_id)     │
                    │  3. 存入 experience_buffer│
                    │  4. 采样 batch           │
                    │  5. SAC 更新 (×4)        │
                    │  6. 更新全局权重          │
                    │  7. 分发新权重给 workers  │
                    └──────────────────────────┘
```

**关键数值**：
- 18 个 worker 并行收集经验
- 每收集 1 个 episode 即触发一次训练（4 步参数更新）
- Buffer 上限 10000 条经验（~40 个 episode）
- 开始训练最少需要 2000 条经验
- 每 32 个 episode 保存 checkpoint
- 每 64 次 Q 网络更新后同步目标网络
- 每 32 个 episode 写 TensorBoard 日志

#### 输出文件

| 路径 | 内容 |
|------|------|
| `model/marvel/checkpoint.pth` | 定期保存的检查点（policy, Q1, Q2, log_alpha, optimizers, episode 计数器） |
| `train/marvel/` | TensorBoard 事件文件 |
| `gifs/marvel/` | 每隔 SAVE_IMG_GAP episode 的可视化 GIF |

#### 从检查点恢复训练

设置 `LOAD_MODEL = True`，并将 checkpoint 放在 `load_model/marvel/checkpoint.pth`，运行 `driver.py` 即可。

### 10.2 测试/评估流程

#### 启动评估

```bash
conda activate marvel
python test_driver.py
```

#### 测试流程

```
test_driver.py
  │
  ├── 加载预训练 PolicyNet（仅策略，无 QNet）
  ├── 创建 10 个 TestRunner (Ray remote)
  │
  └── 对每种配置 (n_agent, fov, sensor_range)：
        │
        ├── 分发 100 个测试 episode 给 10 个 worker
        ├── 每个 test worker：
        │     ├── TestWorker.run_episode()
        │     │     ├── 每步：get_observation(pad=False)
        │     │     ├── select_next_waypoint(greedy=True)
        │     │     ├── 冲突解决
        │     │     └── 运动模拟
        │     └── 返回 perf_metrics
        │
        └── 聚合和打印所有指标
```

**注意**：测试时 `pad=False`，不使用填充（不需要 batch 处理）。测试使用确定性贪婪策略。

#### 输出指标示例

```
|#Test set: maps_test
|#Total test: 100
|#Number of agents: 6
|#FOV (degrees): 120
|#Sensor range (m): 10
|#Average max length: 284.5
|#Max max length: 412.3
|#Min max length: 156.8
|#Std max length: 52.3
|#Average explored rate: 0.987
|#Average success rate: 0.95
|#Average distance to 0.9 explored: 198.4
|#Std distance to 0.9 explored: 45.2
|#Average overlap ratio: 0.15
|#Std overlap ratio: 0.08
```

---

## 11. 关键设计决策与创新点

### 11.1 分层决策架构

MARVEL 采用 **高层 RL 决策 + 低层路径规划** 的分层架构：

```
RL 策略（PolicyNet）
    ↓ 选择目标节点 + 朝向
A* 路径规划（NodeManager.a_star）
    ↓ 生成可达路径
运动模型（compute_allowable_heading）
    ↓ 动力学约束的朝向执行
传感器模拟
    ↓ 更新信念地图
```

这种设计的好处：
- RL 只需学习高层探索策略，不需要处理底层控制
- A* 保证路径可达性和安全性
- 运动模型保证物理可行性

### 11.2 图基状态表示

使用规则网格节点（4m 间距）而非密集栅格作为状态表示：
- **降维**：360 个节点的图比数万像素的栅格更紧凑
- **泛化**：图结构对地图大小和形状的变化更鲁棒
- **结构化**：节点间通过邻接矩阵编码空间关系
- **注意力适用**：Transformer 天然适合处理图结构数据

### 11.3 前沿分布 + 朝向特征融合

新颖的朝向感知前沿表示：
- **前沿角度分布**：36 个角度桶记录每个节点周围的前沿密度分布
- **朝向访问记录**：记录已经观察过的角度范围，避免重复
- **滑动窗口最优朝向**：在 FoV 窗口内找到前沿密度最高的方向
- **多候选朝向**：为每个节点提供 3 个最优候选朝向，增加策略灵活性

### 11.4 集中训练 - 分散执行（CTDE）

- **训练时**：QNet 可以访问 Ground Truth 全局信息 + 所有智能体的动作
- **执行时**：PolicyNet 仅使用局部观测（自身信念地图 + 局部图）
- **参数共享**：所有机器人共享同一套策略网络参数
- **优势**：分散执行降低通信需求，集中训练提高学习效率

### 11.5 冲突解决机制

多机器人选择同一目标时的冲突解决：
1. 按距离排序（距离近的优先）
2. 距离近的保持原选择
3. 距离远的通过四叉树最近邻搜索寻找替代节点
4. 确保替代节点未被其他机器人选择

### 11.6 连续运动模拟

使用 6 个子步（`NUM_SIM_STEPS=6`）在每个决策步之间模拟连续运动：
- 传感器在每个子步更新（更真实的增量感知）
- 朝向在子步之间平滑过渡
- 模拟真实的增量式探索过程

---

## 12. 原版 baseline 的工程缺陷诊断

> 本章节基于对 `D:\工作区\MARVEL\` 原版代码的逐行审阅，**不针对论文实验设置内成功的部分**（论文 Table I-III 在 N ∈ {2,4,8} 时全部 100% 成功），而是聚焦于"在论文外场景将会失效"的设计缺陷。这些缺陷与 `期刊级研究规划.md` 的 C1-C5 一一对应。

### 12.1 决策与协调层

#### 12.1.1 协调机制完全依赖端到端涌现
原版 4 件套协调机制（occupancy ternary + 距离排序 reroute + privileged critic + 运动模型兜底）**没有任何显式联合优化层**：

- `multi_agent_worker.py:75-110` 中的 reroute 仅按距离排序 + 四叉树最近邻替代，不考虑替代节点的 utility / Q 值；
- `occupancy ∈ {-1, 0, 1}` 信息容量极低，多 Agent 进入同一区域时无法区分；
- attentive privileged critic（`model.py:368-380` 的 `agent_decoder`）仅训练期生效，执行期 actor 仍独立采样；
- 这套机制依赖 SAC 训练时把"避让"压入策略权重——但**当 N_agents 偏离训练分布、或地图拓扑变化时，无显式 fallback**。

**对接：C1 VJC（Value-Aware Joint Coordination）**——把执行期协调升级为价值驱动的 K×3 联合 ILP 分配，作为 reroute 的 principled 替代。

#### 12.1.2 朝向脱离协调
原版动作空间是 (节点 × 朝向) = K×3 = 75，但 reroute 只对节点冲突重路由，朝向**强制保留**。当目标节点被替换后，原本对应"目标方向"的朝向变成了"反方向"——与 MARVEL 的 heading-aware 核心创新逻辑冲突。

**对接：C1 VJC 的 K×3 联合分配**。

#### 12.1.3 距离排序破坏 SAC 信用对称性
`arriving_sequence = np.argsort(dist_list)` 让远距离 Agent 总是让步。这等价于隐含的"二等公民"——长期训练中那些初始距离偏大的 Agent 会习得"主动放弃高价值节点"的退化策略。

**对接：C1 VJC 用 SAC 目标 Q-α log π 作为代价，对所有 Agent 对称处理**。

### 12.2 通信与可扩展性层

#### 12.2.1 全连通通信硬假设
论文 Section III.A 显式声明 *"perfect communication between agents, allowing them to exchange information and maintain a shared map"*。代码层面，`env.py:robot_belief` 是单一共享变量，所有 Agent 在 `update_robot_belief` 中写入同一个数组——这等价于零延迟、零丢包、无距离衰减的理想通信信道。

**对接：C3 CC-CTDE**——把通信图、距离限制、丢包率作为训练随机化变量。

#### 12.2.2 N_AGENTS / 地图尺度硬编码
- `parameter.py:35` `N_AGENTS = 4` 是常量；`env.py:39` 用 `N_AGENTS` 采样初始位置——训练时 N 不变；
- `parameter.py:83` `NODE_PADDING_SIZE = 360` 是常量，无 batch 内动态调整；
- 测试时虽然可改 N，但策略本身从未见过 N≠4 的训练分布。

**对接：C4 C³T**——课程式 N、FoV、ds、map_scale 采样 + Set-encoded occupancy 实现置换不变 + sinusoidal 坐标编码实现尺度无关。

### 12.3 训练算法层

#### 12.3.1 折扣因子 γ=1 与有限视界的张力
- `parameter.py:73` GAMMA=1，但 `MAX_EPISODE_STEP=128` 是软上限；
- 当 episode 在 128 步被截断（未完成探索），Q 网络仍按"未完成"自举 → bootstrap 偏差；
- 这在 Pearl/Sutton 文献中被称为 *spurious bootstrapping*，对小 buffer 的 SAC 尤其有害。

**对接：可纳入 C5 理论分析章节作为命题 2 的边界条件讨论**。

#### 12.3.2 目标网络硬复制 vs SAC 标准软更新
`driver.py:332-338` 每 64 次 Q 更新执行完整 `load_state_dict`（硬复制），而 SAC 原文用 Polyak 软更新 τ=0.005。硬更新在 LR=1e-5 的小步长下会引入周期性目标突变，可能减慢收敛或引发振荡。

**对接：工程优化项**，不直接构成方法 contribution，但实施 C1-C4 时可顺手修补。

#### 12.3.3 Replay buffer 容量与样本利用率失衡
- `REPLAY_SIZE=10000`、N_AGENTS=4、平均 episode ~100 步 → 约 25 个 episode 的数据；
- SAC 通用配置是 1e5–1e6；
- BATCH_SIZE=256，每 episode 4 次更新，样本平均复用率仅 ~1.5 次。

**对接：工程项 + C5 理论命题的样本复杂度边界**。

#### 12.3.4 trajectory_reward 几乎是常数
`multi_agent_worker.py:179` `trajectory_reward = np.cos(robot.heading - trajectory_angle)`，但 `compute_allowable_heading` 已经把 final_heading 强行设为运动方向——在 yaw_rate 充足时该项恒为 1，不传递梯度信号。

**对接：奖励函数清理项**，可在 C1 实施时同步移除。

### 12.4 网络架构层

#### 12.4.1 SingleHeadAttention 未对填充候选屏蔽 logits
`model.py:42-44` 的 `SingleHeadAttention.forward`：

```python
if mask is not None:
    U = U.masked_fill(mask == 1, -1e8)
attention = torch.log_softmax(U, dim=-1)
```

mask 只屏蔽**编码器侧**的 `node_padding_mask`；指针网络对 K×3=75 个候选的输出层**没有对邻居层的 edge_padding_mask 屏蔽**。当真实邻居数 < 25 时，`multinomial(logp.exp())` 可能采样到填充位置（节点索引 0）。

**对接：实施 C1 VJC 时需要顺手修复此 bug**（VJC 的 cost 矩阵也需要对填充候选 mask）。

#### 12.4.2 occupancy ternary 信息瓶颈
`agent.py` 中 `occupancy` 是 scalar，编码三种状态。对 N=2 或 N=4 时勉强可用，但 N≥8 时同一节点附近可能有多个 Agent，ternary 信号丧失分辨能力。

**对接：C4 C³T 的 Set-encoded occupancy** —— 将 scalar 升级为对队友 set 的 attention 输出。

### 12.5 工程实现层

| 缺陷 | 文件 | 影响 |
|------|------|------|
| 训练/测试默认参数不完全一致 | `parameter.py` vs `test_parameter.py` 在 `UPDATING_MAP_SIZE` 等参数上不同 | 已知偏差，文档需补 |
| Ray worker 无 seed 控制 | `runner.py` | episode 复现性差 |
| `NUM_EPISODE_BUFFER=40` 是字段数非 episode 数 | `parameter.py:32` | 命名误导 |
| 视频生成依赖 episode_index 全局唯一 | `env.py:18` | 多进程并发可能冲突 |

**对接：工程修补项**，在 Week 12-13 整体修复。

---

## 13. 与期刊级研究规划的映射

下表把第 12 章列出的 baseline 缺陷与 `期刊级研究规划.md` 中的 5 个 contribution 一一对接，作为两份文档的**接口表**：

| 缺陷 ID | 描述 | 对应 contribution | 期刊规划章节 |
|---------|------|------------------|--------------|
| 12.1.1 | 协调机制无显式联合优化层 | **C1 VJC** | §3 |
| 12.1.2 | 朝向脱离协调 | **C1 VJC（K×3 联合）** | §3.2 Step 2 |
| 12.1.3 | 距离排序破坏 SAC 对称性 | **C1 VJC + C5 命题 2** | §3.4 + §7 |
| 12.2.1 | 全连通通信硬假设 | **C3 CC-CTDE** | §5 |
| 12.2.2 | N_AGENTS / 地图尺度硬编码 | **C4 C³T** | §6 |
| 12.3.1 | γ=1 与有限视界张力 | **C5 命题 2 边界** | §7.1 |
| 12.3.2 | 目标网络硬复制 | 工程项 | §10 Week 12 |
| 12.3.3 | Replay buffer 过小 | 工程项 + C5 样本复杂度 | §7.1 |
| 12.3.4 | trajectory_reward 失效 | 奖励清理 | §3.5 实施时顺手 |
| 12.4.1 | SingleHeadAttention 未 mask logits | C1 VJC 实施时修复 | §3.5 |
| 12.4.2 | occupancy ternary 瓶颈 | **C4 C³T Set Attention** | §6.2.2 |
| 12.5.* | 工程实现层 | 全量修补 | §10 Week 12 |

**两份文档的角色分工**：

- 本文档（`MARVEL技术文档.md`）= baseline 解读 + 缺陷诊断（**"是什么"**）
- `期刊级研究规划.md` = 创新方案 + 实施细节 + 实验设计（**"做什么"**）
- `缺陷诊断与切入点决策.md` = 优先级与切入策略（**"先做什么"**）

---

## 14. 推荐的研究切入路径

根据 C1-C5 的 dependency 与最低风险原则：

```
        Week 1-2   →   Week 3-4   →   Week 5     →   Week 6-7   →   Week 8+
        ────────       ────────       ─────         ─────────       ──────
        C1 VJC     →   C2 ISCP   →   C3 CC-CTDE →   C4 C³T      →   C5 Theory
        (基础)         (去中心化)     (通信受限)      (跨规模)        (理论)
            │              │              │              │              │
            └───────────────────── Week 12 全量 ablation ─────────────────┘
                                       │
                                       ▼
                              Week 13-15 论文写作
```

**最关键的 1 个决策点**：Week 2 末（C1 实施完）做一次 50-ep × 3 cost mode 的 MVV（Minimum Viable Validation）。如果 `Q − α log π` 的探索率 ≥ 原版 reroute 的 95%，整个 13-15 周计划证明可行；否则按规划 §12 fallback。

---

*文档生成日期：2026-06-25（v2，对接 baseline = `D:\工作区\MARVEL\`）*
*配套文档：`期刊级研究规划.md`（科研方向）、`缺陷诊断与切入点决策.md`（切入策略）*


## 附录 A：快速参考命令

```bash
# 环境安装
conda env create -f marvel.yml
conda activate marvel

# 训练（使用默认配置）
python driver.py

# 评估（使用预训练模型）
python test_driver.py

# 单次可视化运行
cd utils
python multi_agent_worker.py   # 训练模式单次运行
python test_worker.py          # 测试模式单次运行

# 查看 TensorBoard
tensorboard --logdir=train/marvel
```

## 附录 B：数据格式说明

### 地图文件格式
- PNG 灰度图像
- 像素值 255：自由空间
- 像素值 1：障碍物
- 像素值 127：未知
- 像素值 208：机器人初始位置

### 坐标系统
- 世界坐标系：以米为单位，原点在地图初始化位置
- 栅格坐标系：以 cell 为单位的整数索引
- 转换函数：`get_cell_position_from_coords()` / `get_coords_from_cell_position()`

### 模型检查点格式
```python
checkpoint = {
    "policy_model": state_dict,
    "q_net1_model": state_dict,
    "q_net2_model": state_dict,
    "log_alpha": tensor,
    "policy_optimizer": state_dict,
    "q_net1_optimizer": state_dict,
    "q_net2_optimizer": state_dict,
    "log_alpha_optimizer": state_dict,
    "episode": int
}
```

## 附录 C：依赖关系图

```
parameter.py ─────────────────────────────────────────────┐
    │                                                      │
    ├── driver.py ──── model.py ──── runner.py ──── multi_agent_worker.py
    │       │              │              │                   │
    │       │              │              │          ┌────────┴────────┐
    │       │              │              │          │                 │
    │       │              │              │      agent.py         env.py
    │       │              │              │          │                 │
    │       │              │              │    ┌─────┴─────┐     sensor.py
    │       │              │              │    │           │     motion_model.py
    │       │              │              │ node_manager  ground_truth
    │       │              │              │       │      _node_manager
    │       │              │              │       │           │
    │       │              │              │    quads.py    quads.py
    │       │              │              │
    │       │              │           utils.py (MapInfo, 前沿检测, 碰撞检测...)
    │       │              │
    │    test_driver.py ── test_worker.py ─── agent.py + env_test.py + node_manager.py
    │
    test_parameter.py
```

---

*文档生成日期：2026-05-28*
*对应代码版本：MARVEL-main (ICRA 2025)*
