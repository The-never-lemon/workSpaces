# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MARVEL (Multi-Agent Reinforcement Learning for constrained field-of-View multi-robot Exploration in Large-scale environments) — an ICRA 2025 paper. A neural framework using graph attention networks + frontier/orientation feature fusion to develop decentralized collaborative policies via MARL for robots with constrained FoV.

**Core idea:** High-level RL policy (Transformer pointer network) selects a target graph node + heading; low-level A* path planning + kinematics model handle physical execution. CTDE (centralized training with decentralized execution) using the SAC algorithm.

Definitive technical reference: `MARVEL技术文档.md` (Chinese, exhaustive).

## Environment Setup

```bash
conda env create -f marvel.yml
conda activate marvel
```

Key dependencies: Python 3.11.8, PyTorch 2.5.1 (CUDA 12.4), Ray 2.44.0, gymnasium 0.28.1.

## Commands

```bash
# Train (with default config from parameter.py)
python driver.py

# Evaluate (with pretrained model from load_model/MARVEL/)
python test_driver.py

# TensorBoard monitoring
tensorboard --logdir=train/marvel

# Single-episode dry runs (inside utils/)
python multi_agent_worker.py   # training mode
python test_worker.py          # test mode
```

## Configuration

All global config lives in two files:

- **`parameter.py`** — training: `N_AGENTS`, `FOV`, `SENSOR_RANGE`, network dimensions (`EMBEDDING_DIM=128`, `NODE_INPUT_DIM=6`), RL hyperparameters (`LR=1e-5`, `GAMMA=1`, `BATCH_SIZE=256`), `TRAIN_ALGO` (0-3), `NUM_META_AGENT=18` (Ray workers), `USE_GPU_GLOBAL=True`/`USE_GPU=False`.
- **`test_parameter.py`** — testing: `NUM_TEST=100`, `NUM_META_AGENT=10`, `GREEDY=True`.

`TRAIN_ALGO` modes:
- `0` — Pure SAC (local obs only)
- `1` — MAAC (QNet sees all agents' actions)
- `2` — Ground Truth (QNet uses global oracle state)
- `3` — MAAC + Ground Truth (default, most complete centralized training)

`USE_JOINT_ASSIGNMENT` in parameter.py toggles Hungarian matching for multi-agent goal assignment (experimental, significantly improves exploration rate per test results).

## Architecture

### Training flow (distributed SAC via Ray)

```
driver.py (main process)
  ├── Global networks: PolicyNet, QNet1, QNet2, TargetQNet1, TargetQNet2
  ├── 18 × RLRunner (Ray remote workers)
  │     └── MultiAgentWorker → runs one episode:
  │           ├── Env (occupancy grid simulation)
  │           ├── NodeManager (local graph, Quadtree-indexed)
  │           ├── GroundTruthNodeManager (oracle graph for critic)
  │           └── N_AGENTS × Agent
  │                 ├── update_graph() → detect frontiers, update nodes
  │                 ├── get_observation() → 9-tensor observation
  │                 ├── select_next_waypoint(policy_net) → (node, heading)
  │                 └── Continuous motion sim (6 substeps, kinematics-constrained)
  └── Experience buffer → sample batch → SAC update (×4 per episode)
```

Testing flow (`test_driver.py` → `TestWorker`):
- Only PolicyNet used (no QNet, no GroundTruthNodeManager)
- Greedy deterministic action selection
- Records exploration progress history and overlap ratios

### Observation space (PolicyNet input)

9 tensors per agent:
1. `node_inputs` [N, 6] — per-node (rel_x, rel_y, utility, guidepost, occupancy, best_heading_angle)
2. `node_padding_mask` — padded to `NODE_PADDING_SIZE=360`
3. `edge_mask` — adjacency matrix mask
4. `current_index` — agent's current node
5. `current_edge` — edges from current node to K=25 neighbors
6. `edge_padding_mask`
7. `frontier_distribution` [N, 36] — 36-bin angular distribution of frontiers per node
8. `heading_visited` [N, 36] — which angles have been visited
9. `neighbor_best_headings` [K, 3, 36] — top-3 heading candidates per neighbor

### Action space

Joint selection of `(neighbor_node ∈ {0..K-1}, heading_candidate ∈ {0..2})` → output dimension = K × 3 = 75.

### PolicyNet architecture

- 6-layer Transformer Encoder (8-head MHA) on node features
- 1D Conv fusion of frontier angular distribution (36→128) with encoder output
- 1-layer Decoder cross-attention to contextualize the current node
- Pointer network (SingleHeadAttention with tanh clipping, scale=10) over neighbor features → log_softmax

### Reward function (per step per robot)

`reward = utility_reward + trajectory_reward + team_reward`

- **Utility reward:** number of observable frontiers from chosen node (normalized)
- **Trajectory reward:** cos(heading − motion_direction)
- **Team reward:** shared global frontier reduction (normalized, −0.5 baseline penalty)
- **Completion bonus:** +10 when exploration succeeds

### Key utility files

| File | Role |
|------|------|
| `utils/model.py` | PolicyNet, QNet, Transformer encoder/decoder, pointer attention |
| `utils/agent.py` | Single robot: graph update, observation building, waypoint selection |
| `utils/multi_agent_worker.py` | Episode runner: agent orchestration, conflict resolution, reward calc, motion sim |
| `utils/env.py` | Occupancy grid environment, belief map, frontier tracking |
| `utils/env_test.py` | Test environment variant (variable agent count, fixed initial heading) |
| `utils/node_manager.py` | Local graph: node creation/update, neighbor relations, Dijkstra/A* |
| `utils/ground_truth_node_manager.py` | Oracle graph for centralized critic (7-dim nodes with `explored` flag) |
| `utils/sensor.py` | LiDAR-like ray casting (Bresenham) with FoV constraints |
| `utils/motion_model.py` | Yaw-rate-constrained heading computation |
| `utils/quads.py` | Quadtree spatial index for O(log n) node lookup |
| `utils/utils.py` | Map coordinate transforms, frontier detection, collision checking, GIF export |
| `utils/runner.py` | Ray-distributed training worker wrapper |
| `utils/test_worker.py` | Test-mode episode runner |

### Map data

- Training: `maps_medium/` (4000 PNGs)
- Testing: `maps_test/` (100 PNGs)
- Pixel encoding: 255=free, 1=occupied, 127=unknown, 208=robot start position
- Internal resolution: `CELL_SIZE=0.4m`, graph nodes every `NODE_RESOLUTION=4m`

### Checkpoint format

```python
checkpoint = {
    "policy_model": state_dict,
    "q_net1_model": state_dict, "q_net2_model": state_dict,
    "log_alpha": tensor,
    "policy_optimizer": state_dict, "q_net1_optimizer": state_dict,
    "q_net2_optimizer": state_dict, "log_alpha_optimizer": state_dict,
    "episode": int
}
```

Saved to `model/marvel/checkpoint.pth` every 32 episodes. Load from `load_model/MARVEL/checkpoint.pth` by setting `LOAD_MODEL=True`.
