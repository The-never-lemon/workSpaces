import numpy as np
import imageio
import os
from scipy.optimize import linear_sum_assignment
from skimage.morphology import label

from parameter import *


def get_cell_position_from_coords(coords, map_info, check_negative=True):
    single_cell = False
    if coords.flatten().shape[0] == 2:
        single_cell = True

    coords = coords.reshape(-1, 2)
    coords_x = coords[:, 0]
    coords_y = coords[:, 1]
    cell_x = ((coords_x - map_info.map_origin_x) / map_info.cell_size)
    cell_y = ((coords_y - map_info.map_origin_y) / map_info.cell_size)

    cell_position = np.around(np.stack((cell_x, cell_y), axis=-1)).astype(int)

    if check_negative:
        assert sum(cell_position.flatten() >= 0) == cell_position.flatten().shape[0], print(cell_position, coords, map_info.map_origin_x, map_info.map_origin_y)
    if single_cell:
        return cell_position[0]
    else:
        return cell_position

def get_coords_from_cell_position(cell_position, map_info):
    cell_position = cell_position.reshape(-1, 2)
    cell_x = cell_position[:, 0]
    cell_y = cell_position[:, 1]
    coords_x = cell_x * map_info.cell_size + map_info.map_origin_x
    coords_y = cell_y * map_info.cell_size + map_info.map_origin_y
    coords = np.stack((coords_x, coords_y), axis=-1)
    coords = np.around(coords, 1)
    if coords.shape[0] == OCCUPIED:
        return coords[0]
    else:
        return coords

def get_free_area_coords(map_info):
    free_indices = np.where(map_info.map == FREE)
    free_cells = np.asarray([free_indices[1], free_indices[0]]).T
    free_coords = get_coords_from_cell_position(free_cells, map_info)
    return free_coords


def get_free_and_connected_map(location, map_info):
    free = (map_info.map == FREE).astype(float)
    labeled_free = label(free, connectivity=2)
    cell = get_cell_position_from_coords(location, map_info)
    label_number = labeled_free[cell[1], cell[0]]
    connected_free_map = (labeled_free == label_number)
    return connected_free_map

def get_updating_node_coords(location, updating_map_info, check_connectivity=True):
    x_min = updating_map_info.map_origin_x
    y_min = updating_map_info.map_origin_y
    x_max = updating_map_info.map_origin_x + (updating_map_info.map.shape[1] - 1) * CELL_SIZE
    y_max = updating_map_info.map_origin_y + (updating_map_info.map.shape[0] - 1) * CELL_SIZE

    if x_min % NODE_RESOLUTION != 0:
        x_min = (x_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
    if x_max % NODE_RESOLUTION != 0:
        x_max = x_max // NODE_RESOLUTION * NODE_RESOLUTION
    if y_min % NODE_RESOLUTION != 0:
        y_min = (y_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
    if y_max % NODE_RESOLUTION != 0:
        y_max = y_max // NODE_RESOLUTION * NODE_RESOLUTION

    x_coords = np.arange(x_min, x_max + 0.1, NODE_RESOLUTION)
    y_coords = np.arange(y_min, y_max + 0.1, NODE_RESOLUTION)
    t1, t2 = np.meshgrid(x_coords, y_coords)
    nodes = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    nodes = np.around(nodes, 1)

    free_connected_map = None

    if not check_connectivity:

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, updating_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < updating_map_info.map.shape[0] and 0 <= cell[0] < updating_map_info.map.shape[1]
            if updating_map_info.map[cell[1], cell[0]] == FREE:
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices].reshape(-1, 2)

    else:
        free_connected_map = get_free_and_connected_map(location, updating_map_info)
        free_connected_map = np.array(free_connected_map)

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, updating_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < free_connected_map.shape[0] and 0 <= cell[0] < free_connected_map.shape[1]
            if free_connected_map[cell[1], cell[0]] == 1:
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices].reshape(-1, 2)

    return nodes, free_connected_map

def get_frontier_in_map(map_info):
    x_len = map_info.map.shape[1]
    y_len = map_info.map.shape[0]
    unknown = (map_info.map == UNKNOWN) * 1
    unknown = np.lib.pad(unknown, ((1, 1), (1, 1)), 'constant', constant_values=0)
    unknown_neighbor = unknown[2:][:, 1:x_len + 1] + unknown[:y_len][:, 1:x_len + 1] + unknown[1:y_len + 1][:, 2:] \
                       + unknown[1:y_len + 1][:, :x_len] + unknown[:y_len][:, 2:] + unknown[2:][:, :x_len] + \
                       unknown[2:][:, 2:] + unknown[:y_len][:, :x_len]
    free_cell_indices = np.where(map_info.map.ravel(order='F') == FREE)[0]
    frontier_cell_1 = np.where(1 < unknown_neighbor.ravel(order='F'))[0]
    frontier_cell_2 = np.where(unknown_neighbor.ravel(order='F') < 8)[0]
    frontier_cell_indices = np.intersect1d(frontier_cell_1, frontier_cell_2)
    frontier_cell_indices = np.intersect1d(free_cell_indices, frontier_cell_indices)

    x = np.linspace(0, x_len - 1, x_len)
    y = np.linspace(0, y_len - 1, y_len)
    t1, t2 = np.meshgrid(x, y)
    cells = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    frontier_cell = cells[frontier_cell_indices]

    frontier_coords = get_coords_from_cell_position(frontier_cell, map_info).reshape(-1, 2)
    if frontier_cell.shape[0] > 0 and FRONTIER_CELL_SIZE != CELL_SIZE:
        frontier_coords = frontier_coords.reshape(-1 ,2)
        frontier_coords = frontier_down_sample(frontier_coords)
    else:
        frontier_coords = set(map(tuple, frontier_coords))
    return frontier_coords

def frontier_down_sample(data, voxel_size=FRONTIER_CELL_SIZE):
    voxel_indices = np.array(data / voxel_size, dtype=int).reshape(-1, 2)

    voxel_dict = {}
    for i, point in enumerate(data):
        voxel_index = tuple(voxel_indices[i])

        if voxel_index not in voxel_dict:
            voxel_dict[voxel_index] = point
        else:
            current_point = voxel_dict[voxel_index]
            if np.linalg.norm(point - np.array(voxel_index) * voxel_size) < np.linalg.norm(
                    current_point - np.array(voxel_index) * voxel_size):
                voxel_dict[voxel_index] = point

    downsampled_data = set(map(tuple, voxel_dict.values()))
    return downsampled_data

def is_frontier(location, map_info):
    cell = get_cell_position_from_coords(location, map_info)
    if map_info.map[cell[1], cell[0]] != FREE:
        return False
    else:
        assert cell[1] - 1 > 0 and cell[1] - 1 > 0 and cell[1] + 2 < map_info.map.shape[1] and cell[0] + 2 < map_info.map.shape[0]
        unknwon = map_info.map[cell[1] - 1:cell[1] + 2, cell[0] - 1: cell[0] + 2] == UNKNOWN
        n = np.sum(unknwon)
        if 1 < n < 8:
            return True
        else:
            return False

def check_collision(start, end, map_info):
    # Bresenham line algorithm checking
    collision = False

    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    map = map_info.map

    x0 = start_cell[0]
    y0 = start_cell[1]
    x1 = end_cell[0]
    y1 = end_cell[1]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        k = map.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        if k == OCCUPIED:
            collision = True
            break
        if k == UNKNOWN:
            collision = True
            break
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return collision


def make_gif(path, n, frame_files, rate):
    with imageio.get_writer('{}/{}_explored_rate_{:.4g}.gif'.format(path, n, rate), mode='I', duration=1) as writer:
        for frame in frame_files:
            image = imageio.imread(frame)
            writer.append_data(image)
    print('gif complete\n')

    for filename in frame_files[:-1]:
        os.remove(filename)

def make_gif_test(path, n, frame_files, rate, n_agents, fov, sensor_range):
    with imageio.get_writer('{}/{}_{}_{}_{}_explored_rate_{:.4g}.gif'.format(path, n, n_agents, fov, sensor_range, rate), mode='I', duration=1) as writer:
        for frame in frame_files:
            image = imageio.imread(frame)
            writer.append_data(image)
    print('gif complete\n')
    for filename in frame_files[:-1]:
        os.remove(filename)


def joint_assignment(agent_scores_list):
    """
    使用 Hungarian 算法求解多智能体目标节点的全局最优一对一分配。

    Args:
        agent_scores_list: list[dict], 每个 dict 映射 {global_node_idx: log_prob_score}

    Returns:
        assignments: list[int], 每个 agent 分配到的全局节点索引
    """
    n_agents = len(agent_scores_list)

    # 收集所有可达节点
    all_nodes = set()
    for scores in agent_scores_list:
        if scores:
            all_nodes.update(scores.keys())
    all_nodes = sorted(all_nodes)
    n_nodes = len(all_nodes)

    # 无节点可达的退化情况
    if n_nodes == 0:
        return [0] * n_agents

    node_to_col = {node: col for col, node in enumerate(all_nodes)}

    # 构建代价矩阵：不可达节点代价为极大值
    INF = 1e8
    n_cols = max(n_agents, n_nodes)
    cost_matrix = np.full((n_agents, n_cols), INF, dtype=np.float64)

    for i, scores in enumerate(agent_scores_list):
        if not scores:
            continue
        for node_idx, log_score in scores.items():
            cost = -log_score
            # NaN/Inf 守卫：防止传入 scipy C 层导致 segfault
            if cost != cost or cost == float('inf') or cost == float('-inf'):
                continue
            cost_matrix[i, node_to_col[node_idx]] = cost

    # 二重保险：清除代价矩阵中残留的非有限值
    cost_matrix = np.nan_to_num(cost_matrix, nan=INF, posinf=INF, neginf=INF)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignments = [0] * n_agents
    used_nodes = set()
    for i in range(n_agents):
        col = col_ind[i]
        if col < n_nodes:
            node = all_nodes[col]
            assignments[i] = node
            used_nodes.add(node)

    # 哑列回退：选择该 Agent 得分最高且尚未被占用的节点
    for i in range(n_agents):
        if col_ind[i] >= n_nodes and agent_scores_list[i]:
            sorted_nodes = sorted(agent_scores_list[i], key=agent_scores_list[i].get, reverse=True)
            for node in sorted_nodes:
                if node not in used_nodes:
                    assignments[i] = node
                    used_nodes.add(node)
                    break
            else:
                assignments[i] = sorted_nodes[0]  # 无可避免的冲突，选最高分

    return assignments


class MapInfo:
    def __init__(self, map, map_origin_x, map_origin_y, cell_size):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
        self.cell_size = cell_size

    def update_map_info(self, map, map_origin_x, map_origin_y):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y


