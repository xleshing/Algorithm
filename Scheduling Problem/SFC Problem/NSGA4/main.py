import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
import time
from csv2list import csv2list
import os


#############################################
# 輔助函數：BFS 找最短路徑
#############################################
def bfs_shortest_path(graph, start, goal):
    """
    使用 BFS 找出從 start 到 goal 的最短路徑（以節點列表回傳），若無路徑則回傳 None
    """
    visited = set()
    queue = deque([[start]])
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor in graph.get(node, []):
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
    return None


def build_graph(network_nodes):
    """
    根據 network_nodes 建立圖結構，
    採用與 NSGA4 相同的方法：利用字典生成式建立 key 為節點 id，
    value 為該節點的鄰居列表。
    """
    # 注意：假設 network_nodes 為一個列表，每個元素是一個字典，
    # 且每個字典至少包含 'id' 與 'neighbors' 欄位。
    return {node['id']: node['neighbors'] for node in network_nodes}


def get_complete_path(assignment, graph):
    """
    根據一筆 SFC 的處理節點 assignment，
    利用 BFS 找出相鄰處理節點間的完整路徑，並將所有段落串接成一個完整的物理路徑
    """
    complete_path = []
    for i in range(len(assignment) - 1):
        segment = bfs_shortest_path(graph, assignment[i], assignment[i + 1])
        if segment is None:
            segment = [assignment[i], assignment[i + 1]]
        if i == 0:
            complete_path.extend(segment)
        else:
            complete_path.extend(segment[1:])  # 避免重複加入前一段的終點
    return complete_path


#############################################
# 獨立目標函數定義
#############################################
def objective_load_balance(solution, network_nodes, sfc_requests, vnf_traffic):
    """
    目標1：最小化節點負載均衡
    根據每個節點處理特定 VNF 時的負載係數（load_per_vnf），
    累計負載後回傳負載標準差。
    """
    node_loads = {node_id: 0.0 for node_id in network_nodes.keys()}
    for req in sfc_requests:
        chain = req['chain']
        assignment = solution[req['id']]
        for i, node_id in enumerate(assignment):
            demand = vnf_traffic[chain[i]]
            # 使用對應 VNF 的負載係數
            load_factor = network_nodes[node_id]['load_per_vnf'][chain[i]]
            node_loads[node_id] += demand * load_factor
    loads_array = np.array(list(node_loads.values()))
    return np.std(loads_array)


def objective_end_to_end_delay_bfs(solution, network_nodes, edges, vnf_traffic, sfc_requests):
    """
    目標2：最小化端到端延遲
    對每筆 SFC，計算：
      - 各節點處理延遲（根據 node 對應 VNF 的處理延遲）
      - 邊延遲：對於每對連續處理節點，利用 BFS 找出完整路徑，
        累計沿路每條邊的延遲 (demand / capacity)
    """
    graph = {node_id: network_nodes[node_id]['neighbors'] for node_id in network_nodes}
    total_delay = 0.0
    for req in sfc_requests:
        chain = req['chain']
        # 這裡 demand 可視需求選擇使用 chain 中對應的 vnf_traffic，本例中各 VNF 流量相同
        demand = vnf_traffic[chain[0]]
        assignment = solution[req['id']]
        node_delay = sum(network_nodes[assignment[i]]['processing_delay'][chain[i]] for i in range(len(assignment)))
        edge_delay = 0.0
        for i in range(len(assignment) - 1):
            path = bfs_shortest_path(graph, assignment[i], assignment[i + 1])
            if path is None:
                raise ValueError(f"{assignment} path is None")
            else:
                for j in range(len(path) - 1):
                    n1, n2 = path[j], path[j + 1]
                    if (n1, n2) in edges:
                        cap = edges[(n1, n2)]
                    elif (n2, n1) in edges:
                        cap = edges[(n2, n1)]
                    else:
                        raise ValueError(f"{n1, n2} not in edges")
                    edge_delay += demand / cap
        total_delay += (node_delay + edge_delay)
    return total_delay


def objective_network_throughput(solution, edges, sfc_requests, vnf_traffic):
    """
    目標3：最大化網路吞吐量（取倒數以最小化）
    - 對每條邊累計經由該邊的流量（利用 BFS 找出完整路徑）
    - 若任一邊的累計流量超過其容量，則回傳 False
    - 否則，以所有邊累計流量的倒數作為目標值
    """
    # 建立無向圖
    graph = {}
    for (n1, n2) in edges:
        graph.setdefault(n1, []).append(n2)
        graph.setdefault(n2, []).append(n1)

    edge_flow = {edge: 0.0 for edge in edges.keys()}
    for req in sfc_requests:
        # 這裡假設各 VNF 流量相同，取 chain 中第一個 VNF 的流量需求
        demand = vnf_traffic[req['chain'][0]]
        assignment = solution[req['id']]
        for i in range(len(assignment) - 1):
            path = bfs_shortest_path(graph, assignment[i], assignment[i + 1])
            if path is None:
                raise ValueError(f"{assignment} path is None")
            for j in range(len(path) - 1):
                n1, n2 = path[j], path[j + 1]
                if (n1, n2) in edges:
                    edge_flow[(n1, n2)] += demand
                elif (n2, n1) in edges:
                    edge_flow[(n2, n1)] += demand

    if not within_capacity(edge_flow, edges):
        raise ValueError(f"{solution} within_capacity")

    # 僅計算所有邊累計流量的倒數作為目標值
    total_flow = sum(edge_flow.values())
    epsilon = 1e-6
    return 1 / (total_flow + epsilon)


def within_capacity(edge_flow, edges):
    """
    檢查每條邊的流量是否都在容量限制內。
    若有任一邊流量超過容量，則回傳 False，否則回傳 True。
    """
    for edge, flow in edge_flow.items():
        if flow > edges[edge]:
            return False
    return True


#############################################
# NSGA4_SFC 類別
#############################################
class NSGA4_SFC:
    def __init__(self, network_nodes, edges, sfc_requests, vnf_traffic, population_size, generations):
        """
        參數說明：
          network_nodes: 節點列表，每個元素包含 'id'、'vnf_types'、'neighbors'、'load_per_vnf'、'processing_delay'
          edges: 邊的字典，鍵為 (node1, node2) 的 tuple，值為邊容量
          sfc_requests: SFC 請求列表，每筆請求包含 'id' 與 'chain'（VNF 連鎖）
          vnf_traffic: 字典，定義每個 VNF type 所需流量
          population_size: 種群規模
          generations: 迭代代數
        """
        self.network_nodes = {node['id']: node for node in network_nodes}
        self.edges = edges
        self.sfc_requests = sfc_requests
        self.vnf_traffic = vnf_traffic
        self.population_size = population_size
        self.generations = generations
        # 初始種群：每筆解為一個字典 { request_id: [node1, node2, ...] }
        self.population = np.array([self.generate_feasible_solution() for _ in range(population_size)])

    def generate_feasible_solution(self):
        solution = {}
        for req in self.sfc_requests:
            solution[req['id']] = self.generate_feasible_assignment_for_request(req)
        return solution

    def generate_feasible_assignment_for_request(self, req):
        """
        對單筆請求，初始化一個可行的處理節點序列（不要求節點間相鄰），
        並利用 repair_assignment_for_request 檢查與修正分配方案。
        """
        chain = req['chain']
        assignment = []
        for vnf in chain:
            candidates = [node_id for node_id, node in self.network_nodes.items() if vnf in node['vnf_types']]
            if not candidates:
                raise ValueError(f"請求 {req['id']} 的 VNF {vnf} 無法分配到任何節點")
            assignment.append(np.random.choice(candidates))

        # 利用 repair_assignment_for_request 檢查並修正分配方案
        assignment = self.repair_assignment_for_request(req, assignment)
        return assignment

    def repair_assignment_for_request(self, req, assignment):
        """
        檢查並修正一個 SFC 請求中的節點分配方案，
        確保在每個位置上所選擇的節點能夠處理該位置所要求的 VNF 類型，
        並且與前一個節點間是可連通的（使用 BFS 檢查）。
        :param req: SFC 請求，包含 'chain'
        :param assignment: 原始的節點分配列表
        :return: 修正後的分配列表
        """
        chain = req['chain']
        # 建立網路圖：根據每個節點的鄰居資訊
        graph = {node_id: self.network_nodes[node_id]['neighbors'] for node_id in self.network_nodes}

        for i in range(len(assignment)):
            valid = True
            # 檢查當前節點是否具備處理對應 VNF 的能力
            if chain[i] not in self.network_nodes[assignment[i]]['vnf_types']:
                valid = False
            # 如果不是第一個節點，還要檢查與前一個節點之間是否連通
            if i > 0:
                prev_node = assignment[i - 1]
                # 使用 BFS 檢查從前一個節點到當前節點是否存在可行路徑
                if bfs_shortest_path(graph, prev_node, assignment[i]) is None:
                    valid = False
            # 如果不符合要求，則重新選擇一個候選節點
            if not valid:
                if i == 0:
                    # 第一個節點只需檢查處理能力
                    candidates = [node_id for node_id, node in self.network_nodes.items()
                                  if chain[i] in node['vnf_types']]
                else:
                    # 從所有能處理該 VNF 的節點中挑選出與前一個節點連通的候選
                    candidates = [node_id for node_id, node in self.network_nodes.items()
                                  if chain[i] in node['vnf_types'] and bfs_shortest_path(graph, assignment[i - 1],
                                                                                         node_id) is not None]
                if candidates:
                    assignment[i] = np.random.choice(candidates)
                else:
                    assignment = self.generate_feasible_assignment_for_request(req)
        return assignment

    def compute_objectives(self, solution):
        """
        計算目標值
        :param solution: 產生的解
        :return: 回傳一個array包含三個目標值
        """
        f1 = objective_load_balance(solution, self.network_nodes, self.sfc_requests, self.vnf_traffic)
        f2 = objective_end_to_end_delay_bfs(solution, self.network_nodes, self.edges, self.vnf_traffic,
                                            self.sfc_requests)
        f3 = objective_network_throughput(solution, self.edges, self.sfc_requests, self.vnf_traffic)
        return np.array([f1, f2, f3])

    # def objective_details(self, solution):
    #     details = {}
    #     # f1 詳細
    #     node_loads = {node_id: 0.0 for node_id in self.network_nodes.keys()}
    #     for req in self.sfc_requests:
    #         demand = self.vnf_traffic[req['chain'][0]]
    #         assignment = solution[req['id']]
    #         for i, node_id in enumerate(assignment):
    #             load_factor = self.network_nodes[node_id]['load_per_vnf'][req['chain'][i]]
    #             node_loads[node_id] += demand * load_factor
    #     details['node_loads'] = node_loads
    #     details['f1_std'] = np.std(np.array(list(node_loads.values())))
    #
    #     # f2 詳細：對每筆 SFC，計算節點延遲、各段邊延遲及完整物理路徑
    #     sfc_details = {}
    #     total_delay = 0.0
    #     graph = {node_id: self.network_nodes[node_id]['neighbors'] for node_id in self.network_nodes}
    #     for req in self.sfc_requests:
    #         demand = self.vnf_traffic[req['chain'][0]]
    #         assignment = solution[req['id']]
    #         node_delays = [self.network_nodes[assignment[i]]['processing_delay'][req['chain'][i]] for i in
    #                        range(len(assignment))]
    #         node_delay = sum(node_delays)
    #         edge_delays = []
    #         complete_path_segments = []
    #         for i in range(len(assignment) - 1):
    #             path = bfs_shortest_path(graph, assignment[i], assignment[i + 1])
    #             if path is None:
    #                 ed = 1e6
    #                 path = [assignment[i], assignment[i + 1]]
    #             else:
    #                 ed = 0.0
    #                 for j in range(len(path) - 1):
    #                     n1, n2 = path[j], path[j + 1]
    #                     if (n1, n2) in self.edges:
    #                         cap = self.edges[(n1, n2)]
    #                     elif (n2, n1) in self.edges:
    #                         cap = self.edges[(n2, n1)]
    #                     else:
    #                         cap = 1e-6
    #                     ed += demand / cap
    #             edge_delays.append(ed)
    #             complete_path_segments.append(path)
    #         sfc_total = node_delay + sum(edge_delays)
    #         # 串接各段完整路徑
    #         complete_path = []
    #         for idx, segment in enumerate(complete_path_segments):
    #             if idx == 0:
    #                 complete_path.extend(segment)
    #             else:
    #                 complete_path.extend(segment[1:])
    #         sfc_details[req['id']] = {
    #             'node_delays': node_delays,
    #             'edge_delays': edge_delays,
    #             'total_delay': sfc_total,
    #             'complete_physical_path': complete_path
    #         }
    #         total_delay += sfc_total
    #     details['sfc_delay_details'] = sfc_details
    #     details['total_delay'] = total_delay
    #     details['f2'] = total_delay
    #
    #     # f3 詳細：各邊流量利用率
    #     graph_edges = {}
    #     for (n1, n2) in self.edges:
    #         graph_edges.setdefault(n1, []).append(n2)
    #         graph_edges.setdefault(n2, []).append(n1)
    #     edge_flow = {edge: 0.0 for edge in self.edges.keys()}
    #     for req in self.sfc_requests:
    #         demand = self.vnf_traffic[req['chain'][0]]
    #         assignment = solution[req['id']]
    #         for i in range(len(assignment) - 1):
    #             path = bfs_shortest_path(graph_edges, assignment[i], assignment[i + 1])
    #             if path is None:
    #                 continue
    #             for j in range(len(path) - 1):
    #                 n1, n2 = path[j], path[j + 1]
    #                 if (n1, n2) in self.edges:
    #                     edge_flow[(n1, n2)] += demand
    #                 elif (n2, n1) in self.edges:
    #                     edge_flow[(n2, n1)] += demand
    #     throughput_sum = 0.0
    #     edge_utilization = {}
    #     for edge, flow in edge_flow.items():
    #         capacity = self.edges[edge]
    #         utilization = flow / capacity
    #         edge_utilization[edge] = {'flow': flow, 'capacity': capacity, 'utilization': utilization}
    #         throughput_sum += utilization
    #     epsilon = 1e-6
    #     details['edge_utilization'] = edge_utilization
    #     details['throughput_sum'] = throughput_sum
    #     details['f3'] = 1 / (throughput_sum + epsilon)
    #     return details

    def fast_non_dominated_sort(self, population_fitness):
        num_solutions = len(population_fitness)
        ranks = np.zeros(num_solutions, dtype=int)
        domination_counts = np.zeros(num_solutions, dtype=int)
        dominated = [[] for _ in range(num_solutions)]
        front = []
        for i in range(num_solutions):
            for j in range(i + 1, num_solutions):
                if np.all(population_fitness[i] <= population_fitness[j]) and np.any(
                        population_fitness[i] < population_fitness[j]):
                    dominated[i].append(j)
                    domination_counts[j] += 1
                elif np.all(population_fitness[j] <= population_fitness[i]) and np.any(
                        population_fitness[j] < population_fitness[i]):
                    dominated[j].append(i)
                    domination_counts[i] += 1
            if domination_counts[i] == 0:
                ranks[i] = 0
                front.append(i)
        i = 0
        while front:
            next_front = []
            for p in front:
                for q in dominated[p]:
                    domination_counts[q] -= 1
                    if domination_counts[q] == 0:
                        ranks[q] = i + 1
                        next_front.append(q)
            front = next_front
            i += 1
        return ranks

    def non_dominated_sort_by_front(self, population_fitness):
        ranks = self.fast_non_dominated_sort(population_fitness)
        fronts = {}
        for i, r in enumerate(ranks):
            fronts.setdefault(r, []).append(i)
        sorted_fronts = [fronts[r] for r in sorted(fronts.keys())]
        return sorted_fronts

    def weighted_euclidean_distance(self, sol1, sol2):
        vec1 = []
        vec2 = []
        for req in sorted(self.sfc_requests, key=lambda x: x['id']):
            assignment1 = sol1[req['id']]
            assignment2 = sol2[req['id']]
            vec1.extend([float(c) for c in assignment1])
            vec2.extend([float(c) for c in assignment2])
        return np.linalg.norm(np.array(vec1) - np.array(vec2))

    def selection(self):
        pop = self.population.copy()
        pop_fitness = np.array([self.compute_objectives(sol) for sol in pop])
        fronts = self.non_dominated_sort_by_front(pop_fitness)
        data_PR = []
        total = 0
        rank = 0
        while rank < len(fronts) and total + len(fronts[rank]) <= 0.5 * self.population_size:
            for idx in fronts[rank]:
                data_PR.append((pop[idx], "Q1"))
            total += len(fronts[rank])
            rank += 1
        if rank < len(fronts):
            for idx in fronts[rank]:
                if total < 1.5 * self.population_size:
                    data_PR.append((pop[idx], "Q2"))
                    total += 1
                else:
                    break
        L = len(data_PR)
        distance_matrix = np.zeros((L, L))
        for i in range(L):
            for j in range(i + 1, L):
                d = self.weighted_euclidean_distance(data_PR[i][0], data_PR[j][0])
                distance_matrix[i, j] = d
                distance_matrix[j, i] = d
        removed = set()
        while L - len(removed) > self.population_size:
            min_d = float('inf')
            min_i, min_j = -1, -1
            for i in range(L):
                if i in removed:
                    continue
                for j in range(i + 1, L):
                    if j in removed:
                        continue
                    if data_PR[i][1] == "Q1" and data_PR[j][1] == "Q1":
                        continue
                    if distance_matrix[i, j] < min_d:
                        min_d = distance_matrix[i, j]
                        min_i, min_j = i, j
            if data_PR[min_i][1] == "Q2" and data_PR[min_j][1] == "Q2":
                min_dist_i = float('inf')
                min_dist_j = float('inf')
                for k in range(L):
                    if k in removed or k in [min_i, min_j]:
                        continue
                    min_dist_i = min(min_dist_i, distance_matrix[min_i, k])
                    min_dist_j = min(min_dist_j, distance_matrix[min_j, k])
                if min_dist_i < min_dist_j:
                    removed.add(min_i)
                else:
                    removed.add(min_j)
            elif data_PR[min_i][1] == "Q1" and data_PR[min_j][1] == "Q2":
                removed.add(min_j)
            elif data_PR[min_i][1] == "Q2" and data_PR[min_j][1] == "Q1":
                removed.add(min_i)
            else:
                removed.add(min_j)
        new_population = []
        for i in range(L):
            if i not in removed:
                new_population.append(data_PR[i][0])
        return np.array(new_population)

    def crossover(self, parent1, parent2):
        child1 = parent1.copy()
        child2 = parent2.copy()
        req_ids = list(parent1.keys())
        if np.random.rand() < 0.9:
            point = np.random.randint(1, len(req_ids))
            for i in range(point, len(req_ids)):
                rid = req_ids[i]
                child1[rid], child2[rid] = child2[rid], child1[rid]
        for req in self.sfc_requests:
            child1[req['id']] = self.repair_assignment_for_request(req, child1[req['id']])
            child2[req['id']] = self.repair_assignment_for_request(req, child2[req['id']])
        return child1, child2

    def mutation(self, solution):
        sol = solution.copy()
        req = np.random.choice(self.sfc_requests)
        rid = req['id']
        chain = req['chain']
        assignment = sol[rid].copy()
        pos = np.random.randint(0, len(assignment))
        candidates = [node_id for node_id, node in self.network_nodes.items() if chain[pos] in node['vnf_types']]
        if candidates:
            assignment[pos] = np.random.choice(candidates)
        sol[rid] = self.repair_assignment_for_request(req, assignment)
        return sol

    def evolve(self):
        for _ in range(self.generations):
            new_population = []
            selected = self.selection()
            pop_size = len(selected)
            for i in range(0, pop_size, 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % pop_size]
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutation(child1))
                new_population.append(self.mutation(child2))
            self.population = np.array(new_population[:self.population_size])
        pop = self.population.copy()
        pop_fitness = np.array([self.compute_objectives(sol) for sol in pop])
        fronts = self.non_dominated_sort_by_front(pop_fitness)
        pareto_front = [pop[idx] for idx in fronts[0]]
        return np.array(pareto_front)


#############################################
# 主程式設定與輸出
#############################################

if __name__ == "__main__":
    # # 節點資料（修改 load_per_unit 為 load_per_vnf，各 NODE 根據可處理的 VNF 定義不同負載係數）
    # network_nodes = [
    #     {'id': 'A', 'vnf_types': ['0', '1'], 'neighbors': ['B', 'C'],
    #      'load_per_vnf': {'0': 0.5, '1': 0.7},
    #      'processing_delay': {'0': 2, '1': 3}},
    #     {'id': 'B', 'vnf_types': ['0', '2', '3'], 'neighbors': ['A', 'D', 'E'],
    #      'load_per_vnf': {'0': 0.6, '2': 0.8, '3': 1.0},
    #      'processing_delay': {'0': 2.5, '2': 2, '3': 2}},
    #     {'id': 'C', 'vnf_types': ['0', '3', '2'], 'neighbors': ['A', 'D', 'G', 'F'],
    #      'load_per_vnf': {'0': 0.55, '3': 0.65, '2': 0.75},
    #      'processing_delay': {'0': 3, '3': 1.5, '2': 2.5}},
    #     {'id': 'D', 'vnf_types': ['0', '2', '3'], 'neighbors': ['B', 'C', 'E', 'G'],
    #      'load_per_vnf': {'0': 0.6, '2': 0.85, '3': 0.95},
    #      'processing_delay': {'0': 3, '2': 1.8, '3': 2}},
    #     {'id': 'E', 'vnf_types': ['3', '1'], 'neighbors': ['B', 'D', 'H'],
    #      'load_per_vnf': {'3': 0.5, '1': 0.6},
    #      'processing_delay': {'3': 3, '1': 1.8}},
    #     {'id': 'F', 'vnf_types': ['1', '3'], 'neighbors': ['C', 'I', 'J'],
    #      'load_per_vnf': {'1': 0.7, '3': 0.8},
    #      'processing_delay': {'1': 3, '3': 1.8}},
    #     {'id': 'G', 'vnf_types': ['1', '2'], 'neighbors': ['C', 'D', 'I', 'K', 'H'],
    #      'load_per_vnf': {'1': 0.65, '2': 0.75},
    #      'processing_delay': {'1': 3, '2': 1.8}},
    #     {'id': 'H', 'vnf_types': ['0', '2', '3'], 'neighbors': ['E', 'G'],
    #      'load_per_vnf': {'0': 0.55, '2': 0.65, '3': 0.75},
    #      'processing_delay': {'0': 3, '2': 1.8, '3': 2}},
    #     {'id': 'I', 'vnf_types': ['0', '2'], 'neighbors': ['F', 'G', 'K'],
    #      'load_per_vnf': {'0': 0.6, '2': 0.7},
    #      'processing_delay': {'0': 3, '2': 1.8}},
    #     {'id': 'J', 'vnf_types': ['2', '1'], 'neighbors': ['F', 'K'],
    #      'load_per_vnf': {'2': 0.7, '1': 0.8},
    #      'processing_delay': {'2': 3, '1': 1.8}},
    #     {'id': 'K', 'vnf_types': ['1', '3'], 'neighbors': ['G', 'I', 'J'],
    #      'load_per_vnf': {'1': 0.55, '3': 0.65},
    #      'processing_delay': {'1': 1.8, '3': 2}},
    # ]
    #
    # # 邊資訊 (無向邊)
    # edges = {
    #     ('A', 'B'): 100,
    #     ('A', 'C'): 80,
    #     ('C', 'F'): 90,
    #     ('F', 'J'): 70,
    #     ('J', 'K'): 60,
    #     ('K', 'G'): 60,
    #     ('G', 'H'): 70,
    #     ('H', 'E'): 80,
    #     ('E', 'B'): 60,
    #     ('D', 'B'): 40,
    #     ('D', 'C'): 100,
    #     ('D', 'G'): 40,
    #     ('D', 'E'): 70,
    #     ('C', 'G'): 50,
    #     ('I', 'F'): 70,
    #     ('I', 'G'): 80,
    #     ('I', 'K'): 70,
    # }
    #
    # # 定義各 VNF 流量需求
    # vnf_traffic = {
    #     '0': 10,
    #     '1': 10,
    #     '2': 10,
    #     '3': 10,
    # }
    #
    # # 定義 4 個 SFC 請求（請求編號用 "0", "1", "2", "3"）
    # sfc_requests = [
    #     {'id': '0', 'chain': ['0', '1', '2']},
    #     {'id': '1', 'chain': ['2', '3']},
    #     {'id': '2', 'chain': ['1', '3']},
    #     {'id': '3', 'chain': ['0', '3']},
    # ]
    # 確保資料夾存在
    os.makedirs("graph1", exist_ok=True)
    os.makedirs("graph2", exist_ok=True)
    os.makedirs("csv", exist_ok=True)
    for num in range(15, 105, 5):  # 包含15~100，間距5
        c2l = csv2list()
        network_nodes = c2l.nodes(f"../problem/nodes/nodes_{num}.csv")
        edges = c2l.edges(f"../problem/edges/edges_{num}.csv")
        vnf_traffic = c2l.vnfs("../problem/vnfs/vnfs_15.csv")
        sfc_requests = c2l.demands("../problem/demands/demands.csv")

        population_size = 20
        generations = 100

        nsga4_sfc = NSGA4_SFC(network_nodes, edges, sfc_requests, vnf_traffic, population_size, generations)

        start_time = time.time()
        pareto_front = nsga4_sfc.evolve()
        end_time = time.time()
        execution_time = end_time - start_time
        print("程式執行時間：", execution_time, "秒")
        # 建立 graph 用於 BFS (完整路徑)
        # graph = {node_id: node['neighbors'] for node_id, node in {n['id']: n for n in network_nodes}.items()}
        # 輸出格式
        print(f"最佳解 (Pareto Front) 共 {len(pareto_front)} 個：")
        graph = build_graph(network_nodes)
        for sol in pareto_front:
            print("-----")
            print("各請求的處理節點序列與完整路徑：")
            for req in sfc_requests:
                complete_path = get_complete_path(sol[req['id']], graph)
                print(f"請求 {req['id']}：處理節點 = {sol[req['id']]}，完整路徑 = {complete_path}")
        print("-----")

        # 輸出各目標函數結果
        for sol in pareto_front:
            obj_vals = nsga4_sfc.compute_objectives(sol)
            print("\n目標函數結果：")
            print(f"節點負載均衡（標準差）： {obj_vals[0]:.4f}")
            print(f"端到端延遲： {obj_vals[1]:.4f}")
            print(f"網路吞吐量目標： {obj_vals[2]:.4f}")

        # 收集所有解目標數據並以 DataFrame 彙總
        solutions_data = []
        for sol in pareto_front:
            obj_vals = nsga4_sfc.compute_objectives(sol)
            solutions_data.append({
                "Execution_time": str(execution_time),
                'Solution': str(sol),
                'LoadBalance': obj_vals[0],
                'Delay': obj_vals[1],
                'Throughput': obj_vals[2]
            })
        df = pd.DataFrame(solutions_data)
        print("\n各目標函數的彙總數據：")
        print(df)

        # === 3D 散點圖：三個目標 ===
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df['LoadBalance'], df['Delay'], df['Throughput'], c='blue', marker='o')
        ax.set_xlabel('LoadBalance')
        ax.set_ylabel('Delay')
        ax.set_zlabel('Throughput')
        ax.set_title('NSGA4 Pareto Front')
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=30, azim=45)
        plt.savefig(f"graph1/graph1_{num}.png")
        plt.close()
        # plt.show()

        # === 二維散點圖：兩兩目標比較 ===
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # LoadBalance 與 Delay
        axs[0].scatter(df['LoadBalance'], df['Delay'], c='red', marker='o')
        axs[0].set_xlabel('LoadBalance')
        axs[0].set_ylabel('Delay')
        axs[0].set_title('NSGA4 LoadBalance vs Delay')

        # LoadBalance 與 Throughput
        axs[1].scatter(df['LoadBalance'], df['Throughput'], c='green', marker='o')
        axs[1].set_xlabel('LoadBalance')
        axs[1].set_ylabel('Throughput')
        axs[1].set_title('NSGA4 LoadBalance vs Throughput')

        # Delay 與 Throughput
        axs[2].scatter(df['Delay'], df['Throughput'], c='purple', marker='o')
        axs[2].set_xlabel('Delay')
        axs[2].set_ylabel('Throughput')
        axs[2].set_title('NSGA4 Delay vs Throughput')

        plt.tight_layout()
        plt.savefig(f"graph2/graph2_{num}.png")
        plt.close()
        # plt.show()

        df.to_csv(f'csv/NSGA4_solutions_data_{num}.csv', index=False)
