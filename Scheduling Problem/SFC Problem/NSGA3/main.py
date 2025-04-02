import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from collections import deque
from csv2list import csv2list
import os


#############################################
# 輔助函數：BFS 與圖構造
#############################################

# def bfs_shortest_path(graph, start, goal):
#     """
#     利用 BFS 在無權重圖中找出從 start 到 goal 的最短路徑，
#     回傳包含起始與終點的節點列表；若找不到則回傳 None
#     """
#     visited = set()
#     queue = deque([[start]])
#     if start == goal:
#         return [start]
#     while queue:
#         path = queue.popleft()
#         node = path[-1]
#         if node not in visited:
#             for neighbor in graph.get(node, []):
#                 new_path = list(path)
#                 new_path.append(neighbor)
#                 if neighbor == goal:
#                     return new_path
#                 queue.append(new_path)
#             visited.add(node)
#     return None

class FindingFailed(Exception):
    """自訂例外，用來表示跳過當前請求"""
    pass


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


# def get_full_path(processing_chain, graph):
#     """
#     給定某筆 SFC 的處理節點序列（processing_chain），
#     利用 BFS 找出相鄰節點間的最短路徑，並將各路徑串接成完整的物理路徑，
#     若遇到無法連通的節點則印出提示並略過。
#     """
#     if len(processing_chain) == 0:
#         return []
#     full_path = [processing_chain[0]]
#     for node in processing_chain[1:]:
#         segment = bfs_shortest_path(graph, full_path[-1], node)
#         if segment is None:
#             print(f"節點 {node} 與 {full_path[-1]} 無連通路，將繞過該節點。")
#             continue
#         full_path.extend(segment[1:])  # 避免重複加入交界節點
#     return full_path


#############################################
# 目標函數
#############################################

# def objective_node_load_balance(solution, network_nodes, edges, vnf_traffic, sfc_requests):
#     """
#     目標1：最小化節點負載均衡
#       - 對每筆請求，僅計算處理節點上的負載：
#           負載 = (該節點對該 VNF 的 load_per_vnf * 流量)
#       - 回傳所有處理節點累計負載的標準差
#     """
#     # 建立節點 id 與其參數的查找字典
#     node_dict = {node['id']: node for node in network_nodes}
#     node_loads = {node['id']: 0.0 for node in network_nodes}
#     for req in sfc_requests:
#         chain = req['chain']
#         demand = vnf_traffic[chain[0]]
#         assignment = solution[req['id']]
#         for i, node_id in enumerate(assignment):
#             load_factor = node_dict[node_id]['load_per_vnf'][chain[i]]
#             node_loads[node_id] += demand * load_factor
#     loads = np.array(list(node_loads.values()))
#     return np.std(loads)


# def objective_end_to_end_delay(solution, network_nodes, edges, vnf_traffic, sfc_requests):
#     """
#     目標2：最小化端到端延遲
#       - 每筆請求延遲 = 處理延遲（各處理節點上對應 VNF 的延遲）
#                         + 路由延遲（處理節點間的完整物理路徑上，各邊 flow/capacity）
#     """
#     graph = build_graph(network_nodes)
#     total_delay = 0.0
#     # 建立查找字典
#     node_dict = {node['id']: node for node in network_nodes}
#     for req in sfc_requests:
#         chain = req['chain']
#         demand = vnf_traffic[chain[0]]
#         assignment = solution[req['id']]
#         proc_delay = 0.0
#         for i, node_id in enumerate(assignment):
#             proc_delay += node_dict[node_id]['processing_delay'][chain[i]]
#         full_path = get_full_path(assignment, graph)
#         route_delay = 0.0
#         for i in range(1, len(full_path)):
#             n1 = full_path[i - 1]
#             n2 = full_path[i]
#             if (n1, n2) in edges:
#                 cap = edges[(n1, n2)]
#             elif (n2, n1) in edges:
#                 cap = edges[(n2, n1)]
#             else:
#                 cap = 1e-6
#             route_delay += demand / cap
#         total_delay += proc_delay + route_delay
#     return total_delay
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


def average_objective_end_to_end_delay_bfs(solution, network_nodes, edges, vnf_traffic, sfc_requests):
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
    return total_delay / len(sfc_requests)


# def objective_throughput(solution, network_nodes, edges, vnf_traffic, sfc_requests):
#     """
#     目標3：最大化網路吞吐量（取倒數以最小化）
#       - 對每筆請求，利用 BFS 得到完整物理路徑，
#         累計各邊上流量與容量的比值後取倒數
#     """
#     graph = build_graph(network_nodes)
#     edge_flow = {edge: 0.0 for edge in edges}
#     for req in sfc_requests:
#         chain = req['chain']
#         demand = vnf_traffic[chain[0]]
#         assignment = solution[req['id']]
#         full_path = get_full_path(assignment, graph)
#         for i in range(1, len(full_path)):
#             n1 = full_path[i - 1]
#             n2 = full_path[i]
#             if (n1, n2) in edges:
#                 edge_flow[(n1, n2)] += demand
#             elif (n2, n1) in edges:
#                 edge_flow[(n2, n1)] += demand
#     total_ratio = 0.0
#     for edge, flow in edge_flow.items():
#         capacity = edges[edge]
#         total_ratio += flow / capacity
#     return 1 / total_ratio if total_ratio > 0 else 1e6


#############################################
# NSGA-III SFC 排程問題類別（輸入結構與 NSGA4 完全一致）
#############################################
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


class NSGA3_SFC:
    def __init__(self, network_nodes, edges, vnf_traffic, sfc_requests, population_size, generations,
                 objective_functions, divisions=4):
        """
        初始化 NSGA-III 排程問題：
          network_nodes: 節點列表，每個元素包含 'id'、'vnf_types'、'neighbors'、'load_per_vnf'、'processing_delay'
          edges: 邊的字典，鍵為 (node1, node2) 的 tuple，值為邊容量
          vnf_traffic: 字典，定義每個 VNF 的流量需求
          sfc_requests: SFC 請求列表，每筆請求為字典，包含 'id' 與 'chain'（VNF 連鎖）
          population_size: 種群規模
          generations: 迭代代數
          objective_functions: 目標函數列表（格式： f(solution, network_nodes, edges, vnf_traffic, sfc_requests) ）
          divisions: 參考點劃分份數
        """
        self.network_nodes = {node['id']: node for node in network_nodes}
        self.edges = edges
        self.sfc_requests = sfc_requests
        self.vnf_traffic = vnf_traffic
        self.population_size = population_size
        self.generations = generations
        self.objective_functions = objective_functions
        self.divisions = divisions
        self.graph = {node_id: self.network_nodes[node_id]['neighbors'] for node_id in self.network_nodes}
        self.population = [self.generate_feasible_solution() for _ in range(population_size)]

    def generate_feasible_solution(self):
        """
        集群初始化
        """
        solution = {}
        for req in list(self.sfc_requests):
            success = False
            for _ in range(500):
                try:
                    solution[req['id']] = self.generate_feasible_assignment_for_request(req)
                    success = True
                    break
                except FindingFailed:
                    # 跳過此請求，繼續處理下一個請求
                    continue
            if not success:
                print(f"Request {req['id']} failed to generate a feasible assignment after 500 tries. Deleting it.")
                self.sfc_requests.remove(req)
        return solution

    def generate_feasible_assignment_for_request(self, req):
        """
        對單筆請求，初始化一個可行的處理節點序列（不要求節點間相鄰）
        """
        chain = req['chain']
        assignment = []
        for i, vnf in enumerate(chain):
            candidates = []
            if i > 0:
                for node_id, node in self.network_nodes.items():
                    if vnf in node['vnf_types'] and bfs_shortest_path(self.graph, assignment[i - 1],
                                                                      node_id) is not None:
                        candidates.append(node_id)
            else:
                for node_id, node in self.network_nodes.items():
                    if vnf in node['vnf_types']:
                        candidates.append(node_id)
            if not candidates:
                raise FindingFailed(f"Request {req['id']} failed to generate a feasible assignment after 500 tries. "
                                    f"Deleting it.")
            else:
                assignment.append(np.random.choice(candidates))

        # # 利用 try/except 捕捉遞迴錯誤
        # try:
        #     assignment = self.repair_assignment_for_request(req, assignment)
        # except RecursionError:
        #     print(f"RecursionError occurred for req {req['id']}, skipping this request.")
        #     # 遇到最大遞迴深度時，拋出自訂例外以便在外層跳過該請求
        #     self.sfc_requests.remove(req)
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

        for i in range(len(assignment)):
            valid = True
            # 檢查當前節點是否具備處理對應 VNF 的能力
            if chain[i] not in self.network_nodes[assignment[i]]['vnf_types']:
                valid = False
            # 如果不是第一個節點，還要檢查與前一個節點之間是否連通
            if i > 0:
                # 使用 BFS 檢查從前一個節點到當前節點是否存在可行路徑
                if bfs_shortest_path(self.graph, assignment[i - 1], assignment[i]) is None:
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
                                  if chain[i] in node['vnf_types'] and bfs_shortest_path(self.graph, assignment[i - 1],
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
        f2 = average_objective_end_to_end_delay_bfs(solution, self.network_nodes, self.edges, self.vnf_traffic,
                                                    self.sfc_requests)
        f3 = objective_network_throughput(solution, self.edges, self.sfc_requests, self.vnf_traffic)
        return np.array([f1, f2, f3])

    # 快速非支配排序與相關方法
    def fast_non_dominated_sort(self, population_fitness):
        num_solutions = len(population_fitness)
        ranks = np.zeros(num_solutions, dtype=int)
        domination_counts = np.zeros(num_solutions, dtype=int)
        dominated_solutions = [[] for _ in range(num_solutions)]
        front = []
        for i in range(num_solutions):
            for j in range(i + 1, num_solutions):
                if np.all(population_fitness[i] <= population_fitness[j]) and np.any(
                        population_fitness[i] < population_fitness[j]):
                    dominated_solutions[i].append(j)
                    domination_counts[j] += 1
                elif np.all(population_fitness[j] <= population_fitness[i]) and np.any(
                        population_fitness[j] < population_fitness[i]):
                    dominated_solutions[j].append(i)
                    domination_counts[i] += 1
            if domination_counts[i] == 0:
                ranks[i] = 0
                front.append(i)
        i = 0
        while front:
            next_front = []
            for p in front:
                for q in dominated_solutions[p]:
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

    def normalize_population(self, fitness_values):
        ideal = np.min(fitness_values, axis=0)
        translated = fitness_values - ideal
        num_objectives = fitness_values.shape[1]
        extreme_points = []
        for i in range(num_objectives):
            weights = np.full(num_objectives, 1e-6)
            weights[i] = 1.0
            asf_values = np.max(translated / weights, axis=1)
            idx = np.argmin(asf_values)
            extreme_points.append(translated[idx])
        extreme_points = np.array(extreme_points)
        try:
            lam = np.linalg.solve(extreme_points.T, np.ones(num_objectives))
            intercepts = 1 / lam
            for i in range(num_objectives):
                if intercepts[i] < np.max(translated[:, i]):
                    intercepts[i] = np.max(translated[:, i])
        except np.linalg.LinAlgError:
            intercepts = np.max(translated, axis=0)
        normalized = translated / intercepts
        return normalized

    def generate_reference_points(self, M, p):
        def recursive_gen(m, left, current):
            if m == 1:
                return [current + [left]]
            points = []
            for i in range(left + 1):
                points.extend(recursive_gen(m - 1, left - i, current + [i]))
            return points

        ref_points = np.array(recursive_gen(M, p, []))
        ref_points = ref_points / p
        return ref_points

    def associate_to_reference_points(self, normalized_values, reference_points):
        norm_ref = reference_points / np.linalg.norm(reference_points, axis=1, keepdims=True)
        assoc_indices = []
        distances = []
        for sol in normalized_values:
            sol_norm = np.linalg.norm(sol)
            if sol_norm == 0:
                assoc_indices.append(0)
                distances.append(np.linalg.norm(sol))
            else:
                dists = []
                for r in norm_ref:
                    proj = np.dot(sol, r) * r
                    d = np.linalg.norm(sol - proj)
                    dists.append(d)
                idx = np.argmin(dists)
                assoc_indices.append(idx)
                distances.append(dists[idx])
        return np.array(assoc_indices), np.array(distances)

    def niche_selection(self, front, assoc, niche_count, remaining_slots):
        candidates = list(front)
        selected = []
        while remaining_slots > 0 and candidates:
            candidate_info = []
            for i, global_idx in enumerate(front):
                if global_idx in candidates:
                    rp = assoc[0][i]
                    d = assoc[1][i]
                    candidate_info.append((global_idx, rp, d))
            if not candidate_info:
                break
            min_count = min([info[1] for info in candidate_info])
            candidate_rps = [info for info in candidate_info if niche_count[info[1]] == min_count]
            selected_candidate = min(candidate_rps, key=lambda x: x[2])
            selected.append(selected_candidate[0])
            candidates.remove(selected_candidate[0])
            niche_count[selected_candidate[1]] += 1
            remaining_slots -= 1
        return selected

    def selection(self):
        pop = self.population.copy()
        pop_fitness = np.array([self.compute_objectives(sol) for sol in self.population])
        fronts = self.non_dominated_sort_by_front(pop_fitness)
        new_indices = []
        for front in fronts:
            if len(new_indices) + len(front) <= self.population_size:
                new_indices.extend(front)
            else:
                remaining_slots = self.population_size - len(new_indices)
                normalized_objectives = self.normalize_population(pop_fitness)
                reference_points = self.generate_reference_points(len(self.objective_functions), self.divisions)
                front_normalized = normalized_objectives[front]
                assoc = self.associate_to_reference_points(front_normalized, reference_points)
                if new_indices:
                    selected_normalized = normalized_objectives[new_indices]
                    selected_assoc = self.associate_to_reference_points(selected_normalized, reference_points)
                    niche_count = {i: 0 for i in range(len(reference_points))}
                    for rp in selected_assoc[0]:
                        niche_count[rp] += 1
                else:
                    niche_count = {i: 0 for i in range(len(reference_points))}
                chosen = self.niche_selection(front, assoc, niche_count, remaining_slots)
                new_indices.extend(chosen)
                break
        new_population = [self.population[i] for i in new_indices]
        return new_population

    # def crossover(self, parent1, parent2):
    #     child1 = parent1.copy()
    #     child2 = parent2.copy()
    #     for req_id in parent1.keys():
    #         if np.random.rand() < 0.5:
    #             child1[req_id], child2[req_id] = child2[req_id], child1[req_id]
    #     return child1, child2
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
            try:
                child1[req['id']] = self.repair_assignment_for_request(req, child1[req['id']])
                child2[req['id']] = self.repair_assignment_for_request(req, child2[req['id']])
            except FindingFailed as e:
                print(f"Crossover failed: {e}")
                child1 = parent1.copy()
                child2 = parent2.copy()
        return child1, child2

    # def mutation(self, solution):
    #     """
    #     突變：隨機選擇某筆請求，並在其處理節點序列中隨機更換一個節點
    #       – 新節點必須支援該 VNF
    #     """
    #     sol = solution.copy()
    #     req = np.random.choice(list(sol.keys()))
    #     chain = sol[req].copy()
    #     # 找出該請求對應的 SFC
    #     sfc = next(item for item in self.sfc_requests if item['id'] == req)
    #     pos = np.random.randint(0, len(chain))
    #     candidates = [node['id'] for node in self.network_nodes if sfc['chain'][pos] in node['vnf_types']]
    #     if candidates:
    #         chain[pos] = np.random.choice(candidates)
    #     else:
    #         chain = self.generate_feasible_assignment_for_request(sfc)
    #     sol[req] = chain
    #     return sol
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
        try:
            sol[rid] = self.repair_assignment_for_request(req, assignment)
        except FindingFailed as e:
            print(f"Mutation failed: {e}")
            sol = solution.copy()
        return sol

    def evolve(self):
        for _ in range(self.generations):
            new_population = []
            selected = self.selection()
            for i in range(0, self.population_size, 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % self.population_size]
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutation(child1))
                new_population.append(self.mutation(child2))
            self.population = new_population[:self.population_size]
        pop = self.population.copy()
        pop_fitness = np.array([self.compute_objectives(sol) for sol in self.population])
        fronts = self.non_dominated_sort_by_front(pop_fitness)
        pareto_front = [pop[idx] for idx in fronts[0]]
        return np.array(pareto_front)


#############################################
# 主程式設定與輸出
#############################################

if __name__ == "__main__":
    # 節點資料（輸入結構與 NSGA4 完全相同）
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
    for i in range(5, 6):
        os.makedirs(f"graph1/data{i}", exist_ok=True)
        os.makedirs(f"graph2/data{i}", exist_ok=True)
        os.makedirs(f"csv/data{i}", exist_ok=True)
        for num in range(15, 105, 5):  # 包含15~100，間距5
            c2l = csv2list()
            network_nodes = c2l.nodes(f"../problem/data{i}/nodes/nodes_{num}.csv")
            edges = c2l.edges(f"../problem/data{i}/edges/edges_{num}.csv")
            vnf_traffic = c2l.vnfs(f"../problem/data{i}/vnfs/vnfs_15.csv")
            sfc_requests = c2l.demands("../problem/demands/demands.csv")

            population_size = 20
            generations = 100

            # 三個目標函數依序為：節點負載均衡、端到端延遲、網路吞吐量
            objectives = [objective_load_balance, average_objective_end_to_end_delay_bfs, objective_network_throughput]

            nsga3_sfc = NSGA3_SFC(network_nodes, edges, vnf_traffic, sfc_requests, population_size, generations,
                                  objectives)

            start_time = time.time()
            best_solutions = nsga3_sfc.evolve()
            end_time = time.time()
            execution_time = end_time - start_time
            print("程式執行時間：", execution_time, "秒")
            # 輸出最佳解（Pareto 前沿解）
            print("最佳解 (Pareto Front) 共", len(best_solutions), "個：")
            for sol in best_solutions:
                print("-----")
                print("各請求的處理節點序列與完整路徑：")
                for req in sfc_requests:
                    full_path = get_complete_path(sol[req['id']], nsga3_sfc.graph)
                    print(f"請求 {req['id']}：處理節點 = {sol[req['id']]}，完整路徑 = {full_path}")
            print("-----")

            # 計算並輸出每個最佳解的目標函數值
            for sol in best_solutions:
                obj_vals = nsga3_sfc.compute_objectives(sol)
                print("\n目標函數結果：")
                print(f"節點負載均衡（標準差）： {obj_vals[0]:.4f}")
                print(f"平均端到端延遲： {obj_vals[1]:.4f}")
                print(f"網路吞吐量目標： {obj_vals[2]:.4f}")

            # 將各最佳解目標值彙整到 DataFrame 中呈現
            objectives_data = []
            for sol in best_solutions:
                obj_vals = nsga3_sfc.compute_objectives(sol)
                objectives_data.append({
                    "Execution_time": str(execution_time),
                    'Solution': str(sol),
                    'LoadBalance': obj_vals[0],
                    'Average Delay': obj_vals[1],
                    'Throughput': obj_vals[2]
                })
            df = pd.DataFrame(objectives_data)
            print("\n各目標函數的彙總數據：")
            print(df)

            # === 3D 散點圖：三個目標 ===
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(df['LoadBalance'], df['Average Delay'], df['Throughput'], c='blue', marker='o')
            ax.set_xlabel('LoadBalance')
            ax.set_ylabel('Average Delay')
            ax.set_zlabel('Throughput')
            ax.set_title('NSGA3 Pareto Front')
            ax.set_box_aspect([1, 1, 1])
            ax.view_init(elev=30, azim=45)
            plt.savefig(f"graph1/data{i}/graph1_{num}.png")
            plt.close()
            # plt.show()

            # === 二維散點圖：兩兩目標比較 ===
            fig, axs = plt.subplots(1, 3, figsize=(18, 5))

            # LoadBalance 與 Average Delay
            axs[0].scatter(df['LoadBalance'], df['Average Delay'], c='red', marker='o')
            axs[0].set_xlabel('LoadBalance')
            axs[0].set_ylabel('Average Delay')
            axs[0].set_title('NSGA3 LoadBalance vs Average Delay')

            # LoadBalance 與 Throughput
            axs[1].scatter(df['LoadBalance'], df['Throughput'], c='green', marker='o')
            axs[1].set_xlabel('LoadBalance')
            axs[1].set_ylabel('Throughput')
            axs[1].set_title('NSGA3 LoadBalance vs Throughput')

            # Average Delay 與 Throughput
            axs[2].scatter(df['Average Delay'], df['Throughput'], c='purple', marker='o')
            axs[2].set_xlabel('Average Delay')
            axs[2].set_ylabel('Throughput')
            axs[2].set_title('NSGA3 Average Delay vs Throughput')

            plt.tight_layout()
            plt.savefig(f"graph2/data{i}/graph2_{num}.png")
            plt.close()
            # plt.show()

            df.to_csv(f'csv/data{i}/NSGA3_solutions_data_{num}.csv', index=False)
