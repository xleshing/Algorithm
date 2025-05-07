import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from collections import deque
from csv2list import csv2list

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

# --------------------------
# NSCO for Knapsack Problem (改為多目標：效益與原狀態改動數)
# --------------------------
class NSCO_Algorithm:
    def __init__(self, network_nodes, edges, sfc_requests, vnf_traffic, coyotes_per_group, coyotes_group, p_leave, generations):
        self.network_nodes = {node['id']: node for node in network_nodes}
        self.edges = edges
        self.sfc_requests = sfc_requests
        self.vnf_traffic = vnf_traffic
        self.coyotes_group = coyotes_group
        self.coyotes_per_group = coyotes_per_group
        self.p_leave = p_leave
        self.generations = generations

        # 初始種群：每筆解為一個字典 { request_id: [node1, node2, ...] }
        self.graph = {node_id: self.network_nodes[node_id]['neighbors'] for node_id in self.network_nodes}
        self.population = np.array([self.generate_feasible_solution() for _ in range(population_size)])

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
                prev_node = assignment[i - 1]
                # 使用 BFS 檢查從前一個節點到當前節點是否存在可行路徑
                if bfs_shortest_path(self.graph, prev_node, assignment[i]) is None:
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

    # --------------------------
    # 非支配排序輔助函式 (均視為 minimization 目標)
    # --------------------------

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

    def nsco_compute_cultural_tendency(self, sub_pop):
        """
        文化傾向：以群內各解的中位數（取 0/1 後四捨五入）作為文化基因
        """
        return np.round(np.median(sub_pop, axis=0)).astype(int)

    def nsco_update_coyote(self, coyote_idx, sub_pop, alpha_coyote, cultural_tendency):
        """
        差分移動更新單隻土狼（解結構：dict<req_id, [node_list]>）：
         1. 隨機取出 qj1, qj2 兩隻「同群土狼索引」
         2. 對每個 SFC 請求，把 alpha、qj1、文化、qj2 的路徑
            轉成二元向量後做差分
         3. 隨機選擇 delta1 或 delta2 作為候選，再反推 node list
         4. 呼叫 repair_assignment_for_request 保證合法
         5. 任一請求修復失敗即退回原解
        """
        # 1. pick two others
        candidates = list(range(self.coyotes_per_group))
        candidates.remove(coyote_idx)
        qj1 = np.random.choice(candidates)
        candidates.remove(qj1)
        qj2 = np.random.choice(candidates)

        new_sol = {}
        for req in self.sfc_requests:
            rid = req['id']

            # --- 2. 轉成二元向量 ---
            # 假設 self.num_nodes 定義了節點總數
            def path_to_vec(path):
                vec = np.zeros(len(self.network_nodes), dtype=int)
                for node in path:
                    vec[int(node)-1] = 1
                return vec

            vec_alpha = path_to_vec(alpha_coyote[rid])
            vec_q1 = path_to_vec(sub_pop[qj1][rid])
            vec_cult = path_to_vec(cultural_tendency[rid])
            vec_q2 = path_to_vec(sub_pop[qj2][rid])

            # 差分
            delta1 = np.abs(vec_alpha - vec_q1)
            delta2 = np.abs(vec_cult - vec_q2)

            # --- 3. 隨機遮罩選 delta1 / delta2 ---
            mask = np.random.rand(len(self.network_nodes)) < 0.5
            cand_vec = np.where(mask, delta1, delta2)

            # 反推成節點清單
            cand_list = [str(i+1) for i, bit in enumerate(cand_vec) if bit > 0]

            # --- 4. 呼叫 repair ---
            try:
                new_sol[rid] = self.repair_assignment_for_request(req, cand_list)
            except FindingFailed:
                # 任何一條修復失敗，整隻土狼退回原始
                return sub_pop[coyote_idx].copy()

        return new_sol

    # def nsco_crossover(self, sub_pop):
    #     """
    #     應用三段式交配公式：
    #       pup_j = father_j   if rnd_j < P_s or j == j1
    #               mother_j   if rnd_j >= P_s + P_a or j == j2
    #               R_j        otherwise
    #     其中 P_s = 1/d, P_a = (1 - P_s)/2
    #     """
    #     d = self.d
    #     # 計算父代貢獻與突變機率
    #     P_s = 1 / d
    #     P_a = (1 - P_s) / 2
    #
    #     # 隨機選兩位父代
    #     parents_idx = np.random.choice(self.coyotes_per_group, 2, replace=False)
    #     father = sub_pop[parents_idx[0], :].copy()
    #     mother = sub_pop[parents_idx[1], :].copy()
    #
    #     # 產生保證繼承位置 j1, j2
    #     perm = np.random.permutation(d)
    #     j1, j2 = perm[0], perm[1]
    #
    #     # 隨機數序列與隨機位點候選解
    #     rnd = np.random.rand(d)
    #     R = np.random.randint(2, size=d)
    #
    #     # 建立子代
    #     pup = np.empty(d, dtype=int)
    #     for j in range(d):
    #         if rnd[j] < P_s or j == j1:
    #             pup[j] = father[j]
    #         elif rnd[j] >= P_s + P_a or j == j2:
    #             pup[j] = mother[j]
    #         else:
    #             pup[j] = R[j]
    #
    #     # 修復為可行解後回傳
    #     return self.repair(pup)
    def nsco_crossover(self, sub_pop):
        """
        使用三段式交配公式在 SFC 賦值格式上做 crossover：
          pup_j = father_j   if rnd_j < P_s or j == j1
                  mother_j   if rnd_j >= P_s + P_a or j == j2
                  R_j        otherwise
        其中 P_s = 1/num_nodes, P_a = (1 - P_s)/2
        最後對每條請求呼叫 repair_assignment_for_request 保證合法。
        """
        N = len(self.network_nodes)
        # 計算父代貢獻與突變機率
        P_s = 1 / N
        P_a = (1 - P_s) / 2

        # 隨機選兩位父代（dict 格式）
        parents_idx = np.random.choice(self.coyotes_per_group, 2, replace=False)
        father = sub_pop[parents_idx[0]]
        mother = sub_pop[parents_idx[1]]

        # 產生保證繼承節點 j1, j2
        perm = np.random.permutation(N)
        j1, j2 = perm[0], perm[1]

        # 隨機數序列與隨機位點候選解
        rnd = np.random.rand(N)
        R = np.random.randint(2, size=N)

        pup = {}
        for req in self.sfc_requests:
            rid = req['id']

            # 將 path list 轉成長度為 N 的二元向量
            def path_to_vec(path):
                vec = np.zeros(N, dtype=int)
                for node in path:
                    vec[int(node)-1] = 1
                return vec

            vec_f = path_to_vec(father[rid])
            vec_m = path_to_vec(mother[rid])

            # 產生子代向量
            child_vec = np.empty(N, dtype=int)
            for j in range(N):
                if rnd[j] < P_s or j == j1:
                    child_vec[j] = vec_f[j]
                elif rnd[j] >= P_s + P_a or j == j2:
                    child_vec[j] = vec_m[j]
                else:
                    child_vec[j] = R[j]

            # 轉回節點清單
            cand_list = [str(i+1) for i, bit in enumerate(child_vec) if bit]

            # 呼叫 repair 函式保證此請求合法
            try:
                pup[rid] = self.repair_assignment_for_request(req, cand_list)
            except FindingFailed:
                # 若任一請求修復失敗，退回父代 1 的完整解
                return father.copy()

        return pup

    # def nsco_update_group(self, population, group_indices, population_age):
    #     """
    #     對一個群內進行更新：
    #      (1) 依多目標評價及非支配排序取得群內 Pareto 前沿，
    #          隨機選取其中一個作為領導者 (alpha) 並計算群文化傾向；
    #      (2) 對每隻個體利用 nsco_update_coyote 嘗試更新，
    #          若新解在多目標上支配原解則予以採納；
    #      (3) 利用 nsco_crossover 產生 pup，
    #          若其在多目標上支配群中部分解，則以年齡輔助替換其中年齡最高者。
    #     """
    #     sub_pop = population[group_indices, :].copy()
    #     sub_objs = np.array([self.multiobj(x) for x in sub_pop])
    #     sub_age = population_age[group_indices].copy()
    #     n_pack = len(group_indices)
    #
    #     fronts = self.fast_non_dominated_sort(sub_objs)
    #     if len(fronts[0]) > 0:
    #         alpha_idx = np.random.choice(fronts[0])
    #         alpha_coyote = sub_pop[alpha_idx, :].copy()
    #     else:
    #         alpha_coyote = sub_pop[0, :].copy()
    #     cultural_tendency = self.nsco_compute_cultural_tendency(sub_pop)
    #
    #     for i in range(n_pack):
    #         new_sol = self.nsco_update_coyote(i, sub_pop, alpha_coyote, cultural_tendency)
    #         new_obj = self.multiobj(new_sol)
    #         if self.dominates(new_obj, sub_objs[i]):
    #             sub_pop[i, :] = new_sol
    #             sub_objs[i] = new_obj
    #             sub_age[i] = 0
    #
    #     if self.d > 1:
    #         pup = self.nsco_crossover(sub_pop)
    #         pup_obj = self.multiobj(pup)
    #         dominated_indices = []
    #         for i in range(n_pack):
    #             if self.dominates(pup_obj, sub_objs[i]):
    #                 dominated_indices.append(i)
    #         if dominated_indices:
    #             ages_candidates = sub_age[dominated_indices]
    #             worst_idx_local = dominated_indices[np.argmax(ages_candidates)]
    #             sub_pop[worst_idx_local, :] = pup
    #             sub_objs[worst_idx_local] = pup_obj
    #             sub_age[worst_idx_local] = 0
    #
    #     population[group_indices, :] = sub_pop
    #     population_age[group_indices] = sub_age
    #     return population, population_age
    def nsco_update_group(self, population, group_indices, population_age):
        """
        更新一個群：
         1) 計算 sub_pop（list of dict）與 sub_objs
         2) 選 alpha_coyote (dict) 與 cultural_tendency (dict)
         3) 對每隻 coyote (dict) 呼叫 nsco_update_coyote，比較 obj，決定是否取代
         4) 用 nsco_crossover 產生 pup (dict)，如果支配某些解，用年齡最老者替換
        """
        # 1. 取出子群
        sub_pop = [population[i] for i in group_indices]
        sub_age = [population_age[i] for i in group_indices]
        # 計算多目標值列表
        sub_objs = [self.compute_objectives(sol) for sol in sub_pop]

        # 2. 非支配排序找 Pareto front
        fronts = self.fast_non_dominated_sort(np.vstack(sub_objs))
        if fronts and fronts[0]:
            alpha_idx = np.random.choice(fronts[0])
        else:
            alpha_idx = 0
        alpha_coyote = sub_pop[alpha_idx]
        cultural_tendency = self.nsco_compute_cultural_tendency(sub_pop)

        n_pack = len(group_indices)
        # 3. 個體更新
        for local_i in range(n_pack):
            orig = sub_pop[local_i]
            new_sol = self.nsco_update_coyote(local_i, sub_pop, alpha_coyote, cultural_tendency)
            new_obj = self.compute_objectives(new_sol)
            # 如果支配原解，接納
            if self.fast_non_dominated_sort(new_obj, sub_objs[local_i]):
                sub_pop[local_i] = new_sol
                sub_objs[local_i] = new_obj
                sub_age[local_i] = 0

        # 4. 交配產生 pup
        if self.d > 1:
            pup = self.nsco_crossover(sub_pop)
            pup_obj = self.compute_objectives(pup)
            # 找出被 pup 支配的子群成員 local 索引
            dominated = [i for i in range(n_pack) if self.fast_non_dominated_sort(pup_obj, sub_objs[i])]
            if dominated:
                # 從被支配者中選擇年齡最大的那位
                ages = [sub_age[i] for i in dominated]
                worst_local = dominated[np.argmax(ages)]
                sub_pop[worst_local] = pup
                sub_objs[worst_local] = pup_obj
                sub_age[worst_local] = 0

        # 5. 寫回主 population 和 population_age
        for local_i, global_i in enumerate(group_indices):
            population[global_i] = sub_pop[local_i]
            population_age[global_i] = sub_age[local_i]

        return population, population_age

    # def nsco_coyote_exchange(self, groups):
    #     """
    #     依機率 p_leave（乘上 coyotes_per_group²）隨機抽取兩個不同群，互換各自一隻解
    #     """
    #     n_groups, coy_per_group = groups.shape
    #     exchange_prob = self.p_leave * (self.coyotes_per_group ** 2)
    #     if n_groups < 2:
    #         return groups
    #     if np.random.rand() < exchange_prob:
    #         g1, g2 = np.random.choice(n_groups, size=2, replace=False)
    #         c1 = np.random.randint(coy_per_group)
    #         c2 = np.random.randint(coy_per_group)
    #         tmp = groups[g1, c1]
    #         groups[g1, c1] = groups[g2, c2]
    #         groups[g2, c2] = tmp
    #     return groups
    def nsco_coyote_exchange(self, population, groups):
        """
        根據交換機率交換兩群的最後一層 Pareto 前緣解：
        - exchange_prob = p_leave * (coyotes_per_group ** 2)
        - 隨機選兩群 g1, g2；取各自子群最後一層 front；若數量不一致，以較多者為主，
          缺失的隨機從前一層補足；然後交換這些位置的索引。
        """
        exchange_prob = self.p_leave * (self.coyotes_per_group ** 2)
        # 機率不符或群數不足時不交換
        if self.n_groups < 2 or np.random.rand() >= exchange_prob:
            return groups

        # 隨機選兩個群
        g1, g2 = np.random.choice(self.n_groups, size=2, replace=False)

        # 使用現有函式計算群內各前緣
        idxs1 = groups[g1]
        pop_objs1 = np.array([self.multiobj(population[idx]) for idx in idxs1])
        fronts1 = self.fast_non_dominated_sort(pop_objs1)

        idxs2 = groups[g2]
        pop_objs2 = np.array([self.multiobj(population[idx]) for idx in idxs2])
        fronts2 = self.fast_non_dominated_sort(pop_objs2)

        last1 = fronts1[-1]
        last2 = fronts2[-1]

        # 確定交換大小
        size = max(len(last1), len(last2))

        # 補足不足的前緣解
        def expand(fronts, last):
            if len(last) == size:
                return last
            prev = fronts[-2] if len(fronts) > 1 else last
            extra = np.random.choice(prev, size - len(last), replace=True)
            return np.concatenate([last, extra])

        pos1 = expand(fronts1, last1)
        pos2 = expand(fronts2, last2)

        # 執行交換
        swap1 = groups[g1][pos1].copy()
        swap2 = groups[g2][pos2].copy()
        groups[g1][pos1] = swap2
        groups[g2][pos2] = swap1

        return groups

    def NSCO_main(self):
        """
        NSCO 的主要演化流程：
         1. 利用 nsco_initialize_population 生成初始族群與群體分配
         2. 每代對各群依 nsco_update_group 更新，並進行群間交換及年齡增長
         3. 每代全族群依多目標評價進行非支配排序，第一前沿作為全域最佳記錄
         4. 最終回傳全域 Pareto 前沿及演化歷程 (archive)
        """
        population, groups, population_age = self.nsco_initialize_population()
        pop_objs = np.array([self.multiobj(x) for x in population])
        fronts = self.fast_non_dominated_sort(pop_objs)
        global_pf = fronts[0]
        global_pf_solutions = population[global_pf, :].copy()
        archive = [global_pf_solutions]

        for iteration in range(self.max_iter):
            for g in range(self.n_groups):
                group_indices = groups[g, :]
                population, population_age = self.nsco_update_group(population, group_indices, population_age)
            groups = self.nsco_coyote_exchange(population, groups)
            population_age += 1

            pop_objs = np.array([self.multiobj(x) for x in population])
            fronts = self.fast_non_dominated_sort(pop_objs)
            global_pf = fronts[0]
            global_pf_solutions = population[global_pf, :].copy()
            archive.append(global_pf_solutions)
            # print(f"Iteration {iteration + 1}: Pareto Front size = {len(global_pf)}")

        # 最終檢查：若全局前沿中無可行解，則以原始狀態作為解
        feasible_front = [sol for sol in global_pf_solutions if self.is_feasible(sol)]
        if len(feasible_front) == 0:
            global_pf_solutions = np.array([self.original_status])
        return global_pf_solutions, archive


# --------------------------
# 測試與示意 (main)
# --------------------------

#############################################
# 主程式設定與輸出
#############################################

if __name__ == "__main__":
    os.makedirs("graph1", exist_ok=True)
    os.makedirs("graph2", exist_ok=True)
    os.makedirs("csv", exist_ok=True)
    for i in range(2, 3):
        os.makedirs(f"graph1/data{i}", exist_ok=True)
        os.makedirs(f"graph2/data{i}", exist_ok=True)
        os.makedirs(f"csv/data{i}", exist_ok=True)
        for num in range(100, 105, 5):  # 包含15~100，間距5
            c2l = csv2list()
            network_nodes = c2l.nodes(f"../problem/data{i}/nodes/nodes_{num}.csv")
            edges = c2l.edges(f"../problem/data{i}/edges/edges_{num}.csv")
            vnf_traffic = c2l.vnfs(f"../problem/data{i}/vnfs/vnfs_{num}.csv")
            sfc_requests = c2l.demands("../problem/demands/demands.csv")

            population_size = 20
            generations = 100

            nsga4_sfc = NSGA4_SFC(network_nodes, edges, sfc_requests, vnf_traffic, population_size, generations)

            start_time = time.time()
            pareto_front, generation_pareto_fronts = nsga4_sfc.evolve()
            end_time = time.time()
            execution_time = end_time - start_time
            print("程式執行時間：", execution_time, "秒")
            print(f"最佳解 (Pareto Front) 共 {len(pareto_front)} 個：")
            for sol in pareto_front:
                print("-----")
                print("各請求的處理節點序列與完整路徑：")
                for req in sfc_requests:
                    complete_path = get_complete_path(sol[req['id']], nsga4_sfc.graph)
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
                    'Average Delay': obj_vals[1],
                    'Throughput': obj_vals[2]
                })
            df = pd.DataFrame(solutions_data)
            print("\n各目標函數的彙總數據：")
            print(df)

            # === 3D 散點圖：三個目標 ===
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(df['LoadBalance'], df['Average Delay'], df['Throughput'], c='blue', marker='o')
            ax.set_xlabel('LoadBalance')
            ax.set_ylabel('Average Delay')
            ax.set_zlabel('Throughput')
            ax.set_title('NSGA4 Pareto Front')
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
            axs[0].set_title('NSGA4 LoadBalance vs Average Delay')

            # LoadBalance 與 Throughput
            axs[1].scatter(df['LoadBalance'], df['Throughput'], c='green', marker='o')
            axs[1].set_xlabel('LoadBalance')
            axs[1].set_ylabel('Throughput')
            axs[1].set_title('NSGA4 LoadBalance vs Throughput')

            # Average Delay 與 Throughput
            axs[2].scatter(df['Average Delay'], df['Throughput'], c='purple', marker='o')
            axs[2].set_xlabel('Average Delay')
            axs[2].set_ylabel('Throughput')
            axs[2].set_title('NSGA4 Average Delay vs Throughput')

            plt.tight_layout()
            plt.savefig(f"graph2/data{i}/graph2_{num}.png")
            plt.close()
            # plt.show()

            df.to_csv(f'csv/data{i}/NSGA4_solutions_data_{num}.csv', index=False)

            generation_objectives_data = []
            for each_sol in range(len(generation_pareto_fronts)):
                obj_data = []
                for sol in generation_pareto_fronts[each_sol]:
                    obj_vals = nsga4_sfc.compute_objectives(sol)
                    obj_data.append({
                        'LoadBalance': obj_vals[0],
                        'Average Delay': obj_vals[1],
                        'Throughput': obj_vals[2]
                    })
                generation_objectives_data.append({
                    "Generation": each_sol,
                    "sol": obj_data
                })
            generation_df = pd.DataFrame(generation_objectives_data)
            print(generation_df)
            generation_df.to_csv(f'csv/NSGA4_generation_solutions_data{i}_{num}.csv', index=False)

