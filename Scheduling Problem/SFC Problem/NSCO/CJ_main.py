import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from collections import deque
from csv2list import csv2list
from collections import Counter


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
def dict2list(population_dict: list) -> list:
    sol_list = []
    for coyote in population_dict:
        for sol in coyote.values():
            for s in sol:
                sol_list.append(s)
    return sol_list


def list2dict(population_list: list, mask: list) -> dict:
    sol_dict = {}
    for m_index in range(len(mask)):
        sol_dict[m_index + 1] = [population_list.pop(0) for _ in range(mask[m_index])]
    return sol_dict


class NSCO_SFC:
    def __init__(self, network_nodes, edges, sfc_requests, vnf_traffic, coyotes_per_group, coyotes_group, p_leave,
                 generations):
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
        # 1) 生成初始族群
        self.population = np.array(
            [self.generate_feasible_solution() for _ in range(self.coyotes_group * self.coyotes_per_group)])
        # 2) 年齡初始化
        self.ages = np.array([0] * len(self.population))
        # 3) 分群：將索引 0..population_size-1 均分成 n_groups
        indices = np.arange(len(self.population))
        splits = np.array_split(indices, self.coyotes_group)
        self.groups = np.vstack([grp for grp in splits])

    def request_shap(self):
        shap = []
        for rs in self.sfc_requests:
            for r in rs.values():
                shap.append(len(r))
        return shap

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
        文化基因：對每個 SFC 請求，在子群（list of dict）中做多數表決 (mode)。
        sub_pop: [ {rid: [node, ...], ...}, {rid: [...], ...}, ... ]
        回傳 dict { rid: [最常見的 node 列表], ... }
        """
        from collections import Counter

        culture = {}
        # 列出所有請求 ID
        all_rids = [req['id'] for req in self.sfc_requests]

        for rid in all_rids:
            # 收集子群中所有解在這個 rid 上的 assignment
            assigns = [sol[rid] for sol in sub_pop]
            # 用 tuple 來做 Counter
            tup_list = [tuple(a) for a in assigns]
            mode_tup, _ = Counter(tup_list).most_common(1)[0]
            # 把最常見的 assignment 放進文化基因
            culture[rid] = list(mode_tup)

        return culture

    def nsco_update_coyote(self, coyote_idx, sub_pop, alpha_coyote, cultural_tendency, generation):
        """
        差分移動 + 動態限額（適用 dict[rid→assignment] 結構）：
          - 每代動態計算跳躍上限
          - 同時從 α 與文化兩條差分路徑各自抽取不超過上限的 rid 更新
          - 其餘 rid 保留現狀
        """
        current = sub_pop[coyote_idx]  # dict[rid → list[node]]
        all_rids = list(current.keys())
        L = len(all_rids)

        # 1) 動態上限：線性衰減
        base_jump_frac, base_culture_frac = 0.3, 0.3
        t, T = generation, max(1, self.generations - 1)
        max_jump = int(np.ceil(base_jump_frac * (1 - t / T) * L))
        max_culture = int(np.ceil(base_culture_frac * (1 - t / T) * L))

        # 2) 選兩隻同群土狼
        idxs = list(range(len(sub_pop)))
        idxs.remove(coyote_idx)
        qj1, qj2 = np.random.choice(idxs, 2, replace=False)
        other1 = sub_pop[qj1]
        other2 = sub_pop[qj2]

        # 3) 計算 diff keys
        diff1 = [rid for rid in all_rids if alpha_coyote[rid] != other1[rid]]
        diff2 = [rid for rid in all_rids if cultural_tendency[rid] != other2[rid]]

        # 4) 抽樣 sel1, sel2
        sel1 = list(np.random.choice(diff1, min(len(diff1), max_jump), replace=False)) if diff1 else []
        available = [rid for rid in all_rids if rid not in sel1]
        diff2_avail = [rid for rid in diff2 if rid in available]
        sel2 = list(
            np.random.choice(diff2_avail, min(len(diff2_avail), max_culture), replace=False)) if diff2_avail else []

        # 5) 生成 new_sol，未動到的 key 自動保留
        new_sol = {rid: list(current[rid]) for rid in all_rids}
        for rid in sel1:
            new_sol[rid] = list(other1[rid])
        for rid in sel2:
            new_sol[rid] = list(other2[rid])

        # 6) 修復並驗證
        for req in self.sfc_requests:
            rid = req['id']
            try:
                new_assign = self.repair_assignment_for_request(req, new_sol[rid])
                new_sol[rid] = new_assign
            except FindingFailed:
                return current.copy()

        return new_sol

    def nsco_crossover(self):
        """
        三段式交配在 dict 結構上：
         - mask 決定從 father/mother/mutation 拷貝哪些 req
        在全局 population 中隨機選兩個父母
        """
        N = len(self.sfc_requests)
        Ps = 1 / N
        Pa = (1 - Ps) / 2
        # 全局隨機選兩個父母索引
        parents = np.random.choice(len(self.population), 2, replace=False)
        parent1 = self.population[parents[0]]
        parent2 = self.population[parents[1]]
        perm = np.random.permutation(N)
        j1, j2 = perm[0], perm[1]
        child = {}
        for idx, req in enumerate(self.sfc_requests):
            rid = req['id']
            r = np.random.rand()
            if r < Ps or idx == j1:
                child[rid] = parent1[rid].copy()
            elif r >= Ps + Pa or idx == j2:
                child[rid] = parent2[rid].copy()
            else:
                try:
                    child[rid] = self.generate_feasible_assignment_for_request(req)
                except FindingFailed:
                    child[rid] = parent1[rid].copy()
        return child

    def nsco_update_group(self, group_indices, generation):
        """
        群內更新：
         1) non-dominated sort -> alpha
         2) update each via nsco_update_coyote
         3) crossover -> pup -> replace worst
        """
        sub_pop = [self.population[i] for i in group_indices]
        sub_age = [self.ages[i] for i in group_indices]
        sub_objs = [self.compute_objectives(sol) for sol in sub_pop]

        fronts = self.non_dominated_sort_by_front(np.vstack(sub_objs))
        alpha_idx = fronts[0][np.random.randint(len(fronts[0]))]
        alpha = sub_pop[alpha_idx]

        culture = self.nsco_compute_cultural_tendency(sub_pop)

        # update coyotes
        for i in range(len(sub_pop)):
            new_sol = self.nsco_update_coyote(i, sub_pop, alpha, culture, generation)
            new_obj = self.compute_objectives(new_sol)
            if np.all(new_obj <= sub_objs[i]) and np.any(new_obj < sub_objs[i]):
                sub_pop[i], sub_objs[i], sub_age[i] = new_sol, new_obj, 0

        # crossover replacement
        if len(sub_pop) > 1:
            pup = self.nsco_crossover()
            pup_obj = self.compute_objectives(pup)
            dominated = [i for i, obj in enumerate(sub_objs) if np.all(pup_obj <= obj) and np.any(pup_obj < obj)]
            if dominated:
                ages = [sub_age[i] for i in dominated]
                worst = dominated[np.argmax(ages)]
                sub_pop[worst], sub_objs[worst], sub_age[worst] = pup, pup_obj, 0

        # 回寫
        for local, gi in enumerate(group_indices):
            self.population[gi] = sub_pop[local]
            self.ages[gi] = sub_age[local]

    def nsco_coyote_exchange(self):
        """
        根據交換機率交換兩群的最後一層 Pareto 前緣解：
        - exchange_prob = p_leave * (group_size ** 2)
        - 隨機選兩群 g1, g2；取各自子群最後一層 front；若數量不一致，以較多者為主，
          缺失的隨機從前一層補足；然後交換這些位置的索引。
        """
        # 取得群數與每群大小
        n_groups = self.groups.shape[0]
        group_size = self.groups.shape[1]
        exchange_prob = self.p_leave * (group_size ** 2)
        # 機率符合或群數足夠時交換
        if n_groups >= 2 or np.random.rand() < exchange_prob:
            # 隨機選兩個群
            g1, g2 = np.random.choice(n_groups, size=2, replace=False)

            # 計算兩群的最後一層 Pareto 前緣索引
            idxs1 = self.groups[g1]
            pop_objs1 = np.array([self.compute_objectives(self.population[idx]) for idx in idxs1])
            fronts1 = self.non_dominated_sort_by_front(pop_objs1)
            last1 = fronts1[-1]

            idxs2 = self.groups[g2]
            pop_objs2 = np.array([self.compute_objectives(self.population[idx]) for idx in idxs2])
            fronts2 = self.non_dominated_sort_by_front(pop_objs2)
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
            swap1 = self.groups[g1][pos1].copy()
            swap2 = self.groups[g2][pos2].copy()
            self.groups[g1][pos1] = swap2
            self.groups[g2][pos2] = swap1

    def evolve(self):
        """
        NSCO 的主要演化流程：
         1. 每代對各群依 nsco_update_group 更新，並進行群間交換及年齡增長
         2. 每代全族群依多目標評價進行非支配排序，第一前沿作為全域最佳記錄
         3. 最終回傳全域 Pareto 前沿及演化歷程 (generation_pareto_fronts)
        """
        # 用來存儲每輪的最佳前緣
        generation_pareto_fronts = []

        for gen in range(self.generations):
            for g in range(self.coyotes_group):
                group_indices = self.groups[g, :]
                self.nsco_update_group(group_indices, gen)
            self.nsco_coyote_exchange()
            self.ages += 1

            # 計算當前人口中每個解的目標值
            pop = self.population.copy()
            pop_fitness = np.array([self.compute_objectives(sol) for sol in self.population])

            # 對當前種群進行非支配排序，獲得最佳前緣
            fronts = self.non_dominated_sort_by_front(pop_fitness)
            current_pareto_front = [pop[idx] for idx in fronts[0]]

            # 保存當前輪的最佳前緣
            generation_pareto_fronts.append(np.array(current_pareto_front))

        pop = self.population.copy()
        pop_fitness = np.array([self.compute_objectives(sol) for sol in pop])
        fronts = self.non_dominated_sort_by_front(pop_fitness)
        pareto_front = [pop[idx] for idx in fronts[0]]
        return np.array(pareto_front), generation_pareto_fronts


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
        for num in range(15, 20, 5):  # 包含15~100，間距5
            c2l = csv2list()
            network_nodes = c2l.nodes(f"../problem/data{i}/nodes/nodes_{num}.csv")
            edges = c2l.edges(f"../problem/data{i}/edges/edges_{num}.csv")
            vnf_traffic = c2l.vnfs(f"../problem/data{i}/vnfs/vnfs_{num}.csv")
            sfc_requests = c2l.demands("../problem/demands/demands.csv")

            coyotes_group = 5
            coyotes_per_group = 4
            p_leave = 0.005
            generations = 100

            nsco_sfc = NSCO_SFC(
                network_nodes,
                edges,
                sfc_requests,
                vnf_traffic,
                coyotes_per_group,
                coyotes_group,
                p_leave,
                generations
            )

            start_time = time.time()
            pareto_front, generation_pareto_fronts = nsco_sfc.evolve()
            end_time = time.time()
            execution_time = end_time - start_time
            print("程式執行時間：", execution_time, "秒")
            print(f"最佳解 (Pareto Front) 共 {len(pareto_front)} 個：")
            for sol in pareto_front:
                print("-----")
                print("各請求的處理節點序列與完整路徑：")
                for req in sfc_requests:
                    complete_path = get_complete_path(sol[req['id']], nsco_sfc.graph)
                    print(f"請求 {req['id']}：處理節點 = {sol[req['id']]}，完整路徑 = {complete_path}")
            print("-----")

            # 輸出各目標函數結果
            for sol in pareto_front:
                obj_vals = nsco_sfc.compute_objectives(sol)
                print("\n目標函數結果：")
                print(f"節點負載均衡（標準差）： {obj_vals[0]:.4f}")
                print(f"端到端延遲： {obj_vals[1]:.4f}")
                print(f"網路吞吐量目標： {obj_vals[2]:.4f}")

            # 收集所有解目標數據並以 DataFrame 彙總
            solutions_data = []
            for sol in pareto_front:
                obj_vals = nsco_sfc.compute_objectives(sol)
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
            ax.set_title('NSCO Pareto Front')
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
            axs[0].set_title('NSCO LoadBalance vs Average Delay')

            # LoadBalance 與 Throughput
            axs[1].scatter(df['LoadBalance'], df['Throughput'], c='green', marker='o')
            axs[1].set_xlabel('LoadBalance')
            axs[1].set_ylabel('Throughput')
            axs[1].set_title('NSCO LoadBalance vs Throughput')

            # Average Delay 與 Throughput
            axs[2].scatter(df['Average Delay'], df['Throughput'], c='purple', marker='o')
            axs[2].set_xlabel('Average Delay')
            axs[2].set_ylabel('Throughput')
            axs[2].set_title('NSCO Average Delay vs Throughput')

            plt.tight_layout()
            plt.savefig(f"graph2/data{i}/graph2_{num}.png")
            plt.close()
            # plt.show()

            df.to_csv(f'csv/data{i}/NSCO_solutions_data_{num}.csv', index=False)

            generation_objectives_data = []
            for each_sol in range(len(generation_pareto_fronts)):
                obj_data = []
                for sol in generation_pareto_fronts[each_sol]:
                    obj_vals = nsco_sfc.compute_objectives(sol)
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
            generation_df.to_csv(f'csv/NSCO_generation_solutions_data{i}_{num}.csv', index=False)
