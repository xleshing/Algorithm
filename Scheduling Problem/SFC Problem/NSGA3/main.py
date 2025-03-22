import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from collections import deque
from mpl_toolkits.mplot3d import Axes3D


# ----------------------------
# BFS 與輔助函數
# ----------------------------

def bfs_shortest_path(graph, start, goal):
    """
    利用 BFS 在無權重圖中找出從 start 到 goal 的最短路徑，
    回傳路徑（包含起始與終點）的節點列表；若找不到則回傳 None
    """
    visited = set()
    queue = deque([[start]])
    if start == goal:
        return [start]
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node not in visited:
            for neighbor in graph[node]:
                new_path = list(path)
                new_path.append(neighbor)
                if neighbor == goal:
                    return new_path
                queue.append(new_path)
            visited.add(node)
    return None


def build_graph(network_topology):
    """
    根據 network_topology 的鄰居資訊構造圖（字典），
    格式為 {node: [neighbors]}。
    """
    graph = {}
    for node in network_topology:
        graph[node] = network_topology[node]['neighbors']
    return graph


def get_full_path(processing_chain, graph):
    """
    給定某筆 SFC 的處理節點序列（processing_chain），利用 BFS 找出各相鄰節點間的最短路徑，
    並將各路徑串接成完整的 SFC 路徑（注意避免重複計算交界的節點）。
    """
    if len(processing_chain) == 0:
        return []
    full_path = [processing_chain[0]]
    for i in range(1, len(processing_chain)):
        segment = bfs_shortest_path(graph, full_path[-1], processing_chain[i])
        if segment is None:
            # 若無法連通，則直接使用該處理節點（後續目標函數會因懲罰產生較高延遲）
            segment = [processing_chain[i]]
        # 除了第一個節點外，將 segment 串接
        full_path.extend(segment[1:])
    return full_path


# ----------------------------
# 目標函數定義（利用 BFS 取得完整路徑）
# ----------------------------

def objective_node_load_balance(solution, network_topology, edges, vnf_flow, requests):
    """
    目標1：最小化節點負載均衡
      - 對每筆請求，利用 BFS 找出處理節點間的完整路徑，
        對每個路由上經過的節點，其負載 = (該節點對單位流量的負載值 * 流量) 累計
      - 回傳所有節點總負載的標準差
    """
    graph = build_graph(network_topology)
    node_loads = {node: 0.0 for node in network_topology}
    for r, processing_chain in enumerate(solution):
        sfc = requests[r]
        # 假設流量需求取該 SFC 第一個 VNF 的流量需求（各階段一致）
        flow = vnf_flow[sfc[0]]
        full_path = get_full_path(processing_chain, graph)
        for node in full_path:
            node_loads[node] += network_topology[node]['load'] * flow
    loads = np.array(list(node_loads.values()))
    return np.std(loads)


def objective_end_to_end_delay(solution, network_topology, edges, vnf_flow, requests):
    """
    目標2：最小化端到端延遲
      - 每筆請求延遲 = 處理延遲（各處理節點上對應 VNF 的延遲） +
                         路由延遲（完整路徑上，每條邊的流量／容量）
    """
    graph = build_graph(network_topology)
    total_delay = 0.0
    for r, processing_chain in enumerate(solution):
        sfc = requests[r]
        flow = vnf_flow[sfc[0]]
        # 處理延遲：對於每個處理節點，根據該階段 VNF 加總延遲
        proc_delay = 0.0
        for idx, node in enumerate(processing_chain):
            vnf = sfc[idx]
            proc_delay += network_topology[node]['processing_delay'][vnf]
        # 路由延遲：利用 BFS 得到完整路徑後，對每條邊累加延遲 = flow/容量
        full_path = get_full_path(processing_chain, graph)
        route_delay = 0.0
        for i in range(1, len(full_path)):
            n1 = full_path[i - 1]
            n2 = full_path[i]
            if (n1, n2) in edges:
                capacity = edges[(n1, n2)]
            elif (n2, n1) in edges:
                capacity = edges[(n2, n1)]
            else:
                capacity = 1e-6  # 懲罰：若路徑上邊不存在，給予極大延遲
            route_delay += flow / capacity
        total_delay += proc_delay + route_delay
    return total_delay


def objective_throughput(solution, network_topology, edges, vnf_flow, requests):
    """
    目標3：最大化網路吞吐量（取倒數以最小化）
      - 對每筆請求，利用 BFS 得到完整路徑，統計各邊上累計流量，
      - 吞吐量指標 = 1 / (所有邊累計流量/容量之和)
    """
    graph = build_graph(network_topology)
    edge_flow = {edge: 0.0 for edge in edges}
    for r, processing_chain in enumerate(solution):
        sfc = requests[r]
        flow = vnf_flow[sfc[0]]
        full_path = get_full_path(processing_chain, graph)
        for i in range(1, len(full_path)):
            n1 = full_path[i - 1]
            n2 = full_path[i]
            if (n1, n2) in edges:
                edge_flow[(n1, n2)] += flow
            elif (n2, n1) in edges:
                edge_flow[(n2, n1)] += flow
    total_ratio = 0.0
    for edge, f_val in edge_flow.items():
        capacity = edges[edge]
        total_ratio += f_val / capacity
    return 1 / total_ratio if total_ratio > 0 else 1e6


# ----------------------------
# NSGA-III SFC 排程問題類別 (不強制相鄰，透過 BFS 連通)
# ----------------------------

class NSGA3_SFC:
    def __init__(self, network_topology, edges, vnf_flow, requests, population_size, generations, objective_functions,
                 divisions=4):
        """
        初始化 NSGA-III 排程問題：
          network_topology: 節點資訊字典
          edges: 邊及其容量字典
          vnf_flow: VNF 流量需求（字典，key 為 VNF 型號）
          requests: SFC 請求集合，每筆請求為一個 VNF chain（list）
          population_size: 種群規模
          generations: 迭代代數
          objective_functions: 目標函數列表（函數格式：f(solution, network_topology, edges, vnf_flow, requests)）
          divisions: 用於參考點劃分的份數
        """
        self.network_topology = network_topology
        self.edges = edges
        self.vnf_flow = vnf_flow
        self.requests = requests
        self.population_size = population_size
        self.generations = generations
        self.objective_functions = objective_functions
        self.divisions = divisions
        # 產生初始種群：每筆解為各請求的「處理節點序列」(長度與 SFC 中 VNF 數相同)
        self.population = [self.generate_feasible_solution() for _ in range(population_size)]

    def generate_feasible_solution(self):
        """
        產生一個可行解：
          對每筆 SFC 請求，對每個 VNF 隨機選擇一個能支援該 VNF 的節點（不要求相鄰）
        """
        solution = []
        for sfc in self.requests:
            chain = []
            for vnf in sfc:
                candidates = [node for node in self.network_topology if vnf in self.network_topology[node]['vnf_types']]
                if not candidates:
                    raise ValueError(f"無節點支援 VNF {vnf}，請檢查拓樸設定！")
                chain.append(np.random.choice(candidates))
            solution.append(chain)
        return solution

    def generate_chain_for_request(self, sfc):
        """
        針對單一請求，重新生成一個可行的處理節點序列（不要求節點間相鄰）
        """
        chain = []
        for vnf in sfc:
            candidates = [node for node in self.network_topology if vnf in self.network_topology[node]['vnf_types']]
            if not candidates:
                raise ValueError(f"無節點支援 VNF {vnf}，請檢查拓樸設定！")
            chain.append(np.random.choice(candidates))
        return chain

    def compute_objectives(self, solution):
        return np.array([f(solution, self.network_topology, self.edges, self.vnf_flow, self.requests) for f in
                         self.objective_functions])

    def fitness(self, solution):
        return self.compute_objectives(solution)

    # 以下 NSGA-III 相關方法（快速非支配排序、正規化、參考點產生與關聯、niche selection）與先前版本類似

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
        population_fitness = np.array([self.fitness(sol) for sol in self.population])
        fronts = self.non_dominated_sort_by_front(population_fitness)
        new_indices = []
        for front in fronts:
            if len(new_indices) + len(front) <= self.population_size:
                new_indices.extend(front)
            else:
                remaining_slots = self.population_size - len(new_indices)
                normalized_objectives = self.normalize_population(population_fitness)
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

    def crossover(self, parent1, parent2):
        """
        交配：在請求層級上交換部分 SFC 處理節點序列（各請求間獨立）
        """
        child1 = parent1.copy()
        child2 = parent2.copy()
        for i in range(len(parent1)):
            if np.random.rand() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        return child1, child2

    def mutation(self, solution):
        """
        突變：隨機選擇某筆請求，並在其處理節點序列中隨機更換一個節點
          - 新節點需支援該 VNF，且與前後節點之間須存在 BFS 可連通路徑
        """
        sol = [chain.copy() for chain in solution]
        req_idx = np.random.randint(0, len(sol))
        chain = sol[req_idx]
        pos = np.random.randint(0, len(chain))
        graph = build_graph(self.network_topology)
        # 選取候選節點（只要支援該階段 VNF）
        candidates = [node for node in self.network_topology if
                      self.requests[req_idx][pos] in self.network_topology[node]['vnf_types']]
        valid_candidates = []
        for cand in candidates:
            valid = True
            if pos > 0:
                if bfs_shortest_path(graph, chain[pos - 1], cand) is None:
                    valid = False
            if pos < len(chain) - 1:
                if bfs_shortest_path(graph, cand, chain[pos + 1]) is None:
                    valid = False
            if valid:
                valid_candidates.append(cand)
        if valid_candidates:
            sol[req_idx][pos] = np.random.choice(valid_candidates)
        else:
            # 若無合法候選，則重新生成該請求的處理節點序列
            sol[req_idx] = self.generate_chain_for_request(self.requests[req_idx])
        return sol

    def repair_solution(self, solution):
        """
        檢查每筆請求的處理節點序列：
          - 每個節點必須支援對應 VNF
          - 利用 BFS 檢查相鄰處理節點間是否連通
        若不符，則重新生成該請求的處理節點序列
        """
        repaired = []
        graph = build_graph(self.network_topology)
        for i, chain in enumerate(solution):
            sfc = self.requests[i]
            feasible = True
            for pos, node in enumerate(chain):
                if sfc[pos] not in self.network_topology[node]['vnf_types']:
                    feasible = False
                    break
                if pos > 0:
                    if bfs_shortest_path(graph, chain[pos - 1], node) is None:
                        feasible = False
                        break
            if not feasible:
                new_chain = self.generate_chain_for_request(sfc)
                repaired.append(new_chain)
            else:
                repaired.append(chain)
        return repaired

    def evolve(self):
        for _ in range(self.generations):
            new_population = []
            selected = self.selection()
            np.random.shuffle(selected)
            for i in range(0, self.population_size, 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % self.population_size]
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutation(child1))
                new_population.append(self.mutation(child2))
            # 對每個子代進行修正
            self.population = [self.repair_solution(sol) for sol in new_population[:self.population_size]]
        population_fitness = np.array([self.fitness(sol) for sol in self.population])
        fronts = self.non_dominated_sort_by_front(population_fitness)
        best_front = fronts[0]
        return [self.population[i] for i in best_front]


# ----------------------------
# 測試案例與輸出
# ----------------------------

if __name__ == "__main__":
    # 定義網路拓樸：每個節點包含四個參數：
    # 1. vnf_types：該節點能處理的 VNF 類型
    # 2. neighbors：直接連接的節點列表（此處用於 BFS 路徑搜尋）
    # 3. load：單位流量負載值
    # 4. processing_delay：各 VNF 處理延遲（字典）

    network_topology = {
        'A': {'vnf_types': ['0', '1'], 'neighbors': ['B', 'C'], 'load': 0.5,
              'processing_delay': {'0': 2, '1': 3}},
        'B': {'vnf_types': ['0', '2', "3"], 'neighbors': ['A', 'D', 'E'], 'load': 0.6,
              'processing_delay': {'0': 2.5, '2': 2, '3': 2}},
        'C': {'vnf_types': ['0', '3', '2'], 'neighbors': ['A', 'D', 'G', "F"], 'load': 0.4,
              'processing_delay': {'0': 3, '3': 1.5, '2': 2.5}},
        'D': {'vnf_types': ['0', '2', "3"], 'neighbors': ['B', 'C', "E", "G"], 'load': 0.7,
              'processing_delay': {'0': 3, '2': 1.8, '3': 2}},
        'E': {'vnf_types': ['3', '1'], 'neighbors': ['B', 'D', "H"], 'load': 0.3,
              'processing_delay': {'3': 3, '1': 1.8}},
        'F': {'vnf_types': ['1', '3'], 'neighbors': ['C', 'I', "J"], 'load': 0.4,
              'processing_delay': {'1': 3, '3': 1.8}},
        'G': {'vnf_types': ['1', '2'], 'neighbors': ['C', 'D', 'I', 'K', 'H'], 'load': 0.8,
              'processing_delay': {'1': 3, '2': 1.8}},
        'H': {'vnf_types': ['0', '2', "3"], 'neighbors': ['E', 'G'], 'load': 0.1,
              'processing_delay': {'0': 3, '2': 1.8, '3': 2}},
        'I': {'vnf_types': ['0', '2'], 'neighbors': ['F', 'G', 'K'], 'load': 0.8,
              'processing_delay': {'0': 3, '2': 1.8}},
        'J': {'vnf_types': ['2', '1'], 'neighbors': ['F', 'K'], 'load': 0.6,
              'processing_delay': {'2': 3, '1': 1.8}},
        'K': {'vnf_types': ['1', "3"], 'neighbors': ['G', 'I', 'J'], 'load': 0.5,
              'processing_delay': {'1': 1.8, '3': 2}},
    }

    # 邊資訊 (無向邊)
    edges = {
        ('A', 'B'): 100,
        ('A', 'C'): 80,
        ('C', 'F'): 90,
        ('F', 'J'): 70,
        ('J', 'K'): 60,
        ('K', 'G'): 60,
        ('G', 'H'): 70,
        ('H', 'E'): 80,
        ('E', 'B'): 60,
        ('D', 'B'): 40,
        ('D', 'C'): 100,
        ('D', 'G'): 40,
        ('D', 'E'): 70,
        ('C', 'G'): 50,
        ('I', 'F'): 70,
        ('I', 'G'): 80,
        ('I', 'K'): 70,
    }

    # 定義各 VNF 流量需求
    vnf_flow = {
        '0': 10,
        '1': 10,
        '2': 10,
        '3': 10,
    }

    # 定義 4 個 SFC 請求（請求編號用 "0", "1", "2", "3"）
    requests = [
        ['0', '1', '2'],
        ['2', '3'],
        ['1', '3'],
        ['0', '3'],
    ]

    population_size = 20
    generations = 100

    # 三個目標函數依序為：節點負載均衡、端到端延遲、網路吞吐量
    objectives = [objective_node_load_balance, objective_end_to_end_delay, objective_throughput]

    nsga3 = NSGA3_SFC(network_topology, edges, vnf_flow, requests, population_size, generations, objectives)
    best_solutions = nsga3.evolve()

    # 輸出最佳解（Pareto 前沿解）
    print("最佳解 (Pareto Front) 共", len(best_solutions), "個：")
    graph = build_graph(network_topology)
    for sol in best_solutions:
        print("-----")
        print("各請求的處理節點序列與完整路徑：")
        for idx, processing_chain in enumerate(sol):
            full_path = get_full_path(processing_chain, graph)
            print(f"請求 {idx}：處理節點 = {processing_chain}，完整路徑 = {full_path}")

    # 計算並輸出每個最佳解的目標函數值
    for sol in best_solutions:
        obj1 = objective_node_load_balance(sol, network_topology, edges, vnf_flow, requests)
        obj2 = objective_end_to_end_delay(sol, network_topology, edges, vnf_flow, requests)
        obj3 = objective_throughput(sol, network_topology, edges, vnf_flow, requests)
        print("\n目標函數結果：")
        print("節點負載均衡（標準差）：", obj1)
        print("端到端延遲：", obj2)
        print("網路吞吐量目標：", obj3)

    # 彙整各最佳解目標值，並以 DataFrame 呈現
    objectives_data = []
    for sol in best_solutions:
        obj1 = objective_node_load_balance(sol, network_topology, edges, vnf_flow, requests)
        obj2 = objective_end_to_end_delay(sol, network_topology, edges, vnf_flow, requests)
        obj3 = objective_throughput(sol, network_topology, edges, vnf_flow, requests)
        objectives_data.append({
            'Solution': str(sol),
            'LoadBalance': obj1,
            'Delay': obj2,
            'Throughput': obj3
        })
    df = pd.DataFrame(objectives_data)
    print("\n各目標函數的彙總數據：")
    print(df)
    # === 3D 散點圖：三個目標 ===
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['LoadBalance'], df['Delay'], df['Throughput'], c='blue', marker='o')
    ax.set_xlabel('LoadBalance')
    ax.set_ylabel('Delay')
    ax.set_zlabel('Throughput')
    ax.set_title('NSGA3 Pareto Front')
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=30, azim=45)
    plt.show()

    # === 二維散點圖：兩兩目標比較 ===
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # LoadBalance 與 Delay
    axs[0].scatter(df['LoadBalance'], df['Delay'], c='red', marker='o')
    axs[0].set_xlabel('LoadBalance')
    axs[0].set_ylabel('Delay')
    axs[0].set_title('NSGA3 LoadBalance vs Delay')

    # LoadBalance 與 Throughput
    axs[1].scatter(df['LoadBalance'], df['Throughput'], c='green', marker='o')
    axs[1].set_xlabel('LoadBalance')
    axs[1].set_ylabel('Throughput')
    axs[1].set_title('NSGA3 LoadBalance vs Throughput')

    # Delay 與 Throughput
    axs[2].scatter(df['Delay'], df['Throughput'], c='purple', marker='o')
    axs[2].set_xlabel('Delay')
    axs[2].set_ylabel('Throughput')
    axs[2].set_title('NSGA3 Delay vs Throughput')

    plt.tight_layout()
    plt.show()
