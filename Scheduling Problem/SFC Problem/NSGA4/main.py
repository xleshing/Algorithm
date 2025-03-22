import numpy as np
import pandas as pd
from collections import deque


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
    計算各節點累計負載（流量需求 × node 的 load_per_unit），回傳負載標準差。
    """
    node_loads = {node_id: 0.0 for node_id in network_nodes.keys()}
    for req in sfc_requests:
        chain = req['chain']
        demand = vnf_traffic[chain[0]]
        assignment = solution[req['id']]
        for node_id in assignment:
            node_loads[node_id] += demand * network_nodes[node_id]['load_per_unit']
    loads_array = np.array(list(node_loads.values()))
    return np.std(loads_array)


def objective_end_to_end_delay_bfs(solution, network_nodes, edges, sfc_requests, vnf_traffic):
    """
    目標2：最小化端到端延遲
    對每筆 SFC，計算：
      - 各節點處理延遲（根據 node 對應 VNF 的處理延遲）
      - 邊延遲：對於每對連續處理節點，利用 BFS 找出完整路徑，
        累計沿路每條邊的延遲 (demand / capacity)
    """
    # graph: 每個節點的鄰居關係
    graph = {node_id: network_nodes[node_id]['neighbors'] for node_id in network_nodes}
    total_delay = 0.0
    for req in sfc_requests:
        chain = req['chain']
        demand = vnf_traffic[chain[0]]
        assignment = solution[req['id']]
        node_delay = sum(network_nodes[assignment[i]]['processing_delay'][chain[i]] for i in range(len(assignment)))
        edge_delay = 0.0
        for i in range(len(assignment) - 1):
            path = bfs_shortest_path(graph, assignment[i], assignment[i + 1])
            if path is None:
                edge_delay += 1e6  # 懲罰
            else:
                for j in range(len(path) - 1):
                    n1, n2 = path[j], path[j + 1]
                    if (n1, n2) in edges:
                        cap = edges[(n1, n2)]
                    elif (n2, n1) in edges:
                        cap = edges[(n2, n1)]
                    else:
                        cap = 1e-6
                    edge_delay += demand / cap
        total_delay += (node_delay + edge_delay)
    return total_delay


def objective_network_throughput(solution, edges, sfc_requests, vnf_traffic):
    """
    目標3：最大化網路吞吐量（取倒數以最小化）
    對每條邊，累計經由該邊的流量（利用 BFS 找出完整路徑），
    最後計算各邊利用率的累加後取倒數
    """
    # 建立無向圖
    graph = {}
    for (n1, n2) in edges:
        graph.setdefault(n1, []).append(n2)
        graph.setdefault(n2, []).append(n1)

    edge_flow = {edge: 0.0 for edge in edges.keys()}
    for req in sfc_requests:
        demand = vnf_traffic[req['chain'][0]]
        assignment = solution[req['id']]
        for i in range(len(assignment) - 1):
            path = bfs_shortest_path(graph, assignment[i], assignment[i + 1])
            if path is None:
                continue
            for j in range(len(path) - 1):
                n1, n2 = path[j], path[j + 1]
                if (n1, n2) in edges:
                    edge_flow[(n1, n2)] += demand
                elif (n2, n1) in edges:
                    edge_flow[(n2, n1)] += demand
    throughput_sum = 0.0
    for edge, flow in edge_flow.items():
        throughput_sum += flow / edges[edge]
    epsilon = 1e-6
    return 1 / (throughput_sum + epsilon)


#############################################
# NSGA4_SFC 類別 (完整路徑版本)
#############################################
class NSGA4_SFC:
    def __init__(self, network_nodes, edges, sfc_requests, vnf_traffic, population_size, generations):
        """
        參數說明：
          network_nodes: 節點列表，每個元素包含 'id'、'vnf_types'、'neighbors'、'load_per_unit'、'processing_delay'
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
        對單一 SFC 請求：對於每個 VNF，從所有能處理該 VNF 的節點中隨機選擇（不要求直接連通）
        """
        chain = req['chain']
        assignment = []
        for vnf in chain:
            candidates = [node_id for node_id, node in self.network_nodes.items() if vnf in node['vnf_types']]
            if not candidates:
                raise ValueError(f"請求 {req['id']} 的 VNF {vnf} 無法分配到任何節點")
            assignment.append(np.random.choice(candidates))
        return assignment

    def repair_assignment_for_request(self, req, assignment):
        """
        檢查每個位置是否分配了能處理對應 VNF 的節點
        """
        chain = req['chain']
        for i in range(len(assignment)):
            if chain[i] not in self.network_nodes[assignment[i]]['vnf_types']:
                candidates = [node_id for node_id, node in self.network_nodes.items() if chain[i] in node['vnf_types']]
                if candidates:
                    assignment[i] = np.random.choice(candidates)
                else:
                    assignment[i] = None
        return assignment

    def compute_objectives(self, solution):
        f1 = objective_load_balance(solution, self.network_nodes, self.sfc_requests, self.vnf_traffic)
        f2 = objective_end_to_end_delay_bfs(solution, self.network_nodes, self.edges, self.sfc_requests,
                                            self.vnf_traffic)
        f3 = objective_network_throughput(solution, self.edges, self.sfc_requests, self.vnf_traffic)
        return np.array([f1, f2, f3])

    def objective_details(self, solution):
        details = {}
        # f1 詳細
        node_loads = {node_id: 0.0 for node_id in self.network_nodes.keys()}
        for req in self.sfc_requests:
            demand = self.vnf_traffic[req['chain'][0]]
            assignment = solution[req['id']]
            for node_id in assignment:
                node_loads[node_id] += demand * self.network_nodes[node_id]['load_per_unit']
        details['node_loads'] = node_loads
        details['f1_std'] = np.std(np.array(list(node_loads.values())))

        # f2 詳細：對每筆 SFC，計算節點延遲、各段邊延遲及完整物理路徑
        sfc_details = {}
        total_delay = 0.0
        graph = {node_id: self.network_nodes[node_id]['neighbors'] for node_id in self.network_nodes}
        for req in self.sfc_requests:
            demand = self.vnf_traffic[req['chain'][0]]
            assignment = solution[req['id']]
            node_delays = [self.network_nodes[assignment[i]]['processing_delay'][req['chain'][i]] for i in
                           range(len(assignment))]
            node_delay = sum(node_delays)
            edge_delays = []
            complete_path_segments = []
            for i in range(len(assignment) - 1):
                path = bfs_shortest_path(graph, assignment[i], assignment[i + 1])
                if path is None:
                    ed = 1e6
                    path = [assignment[i], assignment[i + 1]]
                else:
                    ed = 0.0
                    for j in range(len(path) - 1):
                        n1, n2 = path[j], path[j + 1]
                        if (n1, n2) in self.edges:
                            cap = self.edges[(n1, n2)]
                        elif (n2, n1) in self.edges:
                            cap = self.edges[(n2, n1)]
                        else:
                            cap = 1e-6
                        ed += demand / cap
                edge_delays.append(ed)
                complete_path_segments.append(path)
            sfc_total = node_delay + sum(edge_delays)
            # 串接各段完整路徑
            complete_path = []
            for idx, segment in enumerate(complete_path_segments):
                if idx == 0:
                    complete_path.extend(segment)
                else:
                    complete_path.extend(segment[1:])
            sfc_details[req['id']] = {
                'node_delays': node_delays,
                'edge_delays': edge_delays,
                'total_delay': sfc_total,
                'complete_physical_path': complete_path
            }
            total_delay += sfc_total
        details['sfc_delay_details'] = sfc_details
        details['total_delay'] = total_delay
        details['f2'] = total_delay

        # f3 詳細：各邊流量利用率
        graph_edges = {}
        for (n1, n2) in self.edges:
            graph_edges.setdefault(n1, []).append(n2)
            graph_edges.setdefault(n2, []).append(n1)
        edge_flow = {edge: 0.0 for edge in self.edges.keys()}
        for req in self.sfc_requests:
            demand = self.vnf_traffic[req['chain'][0]]
            assignment = solution[req['id']]
            for i in range(len(assignment) - 1):
                path = bfs_shortest_path(graph_edges, assignment[i], assignment[i + 1])
                if path is None:
                    continue
                for j in range(len(path) - 1):
                    n1, n2 = path[j], path[j + 1]
                    if (n1, n2) in self.edges:
                        edge_flow[(n1, n2)] += demand
                    elif (n2, n1) in self.edges:
                        edge_flow[(n2, n1)] += demand
        throughput_sum = 0.0
        edge_utilization = {}
        for edge, flow in edge_flow.items():
            capacity = self.edges[edge]
            utilization = flow / capacity
            edge_utilization[edge] = {'flow': flow, 'capacity': capacity, 'utilization': utilization}
            throughput_sum += utilization
        epsilon = 1e-6
        details['edge_utilization'] = edge_utilization
        details['throughput_sum'] = throughput_sum
        details['f3'] = 1 / (throughput_sum + epsilon)
        return details

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
            vec1.extend([float(ord(c)) for c in assignment1])
            vec2.extend([float(ord(c)) for c in assignment2])
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
                parent2 = selected[(i+1) % pop_size]
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
# 節點資料
network_nodes = [
    {'id': 'A', 'vnf_types': ['0', '1'], 'neighbors': ['B', 'C'], 'load_per_unit': 0.5,
     'processing_delay': {'0': 2, '1': 3}},
    {'id': 'B', 'vnf_types': ['0', '2', "3"], 'neighbors': ['A', 'D', 'E'], 'load_per_unit': 0.6,
     'processing_delay': {'0': 2.5, '2': 2, '3': 2}},
    {'id': 'C', 'vnf_types': ['0', '3', '2'], 'neighbors': ['A', 'D', 'G', "F"], 'load_per_unit': 0.4,
     'processing_delay': {'0': 3, '3': 1.5, '2': 2.5}},
    {'id': 'D', 'vnf_types': ['0', '2', "3"], 'neighbors': ['B', 'C', "E", "G"], 'load_per_unit': 0.7,
     'processing_delay': {'0': 3, '2': 1.8, '3': 2}},
    {'id': 'E', 'vnf_types': ['3', '1'], 'neighbors': ['B', 'D', "H"], 'load_per_unit': 0.3,
     'processing_delay': {'3': 3, '1': 1.8}},
    {'id': 'F', 'vnf_types': ['1', '3'], 'neighbors': ['C', 'I', "J"], 'load_per_unit': 0.4,
     'processing_delay': {'1': 3, '3': 1.8}},
    {'id': 'G', 'vnf_types': ['1', '2'], 'neighbors': ['C', 'D', 'I', 'K', 'H'], 'load_per_unit': 0.8,
     'processing_delay': {'1': 3, '2': 1.8}},
    {'id': 'H', 'vnf_types': ['0', '2', "3"], 'neighbors': ['E', 'G'], 'load_per_unit': 0.1,
     'processing_delay': {'0': 3, '2': 1.8, '3': 2}},
    {'id': 'I', 'vnf_types': ['0', '2'], 'neighbors': ['F', 'G', 'K'], 'load_per_unit': 0.8,
     'processing_delay': {'0': 3, '2': 1.8}},
    {'id': 'J', 'vnf_types': ['2', '1'], 'neighbors': ['F', 'K'], 'load_per_unit': 0.6,
     'processing_delay': {'2': 3, '1': 1.8}},
    {'id': 'K', 'vnf_types': ['1', "3"], 'neighbors': ['G', 'I', 'J'], 'load_per_unit': 0.5,
     'processing_delay': {'1': 1.8, '3': 2}},
]

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
vnf_traffic = {
    '0': 10,
    '1': 10,
    '2': 10,
    '3': 10,
}

# 定義 4 個 SFC 請求（請求編號用 "0", "1", "2", "3"）
sfc_requests = [
    {'id': '0', 'chain': ['0', '1', '2']},
    {'id': '1', 'chain': ['2', '3']},
    {'id': '2', 'chain': ['1', '3']},
    {'id': '3', 'chain': ['0', '3']},
]

population_size = 20
generations = 50

nsga4_sfc = NSGA4_SFC(network_nodes, edges, sfc_requests, vnf_traffic, population_size, generations)
pareto_front = nsga4_sfc.evolve()

# 建立 graph 用於 BFS (完整路徑)
graph = {node_id: node['neighbors'] for node_id, node in {n['id']: n for n in network_nodes}.items()}

# 輸出格式
print(f"最佳解 (Pareto Front) 共 {len(pareto_front)} 個：")
for sol in pareto_front:
    print("-----")
    print("各請求的處理節點序列與完整路徑：")
    for req in sfc_requests:
        assignment = sol[req['id']]
        complete_path = get_complete_path(assignment, graph)
        print(f"請求 {req['id']}：處理節點 = {assignment}，完整路徑 = {complete_path}")
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
        'Solution': str(sol),
        'LoadBalance': obj_vals[0],
        'Delay': obj_vals[1],
        'Throughput': obj_vals[2]
    })
df = pd.DataFrame(solutions_data)
print("\n各目標函數的彙總數據：")
print(df)
