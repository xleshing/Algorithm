import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates, scatter_matrix

#############################################
# 獨立定義各目標函數
#############################################
def objective_load_balance(solution, network_nodes, sfc_requests, vnf_traffic):
    """
    目標1：最小化節點負載均衡
    計算各節點總負載（流量需求 * node 的 load_per_unit），並回傳負載標準差。
    """
    node_loads = {node_id: 0.0 for node_id in network_nodes.keys()}
    for req in sfc_requests:
        chain = req['chain']
        demand = vnf_traffic[chain[0]]
        assignment = solution[req['id']]
        for i, node_id in enumerate(assignment):
            node = network_nodes[node_id]
            node_loads[node_id] += demand * node['load_per_unit']
    loads_array = np.array(list(node_loads.values()))
    return np.std(loads_array)

def objective_end_to_end_delay(solution, network_nodes, edges, sfc_requests, vnf_traffic):
    """
    目標2：最小化端到端延遲
    對每筆 SFC，累計：
      - 各節點處理延遲：根據 node 對應 VNF 的處理延遲
      - 邊延遲：定義為 (demand / 邊容量)
    回傳所有請求延遲的累計值。
    """
    total_delay = 0.0
    for req in sfc_requests:
        chain = req['chain']
        demand = vnf_traffic[chain[0]]
        assignment = solution[req['id']]
        node_delay = 0.0
        for i, node_id in enumerate(assignment):
            node_delay += network_nodes[node_id]['processing_delay'][chain[i]]
        edge_delay = 0.0
        for i in range(len(assignment) - 1):
            n1 = assignment[i]
            n2 = assignment[i + 1]
            if (n1, n2) in edges:
                cap = edges[(n1, n2)]
            elif (n2, n1) in edges:
                cap = edges[(n2, n1)]
            else:
                cap = 1e-6  # 懲罰值
            edge_delay += demand / cap
        total_delay += (node_delay + edge_delay)
    return total_delay

def objective_network_throughput(solution, edges, sfc_requests, vnf_traffic):
    """
    目標3：最大化網路吞吐量（取倒數以最小化）
    對每條邊計算 (流量/容量) 累計後取倒數。
    """
    edge_flow = {edge: 0.0 for edge in edges.keys()}
    for req in sfc_requests:
        chain = req['chain']
        demand = vnf_traffic[chain[0]]
        assignment = solution[req['id']]
        for i in range(len(assignment) - 1):
            n1 = assignment[i]
            n2 = assignment[i + 1]
            if (n1, n2) in edges:
                edge_flow[(n1, n2)] += demand
            elif (n2, n1) in edges:
                edge_flow[(n2, n1)] += demand
    throughput_sum = 0.0
    for edge, flow in edge_flow.items():
        capacity = edges[edge]
        throughput_sum += flow / capacity
    epsilon = 1e-6
    return 1 / (throughput_sum + epsilon)

#############################################
# NSGA4_SFC 類別定義
#############################################
class NSGA4_SFC:
    def __init__(self, network_nodes, edges, sfc_requests, vnf_traffic, population_size, generations):
        """
        參數:
          network_nodes: 節點列表，每個元素為字典，包含：
              'id': 節點編號 (str)
              'vnf_types': 該節點能處理的 VNF type（列表）
              'neighbors': 該節點直接連通的鄰居（列表）
              'load_per_unit': 每單位流量負載值 (float)
              'processing_delay': 字典，鍵為 VNF type，值為處理延遲 (float)
          edges: 邊的字典，鍵為 (node1, node2) 的 tuple，值為邊容量
          sfc_requests: 請求列表，每筆請求為字典，包含：
              'id': 請求編號 (str)
              'chain': VNF 連鎖（例如 ['VNF1','VNF2',... ]）
          vnf_traffic: 每個 VNF type 被請求時所需流量 (float)
          population_size: 種群規模
          generations: 迭代代數
        """
        self.network_nodes = {node['id']: node for node in network_nodes}
        self.edges = edges
        self.sfc_requests = sfc_requests
        self.vnf_traffic = vnf_traffic
        self.population_size = population_size
        self.generations = generations
        self.population = np.array([self.generate_feasible_solution() for _ in range(population_size)])

    def generate_feasible_solution(self):
        """針對所有 SFC 請求產生一組可行解"""
        solution = {}
        for req in self.sfc_requests:
            assignment = self.generate_feasible_assignment_for_request(req)
            solution[req['id']] = assignment
        return solution

    def generate_feasible_assignment_for_request(self, req, max_attempts=100):
        """
        產生單一 SFC 請求的可行節點序列：
          - 第一個 VNF 隨機選擇任一能處理該 VNF 的節點
          - 後續 VNF 必須從前一節點的鄰居中選出且能處理該 VNF
        """
        chain = req['chain']
        attempt = 0
        while attempt < max_attempts:
            assignment = []
            candidates = [node_id for node_id, node in self.network_nodes.items() if chain[0] in node['vnf_types']]
            if not candidates:
                raise ValueError(f"請求 {req['id']} 的第一個 VNF {chain[0]} 無法分配到任何節點")
            node_choice = np.random.choice(candidates)
            assignment.append(node_choice)
            feasible = True
            for vnf in chain[1:]:
                prev_node = self.network_nodes[assignment[-1]]
                candidates = [nbr for nbr in prev_node['neighbors'] if vnf in self.network_nodes[nbr]['vnf_types']]
                if not candidates:
                    feasible = False
                    break
                node_choice = np.random.choice(candidates)
                assignment.append(node_choice)
            if feasible:
                return assignment
            attempt += 1
        return self.generate_feasible_assignment_for_request(req)

    def repair_assignment_for_request(self, req, assignment):
        """
        檢查並修正單一請求的節點序列：
          - 檢查相鄰節點連通性與該節點能否處理對應 VNF
          - 如有不符則嘗試修補，或重新產生 assignment
        """
        chain = req['chain']
        for i in range(1, len(assignment)):
            prev_node = self.network_nodes[assignment[i - 1]]
            curr_node = self.network_nodes[assignment[i]]
            if assignment[i] not in prev_node['neighbors'] or chain[i] not in curr_node['vnf_types']:
                candidates = [nbr for nbr in prev_node['neighbors'] if chain[i] in self.network_nodes[nbr]['vnf_types']]
                if candidates:
                    assignment[i] = np.random.choice(candidates)
                else:
                    return self.generate_feasible_assignment_for_request(req)
        return assignment

    def compute_objectives(self, solution):
        """
        利用獨立定義的目標函數計算多目標值，分別為：
          f1: 節點負載均衡（objective_load_balance）
          f2: 端到端延遲（objective_end_to_end_delay）
          f3: 網路吞吐量倒數（objective_network_throughput）
        """
        f1 = objective_load_balance(solution, self.network_nodes, self.sfc_requests, self.vnf_traffic)
        f2 = objective_end_to_end_delay(solution, self.network_nodes, self.edges, self.sfc_requests, self.vnf_traffic)
        f3 = objective_network_throughput(solution, self.edges, self.sfc_requests, self.vnf_traffic)
        return np.array([f1, f2, f3])

    def fast_non_dominated_sort(self, population_fitness):
        num_solutions = len(population_fitness)
        ranks = np.zeros(num_solutions, dtype=int)
        domination_counts = np.zeros(num_solutions, dtype=int)
        dominated = [[] for _ in range(num_solutions)]
        front = []
        for i in range(num_solutions):
            for j in range(i + 1, num_solutions):
                if np.all(population_fitness[i] <= population_fitness[j]) and np.any(population_fitness[i] < population_fitness[j]):
                    dominated[i].append(j)
                    domination_counts[j] += 1
                elif np.all(population_fitness[j] <= population_fitness[i]) and np.any(population_fitness[j] < population_fitness[i]):
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
        # 將每個解展開成向量：依據 sfc_requests 的排序，串接各 SFC 的節點編號（轉為數值）
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
        data_PR = []  # (solution, "Q1"/"Q2")
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
        if pos == 0:
            candidates = [node_id for node_id, node in self.network_nodes.items() if chain[0] in node['vnf_types']]
            if candidates:
                assignment[0] = np.random.choice(candidates)
        else:
            prev_node = self.network_nodes[assignment[pos - 1]]
            candidates = [nbr for nbr in prev_node['neighbors'] if chain[pos] in self.network_nodes[nbr]['vnf_types']]
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

    def objective_details(self, solution):
        """
        回傳各目標的詳細計算結果：
          - f1 詳細：各 node 負載與標準差
          - f2 詳細：每筆 SFC 的節點延遲、邊延遲及總延遲
          - f3 詳細：每條邊的流量、容量、利用率與累計利用率
        """
        details = {}
        # f1 詳細
        node_loads = {node_id: 0.0 for node_id in self.network_nodes.keys()}
        for req in self.sfc_requests:
            chain = req['chain']
            demand = self.vnf_traffic[chain[0]]
            assignment = solution[req['id']]
            for i, node_id in enumerate(assignment):
                node = self.network_nodes[node_id]
                node_loads[node_id] += demand * node['load_per_unit']
        loads_array = np.array(list(node_loads.values()))
        details['node_loads'] = node_loads
        details['f1_std'] = np.std(loads_array)

        # f2 詳細
        sfc_details = {}
        total_delay = 0.0
        for req in self.sfc_requests:
            chain = req['chain']
            demand = self.vnf_traffic[chain[0]]
            assignment = solution[req['id']]
            node_delay = 0.0
            node_delays = []
            for i, node_id in enumerate(assignment):
                d = self.network_nodes[node_id]['processing_delay'][chain[i]]
                node_delays.append(d)
                node_delay += d
            edge_delay = 0.0
            edge_delays = []
            for i in range(len(assignment) - 1):
                n1 = assignment[i]
                n2 = assignment[i + 1]
                if (n1, n2) in self.edges:
                    cap = self.edges[(n1, n2)]
                elif (n2, n1) in self.edges:
                    cap = self.edges[(n2, n1)]
                else:
                    cap = 1e-6
                ed = demand / cap
                edge_delays.append(ed)
                edge_delay += ed
            total = node_delay + edge_delay
            sfc_details[req['id']] = {
                'node_delays': node_delays,
                'edge_delays': edge_delays,
                'total_delay': total
            }
            total_delay += total
        details['sfc_delay_details'] = sfc_details
        details['total_delay'] = total_delay
        details['f2'] = total_delay

        # f3 詳細
        edge_flow = {edge: 0.0 for edge in self.edges.keys()}
        for req in self.sfc_requests:
            chain = req['chain']
            demand = self.vnf_traffic[chain[0]]
            assignment = solution[req['id']]
            for i in range(len(assignment) - 1):
                n1 = assignment[i]
                n2 = assignment[i + 1]
                if (n1, n2) in self.edges:
                    edge_flow[(n1, n2)] += demand
                elif (n2, n1) in self.edges:
                    edge_flow[(n2, n1)] += demand
        edge_utilization = {}
        throughput_sum = 0.0
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

#############################################
# 主程式：設定網路拓樸、邊、VNF 流量需求與 SFC 請求
#############################################
if __name__ == "__main__":
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
    generations = 100

    nsga4_sfc = NSGA4_SFC(network_nodes, edges, sfc_requests, vnf_traffic, population_size, generations)
    pareto_front = nsga4_sfc.evolve()

    # 依據格式輸出最佳解（Pareto Front）
    num_solutions = len(pareto_front)
    print(f"最佳解 (Pareto Front) 共 {num_solutions} 個：")
    for sol in pareto_front:
        print("-----")
        print("各請求的 SFC 指派：")
        # 依據 sfc_requests 的順序，以索引 0,1,2,... 輸出指派
        for idx, req in enumerate(sfc_requests):
            print(f"請求 {idx}: {sol[req['id']]}")
    print("-----\n")

    # 輸出各解的目標函數結果
    print("各目標函數結果：")
    summary_data = []
    for idx, sol in enumerate(pareto_front):
        obj_vals = nsga4_sfc.compute_objectives(sol)
        print(f"解 {idx + 1}:")
        print(f"  節點負載均衡（標準差）： {obj_vals[0]:.6f}")
        print(f"  端到端延遲： {obj_vals[1]:.6f}")
        print(f"  網路吞吐量目標： {obj_vals[2]:.6f}")
        print("-----")
        summary_data.append({
            "Solution": sol,
            "LoadBalance": obj_vals[0],
            "Delay": obj_vals[1],
            "Throughput": obj_vals[2]
        })

    # 彙總結果以 DataFrame 輸出
    df = pd.DataFrame(summary_data)
    # 若 Solution 欄位為物件，可使用 str() 轉換
    df["Solution"] = df["Solution"].apply(lambda sol: str(sol))
    print("\n各目標函數的彙總數據：")
    print(df)

    # === 3D 散點圖：三個目標 ===
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['LoadBalance'], df['Delay'], df['Throughput'], c='blue', marker='o')
    ax.set_xlabel('LoadBalance')
    ax.set_ylabel('Delay')
    ax.set_zlabel('Throughput')
    ax.set_title('NSGA4_direct Pareto Front')
    ax.view_init(elev=30, azim=45)
    plt.show()

    # === 二維散點圖：兩兩目標比較 ===
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # LoadBalance 與 Delay
    axs[0].scatter(df['LoadBalance'], df['Delay'], c='red', marker='o')
    axs[0].set_xlabel('LoadBalance')
    axs[0].set_ylabel('Delay')
    axs[0].set_title('NSGA4_direct LoadBalance vs Delay')

    # LoadBalance 與 Throughput
    axs[1].scatter(df['LoadBalance'], df['Throughput'], c='green', marker='o')
    axs[1].set_xlabel('LoadBalance')
    axs[1].set_ylabel('Throughput')
    axs[1].set_title('NSGA4_direct LoadBalance vs Throughput')

    # Delay 與 Throughput
    axs[2].scatter(df['Delay'], df['Throughput'], c='purple', marker='o')
    axs[2].set_xlabel('Delay')
    axs[2].set_ylabel('Throughput')
    axs[2].set_title('NSGA4_direct Delay vs Throughput')

    plt.tight_layout()
    plt.show()
