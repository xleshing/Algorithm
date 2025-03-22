import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


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
    累計各節點處理 VNF 流量後的負載，回傳所有節點負載的標準差
    """
    node_loads = {node_id: 0.0 for node_id in network_nodes.keys()}
    for req in sfc_requests:
        chain = req['chain']
        assignment = solution[req['id']]
        for i, node_id in enumerate(assignment):
            demand = vnf_traffic[chain[0]]
            load_factor = network_nodes[node_id]['load_per_vnf'][chain[i]]
            node_loads[node_id] += demand * load_factor
    return np.std(np.array(list(node_loads.values())))


def objective_end_to_end_delay_bfs(solution, network_nodes, edges, sfc_requests, vnf_traffic):
    """
    目標2：最小化端到端延遲
    累計各 SFC 中節點處理延遲與沿邊延遲（demand/capacity）的總和
    """
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
                edge_delay += 1e6
            else:
                for j in range(len(path) - 1):
                    n1, n2 = path[j], path[j + 1]
                    cap = edges.get((n1, n2), edges.get((n2, n1), 1e-6))
                    edge_delay += demand / cap
        total_delay += (node_delay + edge_delay)
    return total_delay


def objective_network_throughput(solution, edges, sfc_requests, vnf_traffic):
    """
    目標3：最大化網路吞吐量（取倒數以達到最小化效果）
    累計每條邊的利用率後取倒數
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
    throughput_sum = sum(flow / edges[edge] for edge, flow in edge_flow.items())
    epsilon = 1e-6
    return 1 / (throughput_sum + epsilon)


#############################################
# NSGreedy 類別 (Greedy 版本)
#############################################
def dominates(obj1, obj2):
    """
    判斷 obj1 是否完全支配 obj2（目標值皆較小且至少一項嚴格較小）
    obj1、obj2 均為目標值的 np.array（順序分別為 [f1, f2, f3]，其中目標皆為 minimization）
    """
    return np.all(obj1 <= obj2) and np.any(obj1 < obj2)


class NSGreedy:
    def __init__(self, network_nodes, edges, sfc_requests, vnf_traffic, population_size, generations):
        """
        參數說明：
          network_nodes: 節點列表，每個元素包含 'id'、'vnf_types'、'neighbors'、'load_per_vnf'、'processing_delay'
          edges: 邊的字典
          sfc_requests: SFC 請求列表，每筆請求包含 'id' 與 'chain'
          vnf_traffic: 各 VNF 所需流量
          population_size: 用於產生多筆 Greedy 解（後續排序選取前 10 筆）
          generations: 不使用（保留參數）
        """
        self.network_nodes = {node['id']: node for node in network_nodes}
        self.edges = edges
        self.sfc_requests = sfc_requests
        self.vnf_traffic = vnf_traffic
        self.population_size = population_size
        self.generations = generations
        self.population = None

    def repair_assignment_for_request(self, req, assignment):
        """
        檢查並修正 assignment 中是否分配了能處理對應 VNF 的節點
        """
        chain = req['chain']
        for i in range(len(assignment)):
            if chain[i] not in self.network_nodes[assignment[i]]['vnf_types']:
                candidates = [node_id for node_id, node in self.network_nodes.items() if chain[i] in node['vnf_types']]
                assignment[i] = np.random.choice(candidates) if candidates else None
        return assignment

    def compute_objectives(self, solution):
        f1 = objective_load_balance(solution, self.network_nodes, self.sfc_requests, self.vnf_traffic)
        f2 = objective_end_to_end_delay_bfs(solution, self.network_nodes, self.edges, self.sfc_requests,
                                            self.vnf_traffic)
        f3 = objective_network_throughput(solution, self.edges, self.sfc_requests, self.vnf_traffic)
        return np.array([f1, f2, f3])

    #############################################
    # Greedy 演算法：依序為每個 SFC 選擇處理節點（加入隨機打亂以增添多樣性）
    #############################################
    def greedy_solution(self):
        solution = {}
        bfs_graph = {node_id: self.network_nodes[node_id]['neighbors'] for node_id in self.network_nodes}
        for req in self.sfc_requests:
            chain = req['chain']
            assignment = []
            for i, vnf in enumerate(chain):
                candidates = [node_id for node_id, node in self.network_nodes.items() if vnf in node['vnf_types']]
                # 隨機打亂候選順序
                random.shuffle(candidates)
                best_candidate = None
                best_cost = float('inf')
                for candidate in candidates:
                    cost = self.network_nodes[candidate]['processing_delay'][vnf]
                    if i > 0:
                        prev_node = assignment[i - 1]
                        path = bfs_shortest_path(bfs_graph, prev_node, candidate)
                        if path is None:
                            edge_cost = 1e6
                        else:
                            edge_cost = 0.0
                            demand = self.vnf_traffic[chain[0]]
                            for j in range(len(path) - 1):
                                n1, n2 = path[j], path[j + 1]
                                cap = self.edges.get((n1, n2), self.edges.get((n2, n1), 1e-6))
                                edge_cost += demand / cap
                        cost += edge_cost
                    if cost < best_cost:
                        best_cost = cost
                        best_candidate = candidate
                assignment.append(best_candidate)
            solution[req['id']] = assignment
        return solution

    #############################################
    # Greedy 演算法入口：產生多筆解，排序後取前 10 筆作為 Pareto 前緣
    #############################################
    def evolve(self):
        """
        使用 Greedy 方式產生候選解，再以非支配排序的方式更新解集合：
          - 初始集合 S 用第一個 Greedy 解建立。
          - 每次產生新解 candidate，計算其目標值。
          - 與 S 中所有解比較：
              * 若 candidate 被 S 中任一解支配，則捨棄 candidate。
              * 否則，移除 S 中被 candidate 完全支配的解，並將 candidate 加入 S。
        最後回傳非支配解集合 S（只回傳解，不回傳目標值）。
        """
        S = []  # 存放 (solution, objective_vector) tuple 的集合
        # 初始解
        initial_sol = self.greedy_solution()
        initial_obj = self.compute_objectives(initial_sol)
        S.append((initial_sol, initial_obj))

        # 進行多代更新
        for gen in range(self.generations):
            candidate = self.greedy_solution()
            candidate_obj = self.compute_objectives(candidate)

            # 標記是否應該捨棄 candidate
            discard = False
            dominated_indices = []
            for i, (sol, sol_obj) in enumerate(S):
                if dominates(sol_obj, candidate_obj):
                    # S 中已有解完全支配 candidate，捨棄 candidate
                    discard = True
                    break
                if dominates(candidate_obj, sol_obj):
                    # candidate 完全支配 S 中的某個解，記錄該解索引，稍後移除
                    dominated_indices.append(i)
            if discard:
                continue  # 捨棄 candidate，進入下一代
            else:
                # 若 candidate 完全支配所有 S 中的解，可直接清空 S（或移除被支配的解）
                if len(dominated_indices) == len(S):
                    S = [(candidate, candidate_obj)]
                else:
                    # 移除被 candidate 支配的解
                    S = [entry for i, entry in enumerate(S) if i not in dominated_indices]
                    # 將 candidate 加入 S
                    S.append((candidate, candidate_obj))
        # 最後回傳 S 中的所有解（不包含目標值）
        pareto_solutions = [sol for sol, obj in S]
        return np.array(pareto_solutions)


#############################################
# 主程式設定與輸出
#############################################
if __name__ == "__main__":
    # 節點資料
    network_nodes = [
        {'id': 'A', 'vnf_types': ['0', '1'], 'neighbors': ['B', 'C'],
         'load_per_vnf': {'0': 0.5, '1': 0.7},
         'processing_delay': {'0': 2, '1': 3}},
        {'id': 'B', 'vnf_types': ['0', '2', '3'], 'neighbors': ['A', 'D', 'E'],
         'load_per_vnf': {'0': 0.6, '2': 0.8, '3': 1.0},
         'processing_delay': {'0': 2.5, '2': 2, '3': 2}},
        {'id': 'C', 'vnf_types': ['0', '3', '2'], 'neighbors': ['A', 'D', 'G', 'F'],
         'load_per_vnf': {'0': 0.55, '3': 0.65, '2': 0.75},
         'processing_delay': {'0': 3, '3': 1.5, '2': 2.5}},
        {'id': 'D', 'vnf_types': ['0', '2', '3'], 'neighbors': ['B', 'C', 'E', 'G'],
         'load_per_vnf': {'0': 0.6, '2': 0.85, '3': 0.95},
         'processing_delay': {'0': 3, '2': 1.8, '3': 2}},
        {'id': 'E', 'vnf_types': ['3', '1'], 'neighbors': ['B', 'D', 'H'],
         'load_per_vnf': {'3': 0.5, '1': 0.6},
         'processing_delay': {'3': 3, '1': 1.8}},
        {'id': 'F', 'vnf_types': ['1', '3'], 'neighbors': ['C', 'I', 'J'],
         'load_per_vnf': {'1': 0.7, '3': 0.8},
         'processing_delay': {'1': 3, '3': 1.8}},
        {'id': 'G', 'vnf_types': ['1', '2'], 'neighbors': ['C', 'D', 'I', 'K', 'H'],
         'load_per_vnf': {'1': 0.65, '2': 0.75},
         'processing_delay': {'1': 3, '2': 1.8}},
        {'id': 'H', 'vnf_types': ['0', '2', '3'], 'neighbors': ['E', 'G'],
         'load_per_vnf': {'0': 0.55, '2': 0.65, '3': 0.75},
         'processing_delay': {'0': 3, '2': 1.8, '3': 2}},
        {'id': 'I', 'vnf_types': ['0', '2'], 'neighbors': ['F', 'G', 'K'],
         'load_per_vnf': {'0': 0.6, '2': 0.7},
         'processing_delay': {'0': 3, '2': 1.8}},
        {'id': 'J', 'vnf_types': ['2', '1'], 'neighbors': ['F', 'K'],
         'load_per_vnf': {'2': 0.7, '1': 0.8},
         'processing_delay': {'2': 3, '1': 1.8}},
        {'id': 'K', 'vnf_types': ['1', '3'], 'neighbors': ['G', 'I', 'J'],
         'load_per_vnf': {'1': 0.55, '3': 0.65},
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

    # 定義 4 個 SFC 請求
    sfc_requests = [
        {'id': '0', 'chain': ['0', '1', '2']},
        {'id': '1', 'chain': ['2', '3']},
        {'id': '2', 'chain': ['1', '3']},
        {'id': '3', 'chain': ['0', '3']},
    ]

    population_size = 100  # 用來產生多筆解的次數
    generations = 100      # 此版本未使用

    ns_greedy = NSGreedy(network_nodes, edges, sfc_requests, vnf_traffic, population_size, generations)
    pareto_front = ns_greedy.evolve()

    # 建立 graph 用於 BFS (完整路徑)
    graph = {node_id: node['neighbors'] for node_id, node in {n['id']: n for n in network_nodes}.items()}

    print(f"最佳解 (Pareto Front) 共 {len(pareto_front)} 筆：")
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
        obj_vals = ns_greedy.compute_objectives(sol)
        print("\n目標函數結果：")
        print(f"節點負載均衡（標準差）： {obj_vals[0]:.4f}")
        print(f"端到端延遲： {obj_vals[1]:.4f}")
        print(f"網路吞吐量目標： {obj_vals[2]:.4f}")

    # 收集所有解目標數據並以 DataFrame 彙總
    solutions_data = []
    for sol in pareto_front:
        obj_vals = ns_greedy.compute_objectives(sol)
        solutions_data.append({
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
    ax.set_title('Greedy Pareto Front')
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=30, azim=45)
    plt.show()

    # === 二維散點圖：兩兩目標比較 ===
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].scatter(df['LoadBalance'], df['Delay'], c='red', marker='o')
    axs[0].set_xlabel('LoadBalance')
    axs[0].set_ylabel('Delay')
    axs[0].set_title('LoadBalance vs Delay')
    axs[1].scatter(df['LoadBalance'], df['Throughput'], c='green', marker='o')
    axs[1].set_xlabel('LoadBalance')
    axs[1].set_ylabel('Throughput')
    axs[1].set_title('LoadBalance vs Throughput')
    axs[2].scatter(df['Delay'], df['Throughput'], c='purple', marker='o')
    axs[2].set_xlabel('Delay')
    axs[2].set_ylabel('Throughput')
    axs[2].set_title('Delay vs Throughput')
    plt.tight_layout()
    plt.show()

    df.to_csv('NSGreedy_solutions_data.csv', index=False)
