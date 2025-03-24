import numpy as np
import pandas as pd
from collections import deque
import itertools
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#############################################
# BFS 找最短路徑
#############################################
def bfs_shortest_path(graph, start, goal):
    """
    使用 BFS 找出從 start 到 goal 的最短路徑（以節點列表回傳），若無路徑則回傳 None
    """
    if start == goal:
        return [start]
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


#############################################
# 三個目標函數 (針對「多條 SFC」的完整解)
#############################################
def objective_load_balance(solution, network_nodes, sfc_requests, vnf_traffic):
    """
    目標1：最小化節點負載均衡（以節點負載標準差衡量）

    solution 的形式: {
        sfc_id_1: [nodeX, nodeY, ...],
        sfc_id_2: [nodeA, nodeB, ...],
        ...
    }
    """
    node_loads = {node_id: 0.0 for node_id in network_nodes.keys()}
    for req in sfc_requests:
        chain = req['chain']
        assignment = solution[req['id']]  # 該 SFC 的節點指派序列
        # 假設整條 chain 使用 chain[0] 的需求量當作流量(僅示範用)
        demand = vnf_traffic[chain[0]]
        for i, node_id in enumerate(assignment):
            vnf = chain[i]
            load_factor = network_nodes[node_id]['load_per_vnf'][vnf]
            node_loads[node_id] += demand * load_factor
    loads_array = np.array(list(node_loads.values()))
    return np.std(loads_array)


def objective_end_to_end_delay_bfs(solution, network_nodes, edges, sfc_requests, vnf_traffic):
    """
    目標2：最小化端到端延遲
    - 節點處理延遲 + 邊延遲
    """
    graph = {node_id: network_nodes[node_id]['neighbors'] for node_id in network_nodes}
    total_delay = 0.0
    for req in sfc_requests:
        chain = req['chain']
        assignment = solution[req['id']]
        demand = vnf_traffic[chain[0]]
        # 節點延遲
        node_delay = 0.0
        for i, node_id in enumerate(assignment):
            vnf = chain[i]
            node_delay += network_nodes[node_id]['processing_delay'][vnf]
        # 邊延遲
        edge_delay = 0.0
        for i in range(len(assignment) - 1):
            path = bfs_shortest_path(graph, assignment[i], assignment[i + 1])
            if not path:
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
    目標3：最大化網路吞吐量（取倒數 => 越小越好）
    """
    # 建立無向圖
    graph = {}
    for (n1, n2) in edges:
        graph.setdefault(n1, []).append(n2)
        graph.setdefault(n2, []).append(n1)

    edge_flow = {edge: 0.0 for edge in edges.keys()}
    for req in sfc_requests:
        chain = req['chain']
        assignment = solution[req['id']]
        demand = vnf_traffic[chain[0]]
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
        throughput_sum += (flow / edges[edge])
    epsilon = 1e-6
    return 1 / (throughput_sum + epsilon)


#############################################
# 非支配比較
#############################################
def is_dominated(objA, objB):
    """
    A_dom_B = True 代表 A 支配 B (三目標都 <= 且至少一個 <)
    這裡的目標越小越好
    """
    A_dom_B = (all(a <= b for a, b in zip(objA, objB)) and any(a < b for a, b in zip(objA, objB)))
    B_dom_A = (all(b <= a for a, b in zip(objA, objB)) and any(b < a for a, b in zip(objA, objB)))
    return A_dom_B, B_dom_A


def non_dominated_insert(front, new_sol, new_obj):
    dominated_by_others = False
    dominates_list = []
    for (sol, obj) in front:
        A_dom_B, B_dom_A = is_dominated(new_obj, obj)
        if B_dom_A:
            # 代表 front 中的解支配 new_sol => 丟棄
            dominated_by_others = True
            break
        if A_dom_B:
            # new_sol 支配 front 中的 (sol, obj) => 之後要移除
            dominates_list.append((sol, obj))
    if dominated_by_others:
        return front  # 丟棄 new_sol

    # 移除被 new_sol 支配的
    updated = []
    for item in front:
        if item not in dominates_list:
            updated.append(item)
    # 加入 new_sol
    updated.append((new_sol, new_obj))
    return updated


#############################################
# 主要類別：NSGreedy
#############################################
class NSGreedy:
    def __init__(self, network_nodes, edges, sfc_requests, vnf_traffic):
        self.network_nodes = {n['id']: n for n in network_nodes}
        self.edges = edges
        self.sfc_requests = sfc_requests
        self.vnf_traffic = vnf_traffic
        # 做 BFS 用的 graph
        self.bfs_graph = {n['id']: n['neighbors'] for n in network_nodes}

    def compute_objectives(self, solution):
        """
        給定「多 SFC 完整解」(dict: sfc_id -> [node...])，計算三目標
        """
        f1 = objective_load_balance(solution, self.network_nodes, self.sfc_requests, self.vnf_traffic)
        f2 = objective_end_to_end_delay_bfs(solution, self.network_nodes, self.edges, self.sfc_requests,
                                            self.vnf_traffic)
        f3 = objective_network_throughput(solution, self.edges, self.sfc_requests, self.vnf_traffic)
        return (f1, f2, f3)

    def non_dominated_insert(self, front, new_sol, new_obj):
        return non_dominated_insert(front, new_sol, new_obj)

    def evolve_one_sfc(self, req):
        """
        示範：對「單一 SFC」做簡單的 Greedy + 非支配比較
        回傳該 SFC 的一組「完整解」 => [(sol, obj), ...]
        sol = [node1, node2, ...]
        """
        chain = req['chain']
        sfc_id = req['id']

        # 初始
        solution_front = []

        for i, vnf in enumerate(chain):
            if i == 0:
                # 第一個 VNF
                candidates = [nid for nid, nd in self.network_nodes.items() if vnf in nd['vnf_types']]
                if not candidates:
                    # 沒得選 => 無解
                    return []
                new_front = []
                for c in candidates:
                    # 先把 sol 存下來(尚未計算真正目標)
                    # 這裡先用 (sol, (0,0,0)) 佔位
                    sol = [c]
                    placeholder_obj = (0, 0, 0)
                    new_front.append((sol, placeholder_obj))
                solution_front = new_front
            else:
                # 擴展
                next_front = []
                for (sol, pobj) in solution_front:
                    last_node = sol[-1]
                    candidates = [nid for nid, nd in self.network_nodes.items() if vnf in nd['vnf_types']]
                    if not candidates:
                        continue
                    for c in candidates:
                        # 檢查 last_node -> c BFS
                        path = bfs_shortest_path(self.bfs_graph, last_node, c)
                        if path is None:
                            continue
                        new_sol = sol + [c]
                        placeholder_obj = (0, 0, 0)
                        # 加入 next_front 時先做非支配插入
                        next_front = non_dominated_insert(next_front, new_sol, placeholder_obj)
                solution_front = next_front

        # 到這裡表示整個 chain 都分配完 => 重新計算真實三目標
        final_solutions = []
        for (sol, _) in solution_front:
            # 組成只含該 SFC 的 dict => { sfc_id: sol }
            big_sol = {req['id']: sol}
            # 其餘 SFC 不處理 => 空
            for other_req in self.sfc_requests:
                if other_req['id'] != sfc_id:
                    big_sol[other_req['id']] = []
            obj_val = self.compute_objectives(big_sol)
            final_solutions.append((sol, obj_val))

        # 再做一次非支配篩選
        final_nd = []
        for (sol, obj) in final_solutions:
            final_nd = self.non_dominated_insert(final_nd, sol, obj)
        return final_nd

    def combine_all_sfc_solutions(self, all_sfc_solutions):
        """
        針對所有 SFC 的解做「排列組合 (笛卡兒積)」，把每條 SFC 的解拼接成「大解」，計算新目標並做非支配比較。

        all_sfc_solutions: dict[sfc_id] = list of (sol, obj)
           - sol: 針對該 sfc_id 的指派序列 (e.g. [nodeA, nodeB,...])
           - obj: 只是個暫存，最終要重新計算「整合後」的大解的三目標
        回傳: final_nd => 多條 SFC 同時分配的非支配解 [(big_sol_dict, (f1,f2,f3)), ...]
        """
        sfc_ids = list(all_sfc_solutions.keys())
        # 把每個 SFC 的解集合收集成 list
        solutions_list = [all_sfc_solutions[sid] for sid in sfc_ids]

        final_nd = []
        # 用 itertools.product 做排列組合
        for combo in itertools.product(*solutions_list):
            # combo = ( (solA, objA), (solB, objB), ... ) 各自屬於不同 SFC
            big_sol = {}
            for i, sid in enumerate(sfc_ids):
                big_sol[sid] = combo[i][0]  # 取出指派序列
            # 用 compute_objectives 計算「n 個 SFC」同時分配的三目標
            new_obj = self.compute_objectives(big_sol)
            # 再做非支配插入
            final_nd = self.non_dominated_insert(final_nd, big_sol, new_obj)

        return final_nd

    def evolve(self):
        """
        主流程:
         1) 逐條 SFC 呼叫 evolve_one_sfc -> 得到該 SFC 的一批解
         2) 把每條 SFC 的解集合儲存起來
         3) 最後呼叫 combine_all_sfc_solutions, 針對所有 SFC 解做排列組合，計算多條 SFC 同時分配的真實三目標
         4) 用非支配比較獲得最終的 Pareto front
        """
        all_sfc_solutions = {}
        for req in self.sfc_requests:
            nd_set = self.evolve_one_sfc(req)  # list of (sol, obj)
            all_sfc_solutions[req['id']] = nd_set

        # 最後合併 => 多條 SFC 的解 全部做「笛卡兒積」=> 計算真正的多 SFC 同時分配目標
        final_nd = self.combine_all_sfc_solutions(all_sfc_solutions)
        return final_nd


#############################################
# 主程式測試
#############################################
if __name__ == "__main__":
    # 節點
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

    vnf_traffic = {
        '0': 10,
        '1': 10,
        '2': 10,
        '3': 10,
    }

    sfc_requests = [
        {'id': '0', 'chain': ['0', '1', '2']},
        {'id': '1', 'chain': ['2', '3']},
        {'id': '2', 'chain': ['1', '3']},
        {'id': '3', 'chain': ['0', '3']},
    ]

    solver = NSGreedy(network_nodes, edges, sfc_requests, vnf_traffic)
    start_time = time.time()
    final_solutions = solver.evolve()
    end_time = time.time()
    execution_time = end_time - start_time
    print("程式執行時間：", execution_time, "秒")

    print(f"最終多 SFC 非支配解數量: {len(final_solutions)}")
    df_data = []
    for idx, (big_sol, obj) in enumerate(final_solutions, start=1):
        print(f"--- 解 #{idx} ---")
        print("  大解 (對每條 SFC 的指派):")
        for req in sfc_requests:
            print(f"    SFC {req['id']}: {big_sol[req['id']]}")
        print(f"  目標 (LoadBalance, Delay, 1/Throughput) = {obj}")
        df_data.append({
            'Solution': str(big_sol),
            'LoadBalance': obj[0],
            'Delay': obj[1],
            'Throughput': obj[2]
        })

    if df_data:
        df = pd.DataFrame(df_data)
        print(df)
        # 3D Scatter
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df['LoadBalance'], df['Delay'], df['Throughput'], marker='o')
        ax.set_xlabel('LoadBalance')
        ax.set_ylabel('Delay')
        ax.set_zlabel('1 / (Sum Util)')
        ax.set_title('Final Pareto Front (All SFC Combined)')
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=30, azim=45)
        plt.show()

        # 2D Scatter
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        axs[0].scatter(df['LoadBalance'], df['Delay'], marker='o')
        axs[0].set_xlabel('LoadBalance')
        axs[0].set_ylabel('Delay')
        axs[0].set_title('LoadBalance vs Delay')

        axs[1].scatter(df['LoadBalance'], df['Throughput'], marker='o')
        axs[1].set_xlabel('LoadBalance')
        axs[1].set_ylabel('1 / (Sum Util)')
        axs[1].set_title('LoadBalance vs Throughput')

        axs[2].scatter(df['Delay'], df['Throughput'], marker='o')
        axs[2].set_xlabel('Delay')
        axs[2].set_ylabel('1 / (Sum Util)')
        axs[2].set_title('Delay vs Throughput')

        plt.tight_layout()
        plt.show()

        df.to_csv('NSGreedy_solutions_data.csv', index=False)

    else:
        print("No feasible solutions found!")
