import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from collections import deque
from csv2list import csv2list

# ----------------------------------------
# 輔助函數與目標函數
# ----------------------------------------
def bfs_shortest_path(graph, start, goal):
    visited = set()
    queue = deque([[start]])
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        if node not in visited:
            visited.add(node)
            for nb in graph.get(node, []):
                queue.append(path + [nb])
    return None


def get_complete_path(assignment, graph):
    complete = []
    for i in range(len(assignment) - 1):
        seg = bfs_shortest_path(graph, assignment[i], assignment[i + 1]) or [assignment[i], assignment[i + 1]]
        complete.extend(seg if i == 0 else seg[1:])
    return complete


def objective_load_balance(solution, network_nodes, sfc_requests, vnf_traffic):
    loads = {n: 0.0 for n in network_nodes}
    for req in sfc_requests:
        chain = req['chain']
        assign = solution[req['id']]
        for i, node in enumerate(assign):
            demand = vnf_traffic[chain[i]]
            loads[str(node)] += demand * network_nodes[str(node)]['load_per_vnf'][str(chain[i])]
    return np.std(list(loads.values()))


def average_objective_end_to_end_delay_bfs(solution, network_nodes, edges, vnf_traffic, sfc_requests):
    graph = {n: network_nodes[n]['neighbors'] for n in network_nodes}
    total = 0.0
    for req in sfc_requests:
        chain = req['chain']
        assign = solution[req['id']]
        node_delay = sum(network_nodes[str(assign[i])]['processing_delay'][chain[i]] for i in range(len(assign)))
        edge_delay = 0.0
        for i in range(len(assign) - 1):
            path = bfs_shortest_path(graph, assign[i], assign[i + 1]) or []
            for j in range(len(path) - 1):
                cap = edges.get((path[j], path[j + 1]), edges.get((path[j + 1], path[j]), 1e-6))
                edge_delay += vnf_traffic[chain[0]] / cap
        total += node_delay + edge_delay
    return total / len(sfc_requests)


def objective_network_throughput(solution, edges, sfc_requests, vnf_traffic):
    graph = {}
    for (n1, n2), cap in edges.items():
        graph.setdefault(n1, []).append(n2)
        graph.setdefault(n2, []).append(n1)
    flow = {e: 0.0 for e in edges}
    for req in sfc_requests:
        assign = solution[req['id']]
        demand = vnf_traffic[req['chain'][0]]
        for i in range(len(assign) - 1):
            path = bfs_shortest_path(graph, assign[i], assign[i + 1]) or []
            for j in range(len(path) - 1):
                e = (path[j], path[j + 1])
                if e in flow:
                    flow[e] += demand
                elif (e[1], e[0]) in flow:
                    flow[(e[1], e[0])] += demand
    for e, f in flow.items():
        if f > edges[e]:
            return float('inf')
    total_flow = sum(flow.values())
    return 1 / (total_flow + 1e-6)


def within_capacity(edge_flow, edges):
    return all(edge_flow[e] <= edges[e] for e in edge_flow)


# ----------------------------------------
# NSCO_SFC 類別
# ----------------------------------------
class NSCO_SFC:
    def __init__(self, network_nodes, edges, sfc_requests, vnf_traffic,
                 coyotes_per_group=5, n_groups=5, p_leave=0.1,
                 max_iter=100, max_delay=100):
        if isinstance(network_nodes, list):
            network_nodes = {node['id']: node for node in network_nodes}
        self.network_nodes = network_nodes
        self.edges = edges
        self.sfc_requests = sfc_requests
        self.vnf_traffic = vnf_traffic
        self.coyotes_per_group = coyotes_per_group
        self.n_groups = n_groups
        self.p_leave = p_leave
        self.max_iter = max_iter
        self.max_delay = max_delay
        self.chain_lens = [len(r['chain']) for r in sfc_requests]
        self.D = sum(self.chain_lens)

    def dominates(self, u, v):
        return np.all(u <= v) and np.any(u < v)

    def fast_non_dominated_sort(self, pop_objs):
        N = len(pop_objs)
        dom_count = np.zeros(N, dtype=int)
        dominated = [[] for _ in range(N)]
        fronts = []
        f1 = []
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if self.dominates(pop_objs[i], pop_objs[j]):
                    dominated[i].append(j)
                elif self.dominates(pop_objs[j], pop_objs[i]):
                    dom_count[i] += 1
            if dom_count[i] == 0:
                f1.append(i)
        fronts.append(f1)
        i = 0
        while fronts[i]:
            next_f = []
            for p in fronts[i]:
                for q in dominated[p]:
                    dom_count[q] -= 1
                    if dom_count[q] == 0:
                        next_f.append(q)
            i += 1
            fronts.append(next_f)
        fronts.pop()
        return fronts

    def encode(self, sol_dict):
        arr = []
        for req in self.sfc_requests:
            arr.extend(sol_dict[req['id']])
        return np.array(arr, dtype=int)

    def decode(self, arr):
        sol = {}
        idx = 0
        for req in self.sfc_requests:
            L = len(req['chain'])
            sol[req['id']] = arr[idx:idx + L].tolist()
            idx += L
        return sol

    def multiobj(self, arr):
        sol = self.decode(arr)
        f1 = objective_load_balance(sol, self.network_nodes, self.sfc_requests, self.vnf_traffic)
        f2 = average_objective_end_to_end_delay_bfs(sol, self.network_nodes, self.edges,
                                                    self.vnf_traffic, self.sfc_requests)
        f3 = objective_network_throughput(sol, self.edges, self.sfc_requests, self.vnf_traffic)
        return np.array([f1, f2, f3])

    def is_feasible(self, arr):
        sol = self.decode(arr)
        graph = {n: self.network_nodes[n]['neighbors'] for n in self.network_nodes}
        flow = {e: 0.0 for e in self.edges}
        for req in self.sfc_requests:
            path = get_complete_path(sol[req['id']], graph)
            dem = self.vnf_traffic[req['chain'][0]]
            for i in range(len(path) - 1):
                e = (path[i], path[i + 1])
                if e in flow:
                    flow[e] += dem
                elif (e[1], e[0]) in flow:
                    flow[(e[1], e[0])] += dem
        return within_capacity(flow, self.edges)

    def repair(self, arr):
        att = 0
        while not self.is_feasible(arr) and att < self.max_delay:
            sol = {}
            for req in self.sfc_requests:
                sol[req['id']] = [
                    np.random.choice([nid for nid, node in self.network_nodes.items()
                                      if vnf in node['vnf_types']])
                    for vnf in req['chain']
                ]
            arr = self.encode(sol)
            att += 1
        return arr

    def nsco_initialize_population(self):
        total = self.n_groups * self.coyotes_per_group
        pop = np.zeros((total, self.D), dtype=int)
        # 对于每个土狼，针对每个请求（链）和链上每个 VNF，从合法节点里随机选
        for idx in range(total):
            arr = []
            for req in self.sfc_requests:
                for vnf in req['chain']:
                    candidates = [
                        nid
                        for nid, node in self.network_nodes.items()
                        if vnf in node['vnf_types']
                    ]
                    arr.append(np.random.choice(candidates))
            pop[idx] = np.array(arr, dtype=int)
        idxs = np.random.permutation(total)
        groups = idxs.reshape(self.n_groups, self.coyotes_per_group)
        ages = np.zeros(total, dtype=int)
        return pop, groups, ages

    def _update_coyote(self, sol, alpha, cult):
        # 隨機突變：從合法節點列表中替換單一基因
        sol_dict = self.decode(sol)
        req = np.random.choice(self.sfc_requests)
        chain = req['chain']
        pos = np.random.randint(len(chain))
        vnf = chain[pos]
        candidates = [nid for nid, node in self.network_nodes.items() if vnf in node['vnf_types']]
        sol_dict[req['id']][pos] = np.random.choice(candidates)
        return self.encode(sol_dict)

    def _crossover(self, sub):
        d = self.D
        P_s = 1 / d
        P_a = (1 - P_s) / 2
        father, mother = sub[np.random.choice(len(sub), 2, False)]
        rnd = np.random.rand(d)
        R = np.random.randint(2, size=d)
        pup = np.empty(d, dtype=int)
        j1, j2 = np.random.choice(d, 2, False)
        for j in range(d):
            if rnd[j] < P_s or j == j1:
                pup[j] = father[j]
            elif rnd[j] >= P_s + P_a or j == j2:
                pup[j] = mother[j]
            else:
                pup[j] = R[j]
        return self.repair(pup)

    def nsco_update_group(self, pop, grp, ages):
        sub = pop[grp]
        objs = np.array([self.multiobj(s) for s in sub])
        fronts = self.fast_non_dominated_sort(objs)
        alpha = sub[np.random.choice(fronts[0])] if fronts[0] else sub[0]
        cult = np.round(np.median(sub, axis=0)).astype(int)
        for li, gi in enumerate(grp):
            new_sol = self._update_coyote(sub[li], alpha, cult)
            if self.dominates(self.multiobj(new_sol), objs[li]):
                pop[gi], ages[gi] = new_sol, 0
        pup = self._crossover(sub)
        pup_obj = self.multiobj(pup)
        for li, gi in enumerate(grp):
            if self.dominates(pup_obj, objs[li]):
                pop[gi], ages[gi] = pup, 0
                break
        return pop, ages

    def nsco_coyote_exchange(self, pop, grps):
        if self.n_groups < 2 or np.random.rand() >= self.p_leave:
            return grps
        g1, g2 = np.random.choice(self.n_groups, 2, False)
        def get_last(g):
            idxs = grps[g]
            objs = np.array([self.multiobj(pop[i]) for i in idxs])
            return self.fast_non_dominated_sort(objs)[-1]
        l1, l2 = get_last(g1), get_last(g2)
        size = max(len(l1), len(l2))
        def expand(front, last):
            if len(last) == size:
                return last
            prev = front[-2] if len(front) > 1 else last
            return np.concatenate([last, np.random.choice(prev, size - len(last), True)])
        f1 = self.fast_non_dominated_sort(np.array([self.multiobj(pop[i]) for i in grps[g1]]))
        f2 = self.fast_non_dominated_sort(np.array([self.multiobj(pop[i]) for i in grps[g2]]))
        p1 = expand(f1, l1)
        p2 = expand(f2, l2)
        swap1 = grps[g1][p1].copy()
        swap2 = grps[g2][p2].copy()
        grps[g1][p1], grps[g2][p2] = swap2, swap1
        return grps

    def run(self):
        pop, grps, ages = self.nsco_initialize_population()
        archive = []
        for it in range(self.max_iter):
            for g in range(self.n_groups):
                pop, ages = self.nsco_update_group(pop, grps[g], ages)
            grps = self.nsco_coyote_exchange(pop, grps)
            ages += 1
            fronts = self.fast_non_dominated_sort(np.array([self.multiobj(sol) for sol in pop]))
            archive.append(pop[fronts[0].copy()])
        final_front = pop[self.fast_non_dominated_sort(np.array([self.multiobj(sol) for sol in pop]))[0]]
        return final_front, archive


# ----------------------------------------
# 主程式：讀取、執行、輸出與視覺化
# ----------------------------------------
if __name__ == "__main__":
    for i in range(2, 3):
        os.makedirs(f"graph1/data{i}", exist_ok=True)
        os.makedirs(f"graph2/data{i}", exist_ok=True)
        os.makedirs(f"csv/data{i}", exist_ok=True)
        for num in range(100, 105, 5):
            c2l = csv2list()
            network_nodes = c2l.nodes(f"../problem/data{i}/nodes/nodes_{num}.csv")
            edges = c2l.edges(f"../problem/data{i}/edges/edges_{num}.csv")
            vnf_traffic = c2l.vnfs(f"../problem/data{i}/vnfs/vnfs_{num}.csv")
            sfc_requests = c2l.demands("../problem/demands/demands.csv")

            nsco = NSCO_SFC(network_nodes, edges, sfc_requests, vnf_traffic,
                            coyotes_per_group=5, n_groups=5,
                            p_leave=0.1, max_iter=50, max_delay=100)

            start = time.time()
            pareto, archive = nsco.run()
            end = time.time()
            print(f"Data{i} Num{num} 執行時間: {end - start:.2f}s, Pareto 數: {len(pareto)}")

            rows = []
            for sol in pareto:
                obj = nsco.multiobj(sol)
                rows.append({"Data": i, "Num": num,
                             "LoadBalance": obj[0],
                             "AvgDelay": obj[1],
                             "Throughput": obj[2]})
            df = pd.DataFrame(rows)
            df.to_csv(f"csv/data{i}/nsco_sfc_results_{i}_{num}.csv", index=False)

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(df['LoadBalance'], df['AvgDelay'], df['Throughput'], marker='o')
            plt.savefig(f"graph1/data{i}/graph1_{num}.png")
            plt.close()

            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            axs[0].scatter(df['LoadBalance'], df['AvgDelay'], marker='o')
            axs[1].scatter(df['LoadBalance'], df['Throughput'], marker='o')
            axs[2].scatter(df['AvgDelay'], df['Throughput'], marker='o')
            plt.tight_layout()
            plt.savefig(f"graph2/data{i}/graph2_{num}.png")
            plt.close()

            it_rows = []
            for it, front in enumerate(archive, start=1):
                for sol in front:
                    obj = nsco.multiobj(sol)
                    it_rows.append({"Data": i, "Num": num, "Iter": it,
                                    "LoadBalance": obj[0],
                                    "AvgDelay": obj[1],
                                    "Throughput": obj[2]})
            pd.DataFrame(it_rows).to_csv(f"csv/data{i}/nsco_sfc_archive_{i}_{num}.csv", index=False)
