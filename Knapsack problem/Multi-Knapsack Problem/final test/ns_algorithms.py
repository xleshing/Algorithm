import numpy as np
from collections import Counter

class NSCO_LoadBalancing:
    """
    NSCO 版本（可與 NSGA4_LoadBalancing 對照比較）
    - 解: 長度 num_requests 的整數向量，每一個請求分配到 [0..num_servers-1] 的某台伺服器
    - 目標: 以 objective_functions(solution, usage, item_values, server_capacities) 列表計算
    - evolve() 回傳 (pareto_front, generation_pareto_fronts) 與 NSGA4 形狀相同
    """
    def __init__(self,
                 num_servers,
                 num_requests,
                 population_size,
                 generations,
                 objective_functions,
                 item_values,
                 server_capacities,
                 # NSCO 族群/群體參數
                 coyotes_per_group=5,
                 n_groups=5,
                 p_leave=0.1,
                 max_delay=200):
        self.num_servers = num_servers
        self.num_requests = num_requests
        self.population_size = population_size
        self.generations = generations
        self.objective_functions = objective_functions
        self.item_values = np.array(item_values)       # (num_requests, m) 通常 m=1
        self.server_capacities = np.array(server_capacities)  # (num_servers, m)
        self.max_delay = max_delay

        # NSCO 參數
        self.coyotes_per_group = coyotes_per_group
        self.n_groups = n_groups
        self.p_leave = p_leave

        # 初始族群
        self.population = np.array([self.generate_feasible_solution() for _ in range(population_size)])

    # ---------- 基礎工具（與 NSGA-4 對齊） ----------
    def generate_feasible_solution(self):
        """隨機指派請求到伺服器，並以貪婪修復容量違反。"""
        sol = np.random.randint(0, self.num_servers, size=self.num_requests, dtype=int)
        return self.repair_solution(sol)

    def compute_objective_usage(self, solution):
        """計算各伺服器已使用量 (num_servers, m)。"""
        m = self.item_values.shape[1]
        usage = np.zeros((self.num_servers, m), dtype=float)
        for j in range(self.num_servers):
            idx = (solution == j)
            if np.any(idx):
                usage[j] = np.sum(self.item_values[idx], axis=0)
        return usage

    def fitness(self, solution):
        usage = self.compute_objective_usage(solution)
        return np.array([f(solution, usage, self.item_values, self.server_capacities)
                         for f in self.objective_functions], dtype=float)

    def dominates(self, u, v):
        """全為最小化目標：u 支配 v 若 u<=v 且 有嚴格 <。"""
        return np.all(u <= v) and np.any(u < v)

    def fast_non_dominated_sort(self, pop_fitness):
        N = len(pop_fitness)
        domination_counts = np.zeros(N, dtype=int)
        dominated = [[] for _ in range(N)]
        fronts = []
        f1 = []
        for i in range(N):
            for j in range(i + 1, N):
                if self.dominates(pop_fitness[i], pop_fitness[j]):
                    dominated[i].append(j)
                    domination_counts[j] += 1
                elif self.dominates(pop_fitness[j], pop_fitness[i]):
                    dominated[j].append(i)
                    domination_counts[i] += 1
            if domination_counts[i] == 0:
                f1.append(i)
        fronts.append(f1)
        k = 0
        while fronts[k]:
            next_front = []
            for p in fronts[k]:
                for q in dominated[p]:
                    domination_counts[q] -= 1
                    if domination_counts[q] == 0:
                        next_front.append(q)
            k += 1
            fronts.append(next_front)
        fronts.pop()
        return fronts

    def non_dominated_sort_by_front(self, pop_fitness):
        fronts = self.fast_non_dominated_sort(pop_fitness)
        return fronts

    def is_feasible(self, solution):
        usage = self.compute_objective_usage(solution)
        return np.all(usage <= self.server_capacities + 1e-12)

    def repair_solution(self, solution):
        """
        若容量超載：從過載伺服器中隨機挑請求，嘗試搬到尚有空間的其他伺服器。
        找不到可行搬移時，重新隨機初始化該基因並再修復，最多 max_delay 次。
        """
        sol = solution.copy()
        usage = self.compute_objective_usage(sol)
        attempts = 0
        while attempts < self.max_delay and np.any(usage > self.server_capacities + 1e-12):
            attempts += 1
            # 找出過載的伺服器
            overloaded = np.where(np.any(usage > self.server_capacities + 1e-12, axis=1))[0]
            if overloaded.size == 0:
                break
            j = np.random.choice(overloaded)
            idx = np.where(sol == j)[0]
            if idx.size == 0:
                break
            np.random.shuffle(idx)
            moved = False
            for i in idx:
                # 試著找一台能容納此請求的伺服器
                candidates = list(range(self.num_servers))
                np.random.shuffle(candidates)
                for k in candidates:
                    if k == j:
                        continue
                    new_usage_j = usage[j] - self.item_values[i]
                    new_usage_k = usage[k] + self.item_values[i]
                    if np.all(new_usage_j <= self.server_capacities[j] + 1e-12) and \
                       np.all(new_usage_k <= self.server_capacities[k] + 1e-12):
                        sol[i] = k
                        usage[j] = new_usage_j
                        usage[k] = new_usage_k
                        moved = True
                        break
                if moved:
                    break
            if not moved:
                # 隨機重指派一個基因並重算 usage
                i = np.random.randint(0, self.num_requests)
                sol[i] = np.random.randint(0, self.num_servers)
                usage = self.compute_objective_usage(sol)
        # 若還是不可行，乾脆重生一個可行解
        if not self.is_feasible(sol):
            return self.generate_feasible_solution()
        return sol

    # ---------- NSCO 特有：群體結構與更新 ----------
    def _mode_per_gene(self, sub_pop):
        """
        文化傾向（categorical 版）：對每個請求位置取眾數（若平手隨機挑）。
        """
        tendency = np.empty(self.num_requests, dtype=int)
        for i in range(self.num_requests):
            cnt = Counter(sub_pop[:, i])
            maxc = max(cnt.values())
            modes = [k for k, v in cnt.items() if v == maxc]
            tendency[i] = np.random.choice(modes)
        return tendency

    def _initialize_groups(self, total):
        idx = np.random.permutation(total)
        groups = idx.reshape(self.n_groups, self.coyotes_per_group)
        ages = np.zeros(total, dtype=int)
        return groups, ages

    def _update_coyote(self, i, sub_pop, alpha_coyote, tendency):
        """
        針對整數指派解的 NSCO 更新：
        - 對每個基因：
          * 若 alpha 與 tendency 相同：以較高機率採用該值；偶爾保留原值以維持多樣性
          * 若不同：隨機在 {alpha, tendency} 擇一
        - 少量隨機突變：把某些基因改派到隨機伺服器
        - 最後 repair
        """
        cur = sub_pop[i].copy()
        child = cur.copy()

        # gene-wise 更新
        same = (alpha_coyote == tendency)
        # 同值處：80% 採用該共識值，20% 保留原值
        mask_same = (np.random.rand(self.num_requests) < 0.8) & same
        child[mask_same] = alpha_coyote[mask_same]
        # 不同值處：50/50 二擇一
        diff = ~same
        pick_alpha = (np.random.rand(self.num_requests) < 0.5) & diff
        pick_tend  = diff & (~pick_alpha)
        child[pick_alpha] = alpha_coyote[pick_alpha]
        child[pick_tend]  = tendency[pick_tend]

        # 小幅突變（加強探索）
        mut_mask = (np.random.rand(self.num_requests) < (1.0 / self.num_requests))
        if np.any(mut_mask):
            child[mut_mask] = np.random.randint(0, self.num_servers, size=np.sum(mut_mask))

        return self.repair_solution(child)

    def _crossover(self, sub_pop):
        """
        兩親交叉（整數指派）+ 少量隨機基因。
        """
        p1_idx, p2_idx = np.random.choice(sub_pop.shape[0], 2, replace=False)
        p1, p2 = sub_pop[p1_idx], sub_pop[p2_idx]
        mask = np.random.rand(self.num_requests) < 0.5
        child = np.where(mask, p1, p2)

        # 少量隨機基因
        mut_mask = (np.random.rand(self.num_requests) < (1.0 / self.num_requests))
        if np.any(mut_mask):
            child[mut_mask] = np.random.randint(0, self.num_servers, size=np.sum(mut_mask))
        return self.repair_solution(child)

    def _update_group(self, population, group_indices, ages):
        """
        群內更新（多目標）：
        1) 以群內前沿隨機挑 alpha，並計算 categorical 文化傾向（每基因眾數）
        2) 嘗試更新每隻土狼：若新解支配舊解則採用；同時重置年齡
        3) 產生一隻 pup，若其能支配群內任一解，則以年齡輔助淘汰
        """
        sub_pop = population[group_indices].copy()
        sub_fit = np.array([self.fitness(x) for x in sub_pop])
        sub_age = ages[group_indices].copy()

        fronts = self.fast_non_dominated_sort(sub_fit)
        if len(fronts) == 0 or len(fronts[0]) == 0:
            alpha = sub_pop[np.random.randint(0, sub_pop.shape[0])]
        else:
            alpha = sub_pop[np.random.choice(fronts[0])]

        tendency = self._mode_per_gene(sub_pop)

        # 單體更新
        for i in range(sub_pop.shape[0]):
            new_sol = self._update_coyote(i, sub_pop, alpha, tendency)
            if self.dominates(self.fitness(new_sol), sub_fit[i]):
                sub_pop[i] = new_sol
                sub_fit[i] = self.fitness(new_sol)
                sub_age[i] = 0

        # 交叉兒童
        pup = self._crossover(sub_pop)
        pup_fit = self.fitness(pup)

        dominated_idx = [i for i in range(sub_pop.shape[0]) if self.dominates(pup_fit, sub_fit[i])]
        if dominated_idx:
            # 淘汰年齡最大的被支配者
            worst_local = dominated_idx[np.argmax(sub_age[dominated_idx])]
            sub_pop[worst_local] = pup
            sub_fit[worst_local] = pup_fit
            sub_age[worst_local] = 0

        # 回寫
        population[group_indices] = sub_pop
        ages[group_indices] = sub_age
        return population, ages

    def _maybe_exchange(self, groups):
        if self.n_groups < 2:
            return groups
        prob = self.p_leave * (self.coyotes_per_group ** 2)
        if np.random.rand() < prob:
            g1, g2 = np.random.choice(self.n_groups, size=2, replace=False)
            c1 = np.random.randint(self.coyotes_per_group)
            c2 = np.random.randint(self.coyotes_per_group)
            groups[g1, c1], groups[g2, c2] = groups[g2, c2], groups[g1, c1]
        return groups

    # ---------- 演化主程式（與 NSGA-4 回傳對齊） ----------
    def evolve(self):
        """
        回傳:
          pareto_front: 最終父族群的第一前沿解（np.array of solutions）
          generation_pareto_fronts: list，每代的第一前沿（np.array of solutions）
        """
        # 初始化族群 + 以 NSCO 方式分群（可多於 population_size 時自動裁切/補足）
        total = self.n_groups * self.coyotes_per_group
        if total < self.population_size:
            # 若群格子少於族群大小，補一些可行解
            extra = [self.generate_feasible_solution() for _ in range(self.population_size - total)]
            population = np.vstack([self.population, np.array(extra)])
            total = population.shape[0]
        else:
            population = self.population.copy()

        groups, ages = self._initialize_groups(total)

        # 初始第一前沿紀錄
        pop_fit = np.array([self.fitness(x) for x in population])
        fronts = self.non_dominated_sort_by_front(pop_fit)
        gen_fronts = [population[fronts[0]].copy()] if len(fronts) and len(fronts[0]) else [population[:1].copy()]

        # 迭代
        for _ in range(self.generations):
            for g in range(self.n_groups):
                population, ages = self._update_group(population, groups[g], ages)
            groups = self._maybe_exchange(groups)
            ages += 1

            pop_fit = np.array([self.fitness(x) for x in population])
            fronts = self.non_dominated_sort_by_front(pop_fit)
            if len(fronts) and len(fronts[0]):
                gen_fronts.append(population[fronts[0]].copy())
            else:
                gen_fronts.append(population[:1].copy())

        # 最終以目前族群萃取第一前沿，再「取前 population_size 個」當作與 NSGA4 同規模的父族群比較
        pop_fit = np.array([self.fitness(x) for x in population])
        fronts = self.non_dominated_sort_by_front(pop_fit)
        final_front = population[fronts[0]].copy() if len(fronts) and len(fronts[0]) else population[:1].copy()

        # 輸出對齊 NSGA-4：第一回傳「最終第一前沿解」，第二回傳「每代第一前沿清單」
        return final_front, gen_fronts

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates, scatter_matrix


class NSGA4_LoadBalancing:
    def __init__(self, num_servers, num_requests, population_size, generations, objective_functions, item_values,
                 server_capacities, divisions=4):
        """
        初始化 NSGA4 負載均衡問題
        參數:
          num_servers: 伺服器數量
          num_requests: 請求（物品）數量
          population_size: 種群規模
          generations: 迭代代數
          objective_functions: 目標函數列表，每個函數格式為 f(solution, usage, item_values, server_capacities) -> float
          item_values: 每個請求的負載值，形狀為 (num_requests, 1)
          server_capacities: 每個伺服器的容量上限，形狀為 (num_servers, 1)
          divisions: 用於生成參考點的劃分份數（NSGA3/4常用）
        """
        self.num_servers = num_servers
        self.num_requests = num_requests
        self.population_size = population_size
        self.generations = generations
        self.objective_functions = objective_functions
        self.item_values = np.array(item_values)
        self.server_capacities = np.array(server_capacities)
        self.divisions = divisions
        # 初始種群：每個請求隨機分配到一個伺服器，並確保不超過容量
        self.population = np.array([self.generate_feasible_solution() for _ in range(population_size)])

    def generate_feasible_solution(self):
        num_objectives = self.item_values.shape[1]  # 這裡為1
        current_usage = np.zeros((self.num_servers, num_objectives))
        solution = np.empty(self.num_requests, dtype=int)
        indices = np.random.permutation(self.num_requests)
        for i in indices:
            feasible = []
            for j in range(self.num_servers):
                if np.all(current_usage[j] + self.item_values[i] <= self.server_capacities[j]):
                    feasible.append(j)
            if feasible:
                chosen = np.random.choice(feasible)
                solution[i] = chosen
                current_usage[chosen] += self.item_values[i]
            else:
                raise ValueError(f"請求 {i} 無法分配到任何伺服器，請檢查參數設定！")
        return solution

    # def compute_objective_usage(self, solution):
    #     num_objectives = self.item_values.shape[1]
    #     usage = np.zeros((self.num_servers, num_objectives))
    #     for j in range(self.num_servers):
    #         indices = (solution == j)
    #         if np.any(indices):
    #             usage[j] = np.sum(self.item_values[indices], axis=0)
    #     if np.any(usage > self.server_capacities):
    #         # 若超出容量則重新生成一個可行解
    #         new_solution = self.generate_feasible_solution()
    #         return self.compute_objective_usage(new_solution)
    #     return usage
    def compute_objective_usage(self, solution):
        """只計算 usage，不再偷偷替換解"""
        num_objectives = self.item_values.shape[1]
        usage = np.zeros((self.num_servers, num_objectives))
        for j in range(self.num_servers):
            indices = (solution == j)
            if np.any(indices):
                usage[j] = np.sum(self.item_values[indices], axis=0)
        return usage

    def fitness(self, solution):
        usage = self.compute_objective_usage(solution)
        raw_values = np.array([f(solution, usage, self.item_values, self.server_capacities)
                               for f in self.objective_functions])
        return raw_values

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

    # def weighted_euclidean_distance(self, sol1, sol2):
    #     # 以標準歐氏距離計算兩個解的距離
    #     return np.linalg.norm(sol1 - sol2)
    def weighted_euclidean_distance(self, sol1, sol2):
        """改用負載率向量計算距離"""
        u1 = self.compute_objective_usage(sol1) / self.server_capacities
        u2 = self.compute_objective_usage(sol2) / self.server_capacities
        return np.linalg.norm(u1.flatten() - u2.flatten())

    # def selection(self):
    #     """
    #     NSGA4 的選擇操作：
    #      1. 將當前種群（父代）合併（這裡僅使用當前種群作為集合）
    #      2. 計算每個解的目標值，進行非支配排序
    #      3. 按順序將前沿解加入中間集合 data_PR，直到：
    #           - 前幾前沿全部加入後總數 ≤ 0.5 * population_size，並標記為 "Q1"
    #           - 接下來加入的標記為 "Q2"，直到中間集合規模達到 1.5 * population_size
    #      4. 計算 data_PR 中每對解的加權歐氏距離，然後通過聚類移除法（cluster removal）逐步刪除解，
    #         直到剩下 population_size 個解
    #      5. 返回這些解作為下一代種群
    #     """
    #     pop = self.population.copy()
    #     pop_fitness = np.array([self.fitness(sol) for sol in pop])
    #     fronts = self.non_dominated_sort_by_front(pop_fitness)
    #     data_PR = []  # 每個元素為 (solution, subarea)，subarea 為 "Q1" 或 "Q2"
    #     total = 0
    #     rank = 0
    #     # 將前沿解全部加入，直到總數不超過 0.5 * population_size，標記為 Q1
    #     while rank < len(fronts) and total + len(fronts[rank]) <= 0.5 * self.population_size:
    #         for idx in fronts[rank]:
    #             data_PR.append((pop[idx], "Q1"))
    #         total += len(fronts[rank])
    #         rank += 1
    #     # 從下一個前沿中加入解，標記為 Q2，直到集合大小達到 1.5 * population_size
    #     if rank < len(fronts):
    #         for idx in fronts[rank]:
    #             if total < 1.5 * self.population_size:
    #                 data_PR.append((pop[idx], "Q2"))
    #                 total += 1
    #             else:
    #                 break
    #
    #     L = len(data_PR)
    #     # 計算 data_PR 中每對解之間的距離矩陣
    #     distance_matrix = np.zeros((L, L))
    #     for i in range(L):
    #         for j in range(i + 1, L):
    #             d = self.weighted_euclidean_distance(data_PR[i][0], data_PR[j][0])
    #             distance_matrix[i, j] = d
    #             distance_matrix[j, i] = d
    #
    #     removed = set()
    #     # 當剩餘解數大於 population_size 時，進行聚類移除
    #     while L - len(removed) > self.population_size:
    #         min_d = float('inf')
    #         min_i, min_j = -1, -1
    #         for i in range(L):
    #             if i in removed:
    #                 continue
    #             for j in range(i + 1, L):
    #                 if j in removed:
    #                     continue
    #                 # 只考慮至少有一個屬於 Q2 的對
    #                 if data_PR[i][1] == "Q1" and data_PR[j][1] == "Q1":
    #                     continue
    #                 if distance_matrix[i, j] < min_d:
    #                     min_d = distance_matrix[i, j]
    #                     min_i, min_j = i, j
    #         # 判斷應刪除哪個解
    #         if data_PR[min_i][1] == "Q2" and data_PR[min_j][1] == "Q2":
    #             # 分別計算兩個解與其他未刪除解的最小距離
    #             min_dist_i = float('inf')
    #             min_dist_j = float('inf')
    #             for k in range(L):
    #                 if k in removed or k == min_i or k == min_j:
    #                     continue
    #                 min_dist_i = min(min_dist_i, distance_matrix[min_i, k])
    #                 min_dist_j = min(min_dist_j, distance_matrix[min_j, k])
    #             if min_dist_i < min_dist_j:
    #                 removed.add(min_i)
    #             else:
    #                 removed.add(min_j)
    #         elif data_PR[min_i][1] == "Q1" and data_PR[min_j][1] == "Q2":
    #             removed.add(min_j)
    #         elif data_PR[min_i][1] == "Q2" and data_PR[min_j][1] == "Q1":
    #             removed.add(min_i)
    #         else:
    #             removed.add(min_j)
    #
    #     new_population = []
    #     for i in range(L):
    #         if i not in removed:
    #             new_population.append(data_PR[i][0])
    #     return np.array(new_population)
    def selection(self, parents=None, children=None):
        """
        改為父 + 子合併後再做 NSGA4 selection
        """
        if parents is None:  # 第一次就直接用 population
            pop = self.population.copy()
        else:
            pop = np.concatenate([parents, children])

        pop_fitness = np.array([self.fitness(sol) for sol in pop])
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
                if i in removed: continue
                for j in range(i + 1, L):
                    if j in removed: continue
                    if data_PR[i][1] == "Q1" and data_PR[j][1] == "Q1":
                        continue
                    if distance_matrix[i, j] < min_d:
                        min_d = distance_matrix[i, j]
                        min_i, min_j = i, j
            if data_PR[min_i][1] == "Q2" and data_PR[min_j][1] == "Q2":
                min_dist_i = min(distance_matrix[min_i, k] for k in range(L) if k not in removed and k != min_i)
                min_dist_j = min(distance_matrix[min_j, k] for k in range(L) if k not in removed and k != min_j)
                removed.add(min_i if min_dist_i < min_dist_j else min_j)
            elif data_PR[min_i][1] == "Q1" and data_PR[min_j][1] == "Q2":
                removed.add(min_j)
            elif data_PR[min_i][1] == "Q2" and data_PR[min_j][1] == "Q1":
                removed.add(min_i)
            else:
                removed.add(min_j)

        new_population = [data_PR[i][0] for i in range(L) if i not in removed]
        return np.array(new_population)

    def crossover(self, parent1, parent2):
        if np.random.rand() < 0.9:
            point = np.random.randint(1, self.num_requests)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()
        child1 = self.repair_solution(child1)
        child2 = self.repair_solution(child2)
        return child1, child2

    def mutation(self, solution):
        sol = solution.copy()
        if np.random.rand() < 0.2:
            idx = np.random.randint(0, self.num_requests)
            sol[idx] = np.random.randint(0, self.num_servers)
        sol = self.repair_solution(sol)
        return sol

    def repair_solution(self, solution):
        usage = self.compute_objective_usage(solution)
        if not np.any(usage > self.server_capacities):
            return solution
        improved = True
        while improved and np.any(usage > self.server_capacities):
            improved = False
            for j in range(self.num_servers):
                if np.any(usage[j] > self.server_capacities[j]):
                    indices = np.where(solution == j)[0]
                    np.random.shuffle(indices)
                    for i in indices:
                        for k in range(self.num_servers):
                            if k == j:
                                continue
                            new_usage_j = usage[j] - self.item_values[i]
                            new_usage_k = usage[k] + self.item_values[i]
                            if np.all(new_usage_j <= self.server_capacities[j]) and np.all(
                                    new_usage_k <= self.server_capacities[k]):
                                solution[i] = k
                                usage[j] = new_usage_j
                                usage[k] = new_usage_k
                                improved = True
                                break
                        if np.all(usage[j] <= self.server_capacities[j]):
                            break
            if not improved and np.any(usage > self.server_capacities):
                return self.generate_feasible_solution()
        return solution

    # def evolve(self):
    #     # 用來存儲每輪的最佳前緣
    #     generation_pareto_fronts = []
    #
    #     for _ in range(self.generations):
    #         new_population = []
    #         # NSGA4 選擇操作取得下一代候選解
    #         selected = self.selection()
    #         pop_size = len(selected)
    #         for i in range(0, pop_size, 2):
    #             parent1 = selected[i]
    #             parent2 = selected[(i + 1) % pop_size]
    #             child1, child2 = self.crossover(parent1, parent2)
    #             new_population.append(self.mutation(child1))
    #             new_population.append(self.mutation(child2))
    #         self.population = np.array(new_population[:self.population_size])
    #
    #         # 計算當前人口中每個解的目標值
    #         pop = self.population.copy()
    #         pop_fitness = np.array([self.fitness(sol) for sol in pop])
    #
    #         # 對當前種群進行非支配排序，獲得最佳前緣
    #         fronts = self.non_dominated_sort_by_front(pop_fitness)
    #         current_pareto_front = [pop[idx] for idx in fronts[0]]
    #
    #         # 保存當前輪的最佳前緣
    #         generation_pareto_fronts.append(np.array(current_pareto_front))
    #     # 演化結束後，返回最終種群中的 Pareto 前沿（非支配解）
    #     pop = self.population.copy()
    #     pop_fitness = np.array([self.fitness(sol) for sol in pop])
    #     fronts = self.non_dominated_sort_by_front(pop_fitness)
    #     pareto_front = [pop[idx] for idx in fronts[0]]
    #     return np.array(pareto_front), generation_pareto_fronts
    def evolve(self):
        generation_pareto_fronts = []
        parents = self.population.copy()

        for _ in range(self.generations):
            children = []
            for i in range(0, len(parents), 2):
                p1, p2 = parents[i], parents[(i + 1) % len(parents)]
                c1, c2 = self.crossover(p1, p2)
                children.append(self.mutation(c1))
                children.append(self.mutation(c2))
            children = np.array(children[:self.population_size])

            parents = self.selection(parents, children)

            pop_fitness = np.array([self.fitness(sol) for sol in parents])
            fronts = self.non_dominated_sort_by_front(pop_fitness)
            current_pareto_front = [parents[idx] for idx in fronts[0]]
            generation_pareto_fronts.append(np.array(current_pareto_front))

        pop_fitness = np.array([self.fitness(sol) for sol in parents])
        fronts = self.non_dominated_sort_by_front(pop_fitness)
        pareto_front = [parents[idx] for idx in fronts[0]]
        return np.array(pareto_front), generation_pareto_fronts


#############################################
# 定義 7 個目標函數
#############################################

# def objective_latency(solution, usage, item_values, server_capacities):
#     # 請求處理延遲最小化：用各伺服器負載率的最大值近似
#     load_ratios = usage.flatten() / server_capacities.flatten()
#     return np.sum(load_ratios)


# def objective_throughput(solution, usage, item_values, server_capacities):
#     # 系統吞吐量最大化：用各伺服器 1/(1+負載率) 之和近似，取負值後最小化
#     load_ratios = usage.flatten() / server_capacities.flatten()
#     throughput = np.sum(1.0 / (1 + load_ratios))
#     return -throughput
def objective_latency(solution, usage, item_values, server_capacities):
    load_ratios = usage.flatten()/server_capacities.flatten()
    return np.max(load_ratios)  # 最大負載率

def objective_cost(solution, usage, item_values, server_capacities):
    load_ratios = usage.flatten()/server_capacities.flatten()
    return np.sum(load_ratios**2)  # 平方和

def objective_resource_utilization(solution, usage, item_values, server_capacities):
    # 資源利用率最優化：使各伺服器負載率分布更均衡（標準差最小）
    load_ratios = usage.flatten() / server_capacities.flatten()
    return np.std(load_ratios)


# def objective_error_rate(solution, usage, item_values, server_capacities):
#     # 錯誤率最小化：假定負載率超過 0.8 會產生錯誤，累計超出部分
#     load_ratios = usage.flatten() / server_capacities.flatten()
#     errors = np.maximum(0, load_ratios - 0.8)
#     return np.sum(errors)


# def objective_cost(solution, usage, item_values, server_capacities):
#     # 成本最小化：簡單以各伺服器負載率總和近似
#     load_ratios = usage.flatten() / server_capacities.flatten()
#     return np.sum(load_ratios)

#
# def objective_availability(solution, usage, item_values, server_capacities):
#     # 服務可用性最大化：以伺服器剩餘容量比例的最小值表示，取負值後最小化
#     slack = (server_capacities.flatten() - usage.flatten()) / server_capacities.flatten()
#     return -np.min(slack)

#
# def objective_load_balancing(solution, usage, item_values, server_capacities):
#     # 負載均衡效果最優化：最小化最大與最小負載率的差
#     load_ratios = usage.flatten() / server_capacities.flatten()
#     return np.max(load_ratios) - np.min(load_ratios)

# def latency_curve(usage):


#############################################
# 主程式入口
#############################################
if __name__ == "__main__":
    # 設定問題參數：假設每個請求負載隨機在 1 到 4 之間，伺服器容量在 50 到 60 之間
    num_servers = 15
    num_requests = 300
    population_size = 20
    generations = 100

    num_objectives = 1  # 每個請求只有一個負載值
    item_values = np.random.randint(1, 5, [num_requests, num_objectives])
    server_capacities = np.random.randint(50, 61, [num_servers, num_objectives])

    # 目標函數列表（共 7 個目標）
    # objectives = [objective_latency, objective_throughput, objective_resource_utilization,
    #               objective_error_rate, objective_cost, objective_availability, objective_load_balancing]

    objectives = [objective_resource_utilization, objective_latency, objective_cost]

    nsga4 = NSGA4_LoadBalancing(num_servers, num_requests, population_size, generations, objectives, item_values,
                                server_capacities)
    pareto_front, generation_pareto_fronts = nsga4.evolve()

    print("最佳前沿解（每個請求分配到的伺服器編號向量）：")
    print(pareto_front)

    best_usage = [nsga4.compute_objective_usage(sol) for sol in pareto_front]
    print("\n各伺服器處理的總負載（使用值）：")
    print(best_usage)

    print("\n各伺服器的容量上限：")
    print(server_capacities)

    occupancy_rate = [usage / server_capacities for usage in best_usage]
    print("\n各伺服器的負載率：")
    print(occupancy_rate)

    print("\n各目標函數的值：")
    # print("吞吐量指標（負吞吐量）：",
    #       [objective_throughput(sol, best_usage[i], nsga4.item_values, nsga4.server_capacities) for i, sol in
    #        enumerate(pareto_front)])
    print("資源均衡指標（負載率標準差）：",
          [objective_resource_utilization(sol, best_usage[i], nsga4.item_values, nsga4.server_capacities) for i, sol in
           enumerate(pareto_front)])
    print("延遲指標（最大負載率）：",
          [objective_latency(sol, best_usage[i], nsga4.item_values, nsga4.server_capacities) for i, sol in
           enumerate(pareto_front)])
    # print("錯誤率指標：",
    #       [objective_error_rate(sol, best_usage[i], nsga4.item_values, nsga4.server_capacities) for i, sol in
    #        enumerate(pareto_front)])
    print("成本指標（負載率總和）：",
          [objective_cost(sol, best_usage[i], nsga4.item_values, nsga4.server_capacities) for i, sol in
           enumerate(pareto_front)])
    # print("可用性指標（負最小剩餘比）：",
    #       [objective_availability(sol, best_usage[i], nsga4.item_values, nsga4.server_capacities) for i, sol in
    #        enumerate(pareto_front)])
    # print("負載均衡指標（最大與最小負載率差）：",
    #       [objective_load_balancing(sol, best_usage[i], nsga4.item_values, nsga4.server_capacities) for i, sol in
    #        enumerate(pareto_front)])

    generation_objectives_data = []
    for each_sol in range(len(generation_pareto_fronts)):
        obj_data = []
        for sol in generation_pareto_fronts[each_sol]:
            obj_vals = nsga4.fitness(sol)
            obj_data.append({
                'LoadBalance': float(obj_vals[0]),
                'Average Delay': float(obj_vals[1]),
                'Cost': float(obj_vals[2]),
            })
        generation_objectives_data.append({
            "Generation": each_sol,
            "sol": obj_data
        })
    generation_df = pd.DataFrame(generation_objectives_data)
    print(generation_df)
    generation_df.to_csv(f'NSGA4_generation_solutions.csv', index=False)
