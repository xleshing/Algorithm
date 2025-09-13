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
