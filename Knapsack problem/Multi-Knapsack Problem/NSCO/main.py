import numpy as np
import matplotlib.pyplot as plt


# --------------------------
# 非支配排序輔助函式 (均視為 minimization 目標)
# --------------------------
def dominates(u, v):
    """
    判斷向量 u 是否支配向量 v（均為 minimization 目標）
    u 支配 v 當且僅當 u 的所有目標值均 ≤ v，且至少有一個目標值 < v
    """
    return np.all(u <= v) and np.any(u < v)


def fast_non_dominated_sort(pop_objs):
    """
    快速非支配排序
    輸入:
      pop_objs: (N x M) 陣列，每列為一解的 M 個目標值
    回傳:
      fronts: 一串列表，每一列表包含該 front 中解的索引 (第一 front 為 Pareto 前沿)
    """
    N = pop_objs.shape[0]
    domination_counts = np.zeros(N, dtype=int)
    dominated = [[] for _ in range(N)]
    fronts = []

    front1 = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if dominates(pop_objs[i], pop_objs[j]):
                dominated[i].append(j)
            elif dominates(pop_objs[j], pop_objs[i]):
                domination_counts[i] += 1
        if domination_counts[i] == 0:
            front1.append(i)
    fronts.append(front1)

    i = 0
    while fronts[i]:
        next_front = []
        for j in fronts[i]:
            for k in dominated[j]:
                domination_counts[k] -= 1
                if domination_counts[k] == 0:
                    next_front.append(k)
        i += 1
        fronts.append(next_front)
    fronts.pop()  # 移除最後空的 front
    return fronts


# --------------------------
# NSCO for Knapsack Problem (改為多目標：效益與原狀態改動數)
# --------------------------
class NSCO_Algorithm:
    def __init__(self, turn_node_on, d, value, weight, capacity, coyotes_per_group, n_groups, p_leave, max_iter,
                 max_delay, original_status, objs_func=None):
        """
        參數定義：
          turn_node_on: 初始節點開啟狀態（若有用）
          d: 問題維度（物品數量）
          value: 每項物品的資源提供值 (1D 陣列)
          weight: 總資源使用量（scalar）
          capacity: 容量限制（SLA）
          coyotes_per_group: 每群土狼數量
          n_groups: 群數
          p_leave: 群間交換機率參數
          max_iter: 最大迭代次數
          max_delay: 約束重生成嘗試次數
          original_status: 原始背包解 (0/1 列表)
          external_obj_func: 外部傳入的目標函數 (接收 x 輸入並回傳效益值)
        """
        self.turn_node_on = turn_node_on
        self.d = d
        self.value = np.array(value)
        self.weight = weight
        self.capacity = capacity
        self.coyotes_per_group = coyotes_per_group
        self.n_groups = n_groups
        self.p_leave = p_leave
        self.max_iter = max_iter
        self.max_delay = max_delay
        self.original_status = np.array(original_status)
        # 採用外部目標函數或預設函數
        self.objs_func = objs_func

    def is_feasible(self, x):
        """
        檢查解 x 是否滿足容量限制：
          若未選物品或目標比率超過容量上限，視為不可行
        """
        if np.dot(x, self.value) == 0:
            return False
        ratio = self.weight / np.dot(x, self.value) * 100
        return (ratio <= self.capacity)

    def repair(self, x):
        """
        當解超過容量上限時，重新生成一個滿足容量限制的解。
        會嘗試最多 max_delay 次，直到找到可行解為止。
        """
        attempts = 0
        while not self.is_feasible(x) and attempts < self.max_delay:
            x = np.random.randint(2, size=self.d)
            attempts += 1
        return x

    def multiobj(self, x):
        f1 = self.objs_func[0]
        f2 = self.objs_func[1]
        return np.array([f1(x), f2(x, self.original_status)])

    def nsco_initialize_population(self):
        """
        初始化 0/1 背包族群，確保解不違反容量限制
        回傳：
          population: (total_coyotes, d) 二進制陣列
          groups: (n_groups, coyotes_per_group) 群體索引配置
          population_age: 每隻土狼的年齡 (全設 0)
        """
        total_coyotes = self.n_groups * self.coyotes_per_group
        population = np.random.randint(2, size=(total_coyotes, self.d))
        # 利用 repair 函數修正不滿足容量限制的解
        for idx in range(total_coyotes):
            if not self.is_feasible(population[idx, :]):
                population[idx, :] = self.repair(population[idx, :])
        indices = np.random.permutation(total_coyotes)
        groups = indices.reshape(self.n_groups, self.coyotes_per_group)
        population_age = np.zeros(total_coyotes, dtype=int)
        return population, groups, population_age

    def nsco_compute_cultural_tendency(self, sub_pop):
        """
        文化傾向：以群內各解的中位數（取 0/1 後四捨五入）作為文化基因
        """
        return np.round(np.median(sub_pop, axis=0)).astype(int)

    def nsco_update_coyote(self, i, sub_pop, alpha_coyote, cultural_tendency):
        """
        更新單隻土狼：
         隨機選取群內兩個其他解，計算差異向量後以隨機方式決定更新方向，
         轉換為 0/1 向量作為候選解。
        """
        candidates = list(range(self.coyotes_per_group))
        candidates.remove(i)
        qj1 = np.random.choice(candidates)
        candidates.remove(qj1)
        qj2 = np.random.choice(candidates)
        delta1 = alpha_coyote - sub_pop[qj1, :]
        delta2 = cultural_tendency - sub_pop[qj2, :]
        rand_mask = np.random.rand(self.d) < 0.5
        candidate = np.where(rand_mask, np.abs(delta1), np.abs(delta2))
        new_sol = (candidate > 0).astype(int)
        # 修正新產生的解
        new_sol = self.repair(new_sol)
        return new_sol

    def nsco_crossover(self, sub_pop):
        """
        父代交叉產生 pup：
         隨機選取兩個父代，根據隨機遮罩決定每一維度來源，
         未選到的部分以隨機 0/1 產生突變
        """
        parents_idx = np.random.choice(self.coyotes_per_group, 2, replace=False)
        mutation_prob = 1 / self.d
        parent_prob = (1 - mutation_prob) / 2

        pdr = np.random.permutation(self.d)
        p1_mask = np.zeros(self.d, dtype=bool)
        p2_mask = np.zeros(self.d, dtype=bool)
        p1_mask[pdr[0]] = True
        p2_mask[pdr[1]] = True
        if self.d > 2:
            rand_vals = np.random.rand(self.d - 2)
            p1_mask[pdr[2:]] = (rand_vals < parent_prob)
            p2_mask[pdr[2:]] = (rand_vals > (1 - parent_prob))
        mut_mask = ~(p1_mask | p2_mask)
        pup = (p1_mask * sub_pop[parents_idx[0], :] +
               p2_mask * sub_pop[parents_idx[1], :] +
               mut_mask * np.random.randint(2, size=self.d))
        # 修正交叉產生的解
        pup = self.repair(pup)
        return pup

    def nsco_update_group(self, population, group_indices, population_age):
        """
        對一個群內進行更新：
         (1) 依多目標評價及非支配排序取得群內 Pareto 前沿，
             隨機選取其中一個作為領導者 (alpha) 並計算群文化傾向；
         (2) 對每隻個體利用 nsco_update_coyote 嘗試更新，
             若新解在多目標上支配原解則予以採納；
         (3) 利用 nsco_crossover 產生 pup，
             若其在多目標上支配群中部分解，則以年齡輔助替換其中年齡最高者。
        """
        sub_pop = population[group_indices, :].copy()
        sub_objs = np.array([self.multiobj(x) for x in sub_pop])
        sub_age = population_age[group_indices].copy()
        n_pack = len(group_indices)

        fronts = fast_non_dominated_sort(sub_objs)
        if len(fronts[0]) > 0:
            alpha_idx = np.random.choice(fronts[0])
            alpha_coyote = sub_pop[alpha_idx, :].copy()
        else:
            alpha_coyote = sub_pop[0, :].copy()
        cultural_tendency = self.nsco_compute_cultural_tendency(sub_pop)

        for i in range(n_pack):
            new_sol = self.nsco_update_coyote(i, sub_pop, alpha_coyote, cultural_tendency)
            new_obj = self.multiobj(new_sol)
            if dominates(new_obj, sub_objs[i]):
                sub_pop[i, :] = new_sol
                sub_objs[i] = new_obj
                sub_age[i] = 0

        if self.d > 1:
            pup = self.nsco_crossover(sub_pop)
            pup_obj = self.multiobj(pup)
            dominated_indices = []
            for i in range(n_pack):
                if dominates(pup_obj, sub_objs[i]):
                    dominated_indices.append(i)
            if dominated_indices:
                ages_candidates = sub_age[dominated_indices]
                worst_idx_local = dominated_indices[np.argmax(ages_candidates)]
                sub_pop[worst_idx_local, :] = pup
                sub_objs[worst_idx_local] = pup_obj
                sub_age[worst_idx_local] = 0

        population[group_indices, :] = sub_pop
        population_age[group_indices] = sub_age
        return population, population_age

    def nsco_coyote_exchange(self, groups):
        """
        依機率 p_leave（乘上 coyotes_per_group²）隨機抽取兩個不同群，互換各自一隻解
        """
        n_groups, coy_per_group = groups.shape
        exchange_prob = self.p_leave * (self.coyotes_per_group ** 2)
        if n_groups < 2:
            return groups
        if np.random.rand() < exchange_prob:
            g1, g2 = np.random.choice(n_groups, size=2, replace=False)
            c1 = np.random.randint(coy_per_group)
            c2 = np.random.randint(coy_per_group)
            tmp = groups[g1, c1]
            groups[g1, c1] = groups[g2, c2]
            groups[g2, c2] = tmp
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
        fronts = fast_non_dominated_sort(pop_objs)
        global_pf = fronts[0]
        global_pf_solutions = population[global_pf, :].copy()
        archive = [global_pf_solutions]

        for iteration in range(self.max_iter):
            for g in range(self.n_groups):
                group_indices = groups[g, :]
                population, population_age = self.nsco_update_group(population, group_indices, population_age)
            groups = self.nsco_coyote_exchange(groups)
            population_age += 1

            pop_objs = np.array([self.multiobj(x) for x in population])
            fronts = fast_non_dominated_sort(pop_objs)
            global_pf = fronts[0]
            global_pf_solutions = population[global_pf, :].copy()
            archive.append(global_pf_solutions)
            print(f"Iteration {iteration + 1}: Pareto Front size = {len(global_pf)}")

        # 最終檢查：若全局前沿中無可行解，則以原始狀態作為解
        feasible_front = [sol for sol in global_pf_solutions if self.is_feasible(sol)]
        if len(feasible_front) == 0:
            global_pf_solutions = np.array([self.original_status])
        return global_pf_solutions, archive


# --------------------------
# 測試與示意 (main)
# --------------------------
if __name__ == "__main__":
    # 外部傳入的目標函數範例，可根據需求調整邏輯
    def load(x):
        ratio = w / np.dot(x, v) * 100
        return 1 / ratio + 1e-6

    def change(x, original_status):
        return np.sum(np.abs(x - original_status))

    # 參數設定
    v = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]  # 每項物品資源量
    w = np.sum([500, 0, 0, 0, 0, 0, 500, 0, 0, 0])  # 總資源使用量
    c = 55  # SLA 容量限制

    nsco = NSCO_Algorithm(
        turn_node_on=0,
        d=len(v),
        value=v,
        weight=w,
        capacity=c,
        coyotes_per_group=5,
        n_groups=5,
        p_leave=0.1,
        max_iter=100,
        max_delay=100,
        original_status=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        objs_func=[load, change]  # 使用外部目標函數
    )

    global_pf_solutions, archive = nsco.NSCO_main()

    print("Final Pareto Front Solutions (NSCO):")
    print(global_pf_solutions)

    # 繪製最終 Pareto 解在目標空間中的分布
    pf_obj_values = np.array([nsco.multiobj(sol) for sol in global_pf_solutions])
    plt.figure()
    plt.scatter(pf_obj_values[:, 0], pf_obj_values[:, 1], c='red', marker='o', edgecolors='k')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Final Pareto Front')
    plt.grid(True)
    plt.show()
