import numpy as np


class NSGA3_Knapsack:
    def __init__(self, num_knapsacks, num_items, population_size, generations, objective_functions, item_values,
                 knapsack_capacities, divisions=4):
        """
        初始化 NSGA-III 多目標背包問題（每個物品僅分配到一個背包）

        參數:
          num_knapsacks: 背包數量
          num_items: 物品數量
          population_size: 種群規模
          generations: 迭代代數
          objective_functions: 目標函數列表，格式為 f(solution, usage, item_values, knapsack_capacities) -> float
          item_values: 每個物品在各目標下的價值，形狀為 (num_items, num_objectives)
          knapsack_capacities: 每個背包在各目標下的容量上限，形狀為 (num_knapsacks, num_objectives)
          divisions: 用於生成參考點的劃分份數
        """
        self.num_knapsacks = num_knapsacks
        self.num_items = num_items
        self.population_size = population_size
        self.generations = generations
        self.objective_functions = objective_functions
        self.item_values = np.array(item_values)
        self.knapsack_capacities = np.array(knapsack_capacities)
        self.divisions = divisions  # 參考點劃分份數
        # 初始種群確保每個物品只分配到一個背包且滿足容量約束
        self.population = np.array([self.generate_feasible_solution() for _ in range(population_size)])
        self.fitness_history = []

    def generate_feasible_solution(self):
        num_objectives = self.item_values.shape[1]
        current_usage = np.zeros((self.num_knapsacks, num_objectives))
        solution = np.empty(self.num_items, dtype=int)
        indices = np.random.permutation(self.num_items)
        for i in indices:
            feasible = []
            for j in range(self.num_knapsacks):
                if np.all(current_usage[j] + self.item_values[i] <= self.knapsack_capacities[j]):
                    feasible.append(j)
            if feasible:
                chosen = np.random.choice(feasible)
                solution[i] = chosen
                current_usage[chosen] += self.item_values[i]
            else:
                raise ValueError(f"物品 {i} 無法分配到任何背包，請檢查參數設定！")
        return solution

    def compute_objective_usage(self, solution):
        num_objectives = self.item_values.shape[1]
        usage = np.zeros((self.num_knapsacks, num_objectives))
        for j in range(self.num_knapsacks):
            indices = (solution == j)
            if np.any(indices):
                usage[j] = np.sum(self.item_values[indices], axis=0)
        if np.any(usage > self.knapsack_capacities):
            print("發現不可行子代，重新初始化一個符合約束的解。")
            new_solution = self.generate_feasible_solution()
            return self.compute_objective_usage(new_solution)
        return usage

    def fitness(self, solution):
        """
        計算個體適應度：
          先計算各背包在各目標下的使用值，再依次調用各目標函數。
        """
        usage = self.compute_objective_usage(solution)
        raw_values = np.array(
            [f(solution, usage, self.item_values, self.knapsack_capacities) for f in self.objective_functions])
        return raw_values

    def normalize_population(self, fitness_values):
        """
        NSGA-III 歸一化步驟：
          1. 計算理想點（各目標的最小值）
          2. 將目標值平移（減去理想點）
          3. 利用 ASF 函數求各目標的極值點
          4. 構造超平面並求各坐標軸截距
          5. 用截距將目標值歸一化到 [0, 1]
        """
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

    def generate_reference_points(self, M, p):
        """
        生成 M 維空間中劃分為 p 份的參考點（單位單純形上的點）
        """

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
        """
        將每個個體的歸一化目標向量與各參考點對應的參考線（從原點出發）做關聯，
        計算垂直距離，並返回最短距離的參考點索引及該距離。
        """
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
                    proj = np.dot(sol, r) * r  # 投影向量
                    d = np.linalg.norm(sol - proj)
                    dists.append(d)
                idx = np.argmin(dists)
                assoc_indices.append(idx)
                distances.append(dists[idx])
        return np.array(assoc_indices), np.array(distances)

    def niche_selection(self, front, assoc, niche_count, remaining_slots):
        """
        對部分前沿中關聯到參考點的個體進行小生境選擇：
          根據每個參考點已有個體數（niche count），選擇距離較近且所屬參考點擁擠度較低的個體
        """
        candidates = list(front)
        selected = []
        # 假定 assoc 的順序與 front 中個體順序一致
        while remaining_slots > 0 and candidates:
            candidate_info = []
            for i, global_idx in enumerate(front):
                if global_idx in candidates:
                    rp = assoc[0][i]
                    d = assoc[1][i]
                    candidate_info.append((global_idx, rp, d))
            if not candidate_info:
                break
            min_count = min([niche_count[info[1]] for info in candidate_info])
            candidate_rps = [info for info in candidate_info if niche_count[info[1]] == min_count]
            selected_candidate = min(candidate_rps, key=lambda x: x[2])
            selected.append(selected_candidate[0])
            candidates.remove(selected_candidate[0])
            niche_count[selected_candidate[1]] += 1
            remaining_slots -= 1
        return selected

    def selection(self):
        """
        NSGA-III 選擇操作：
          1. 對全體種群計算目標值，進行非支配排序，獲得各前沿集合
          2. 依次將完整前沿加入下一代，直到最後一個前沿無法全部收納
          3. 對最後前沿中的個體進行歸一化和參考點關聯，
             並依據各參考點的小生境數量選擇部分個體填滿下一代
        """
        population_fitness = np.array([self.fitness(sol) for sol in self.population])
        fronts = self.non_dominated_sort_by_front(population_fitness)
        new_indices = []
        for front in fronts:
            if len(new_indices) + len(front) <= self.population_size:
                new_indices.extend(front)
            else:
                remaining_slots = self.population_size - len(new_indices)
                normalized_objectives = self.normalize_population(population_fitness)
                reference_points = self.generate_reference_points(self.item_values.shape[1], self.divisions)
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
        return self.population[new_indices]

    def crossover(self, parent1, parent2):
        """
        單點交叉：隨機選取一個交叉點，前半部分來自 parent1，後半部分來自 parent2，
        交叉後利用 repair_solution 修正使子代滿足約束
        """
        if np.random.rand() < 0.9:
            point = np.random.randint(1, self.num_items)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        child1 = self.repair_solution(child1)
        child2 = self.repair_solution(child2)
        return child1, child2

    def mutation(self, solution):
        """
        變異操作：隨機選取一個物品，改變其背包分配，然後修正解
        """
        sol = solution.copy()
        if np.random.rand() < 0.2:
            idx = np.random.randint(0, self.num_items)
            sol[idx] = np.random.randint(0, self.num_knapsacks)
        sol = self.repair_solution(sol)
        return sol

    def repair_solution(self, solution):
        usage = self.compute_objective_usage(solution)
        if not np.any(usage > self.knapsack_capacities):
            return solution
        improved = True
        while improved and np.any(usage > self.knapsack_capacities):
            improved = False
            for j in range(self.num_knapsacks):
                if np.any(usage[j] > self.knapsack_capacities[j]):
                    indices = np.where(solution == j)[0]
                    np.random.shuffle(indices)
                    for i in indices:
                        for k in range(self.num_knapsacks):
                            if k == j:
                                continue
                            new_usage_j = usage[j] - self.item_values[i]
                            new_usage_k = usage[k] + self.item_values[i]
                            if np.all(new_usage_j <= self.knapsack_capacities[j]) and np.all(
                                    new_usage_k <= self.knapsack_capacities[k]):
                                solution[i] = k
                                usage[j] = new_usage_j
                                usage[k] = new_usage_k
                                improved = True
                                break
                        if np.all(usage[j] <= self.knapsack_capacities[j]):
                            break
            if not improved and np.any(usage > self.knapsack_capacities):
                print("子代修正失敗，重新生成一個可行解。")
                return self.generate_feasible_solution()
        return solution

    def evolve(self):
        for _ in range(self.generations):
            new_population = []
            selected = self.selection()
            for i in range(0, self.population_size, 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % self.population_size]
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutation(child1))
                new_population.append(self.mutation(child2))
            self.population = np.array(new_population[:self.population_size])
        population_fitness = np.array([self.fitness(sol) for sol in self.population])
        fronts = self.non_dominated_sort_by_front(population_fitness)
        best_front = fronts[0]
        best_index = best_front[0]
        return self.population[best_index]


# 定義兩個目標函數示例
def objective_example1(solution, usage, item_values, knapsack_capacities):
    """
    目標1：使各背包使用值均衡——最小化各背包使用率標準差
    """
    return np.std(usage / knapsack_capacities)


def objective_example2(solution, usage, item_values, knapsack_capacities):
    """
    目標2：最小化所有背包的總使用率
    """
    return np.sum(usage / knapsack_capacities)


if __name__ == "__main__":
    # 參數設定
    num_knapsacks = 5  # 背包數量
    num_items = 50  # 物品數量
    population_size = 20
    generations = 50

    # 此處設定 2 維目標（因此每個物品和每個背包均為2維數據）
    num_objectives = 2
    item_values = np.random.randint(1, 2, [num_items, num_objectives])
    knapsack_capacities = np.random.randint(20, 21, [num_knapsacks, num_objectives])

    objectives = [objective_example1, objective_example2]

    nsga3 = NSGA3_Knapsack(num_knapsacks, num_items, population_size, generations, objectives, item_values,
                           knapsack_capacities)
    best_solution = nsga3.evolve()

    print("最佳解（每個物品的分配向量）：")
    print(best_solution)

    best_usage = nsga3.compute_objective_usage(best_solution)
    print("\n各背包在各目標下的使用值：")
    print(best_usage)

    print("\n各背包在各目標下的容量上限：")
    print(knapsack_capacities)

    occupancy_rate = best_usage / knapsack_capacities
    print("\n每個背包的每個目標維度佔用率：")
    print(occupancy_rate)

    print("\nNormalized spread (std of occupancy rates):", np.std(occupancy_rate))
