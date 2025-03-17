import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates, scatter_matrix
class NSGA3_Knapsack:
    def __init__(self, num_knapsacks, num_items, population_size, generations, objective_functions, item_values,
                 knapsack_capacities, divisions=4):
        """
        初始化 NSGA-III 多目標負載均衡問題
        參數:
          num_knapsacks: 伺服器（背包）數量
          num_items: 請求（物品）數量
          population_size: 種群規模
          generations: 迭代代數
          objective_functions: 目標函數列表，每個函數格式為 f(solution, usage, item_values, knapsack_capacities) -> float
          item_values: 每個請求的負載值，形狀為 (num_items, 1)
          knapsack_capacities: 每個伺服器的容量上限，形狀為 (num_knapsacks, 1)
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
        self.population = np.array([self.generate_feasible_solution() for _ in range(population_size)])
        self.fitness_history = []

    def generate_feasible_solution(self):
        num_objectives = self.item_values.shape[1]  # 這裡為1
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
                raise ValueError(f"請求 {i} 無法分配到任何伺服器，請檢查參數設定！")
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
        usage = self.compute_objective_usage(solution)
        raw_values = np.array(
            [f(solution, usage, self.item_values, self.knapsack_capacities) for f in self.objective_functions])
        return raw_values

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

    def fast_non_dominated_sort(self, population_fitness):
        num_solutions = len(population_fitness)
        ranks = np.zeros(num_solutions, dtype=int)
        domination_counts = np.zeros(num_solutions, dtype=int)
        dominated_solutions = [[] for _ in range(num_solutions)]
        front = []

        for i in range(num_solutions):
            for j in range(i + 1, num_solutions):
                if np.all(population_fitness[i] <= population_fitness[j]) and np.any(population_fitness[i] < population_fitness[j]):
                    dominated_solutions[i].append(j)
                    domination_counts[j] += 1
                elif np.all(population_fitness[j] <= population_fitness[i]) and np.any(population_fitness[j] < population_fitness[i]):
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
            min_count = min([niche_count[info[1]] for info in candidate_info])
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
                            if np.all(new_usage_j <= self.knapsack_capacities[j]) and np.all(new_usage_k <= self.knapsack_capacities[k]):
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
        # best_index = best_front[0]
        # return self.population[best_index]
        return self.population[best_front]

# 以下定義7個目標函數，新目標如下：
# 1. 請求處理延遲最小化：以各伺服器負載率最大值近似
def objective_latency(solution, usage, item_values, knapsack_capacities):
    load_ratios = usage.flatten() / knapsack_capacities.flatten()
    return np.max(load_ratios)

# 2. 系統吞吐量最大化：以各伺服器 1/(1+負載率) 之和近似，取負值後最小化
def objective_throughput(solution, usage, item_values, knapsack_capacities):
    load_ratios = usage.flatten() / knapsack_capacities.flatten()
    throughput = np.sum(1.0 / (1 + load_ratios))
    return -throughput

# 3. 資源利用率最優化：使各伺服器負載率分布更均衡（標準差最小）
def objective_resource_utilization(solution, usage, item_values, knapsack_capacities):
    load_ratios = usage.flatten() / knapsack_capacities.flatten()
    return np.std(load_ratios)

# 4. 錯誤率最小化：假定負載率超過0.8產生錯誤，累計超過部分
def objective_error_rate(solution, usage, item_values, knapsack_capacities):
    load_ratios = usage.flatten() / knapsack_capacities.flatten()
    errors = np.maximum(0, load_ratios - 0.8)
    return np.sum(errors)

# 5. 成本最小化：簡單以負載率總和近似
def objective_cost(solution, usage, item_values, knapsack_capacities):
    load_ratios = usage.flatten() / knapsack_capacities.flatten()
    return np.sum(load_ratios)

# 6. 服務可用性最大化：以伺服器剩餘容量比例的最小值表示，取負值最小化
def objective_availability(solution, usage, item_values, knapsack_capacities):
    slack = (knapsack_capacities.flatten() - usage.flatten()) / knapsack_capacities.flatten()
    return -np.min(slack)

# 7. 負載均衡效果最優化：最小化最大與最小負載率的差
def objective_load_balancing(solution, usage, item_values, knapsack_capacities):
    load_ratios = usage.flatten() / knapsack_capacities.flatten()
    return np.max(load_ratios) - np.min(load_ratios)

if __name__ == "__main__":
    # 參數設定：假設每個請求的負載隨機在1到4之間，伺服器容量在50到60之間
    num_knapsacks = 5
    num_items = 50
    population_size = 20
    generations = 50

    num_objectives = 1  # 每個請求只有一個負載值
    item_values = np.random.randint(1, 5, [num_items, num_objectives])
    knapsack_capacities = np.random.randint(50, 61, [num_knapsacks, num_objectives])

    # 目標函數列表（共7個目標）
    objectives = [objective_latency, objective_throughput, objective_resource_utilization,
                  objective_error_rate, objective_cost, objective_availability, objective_load_balancing]

    nsga3 = NSGA3_Knapsack(num_knapsacks, num_items, population_size, generations, objectives, item_values,
                           knapsack_capacities)
    best_solution = nsga3.evolve()

    print("最佳解（每個請求分配到的伺服器編號向量）：")
    print(best_solution)

    best_usage = [nsga3.compute_objective_usage(solution) for solution in best_solution]
    print("\n各伺服器處理的總負載（使用值）：")
    print(best_usage)

    print("\n各伺服器的容量上限：")
    print(knapsack_capacities)

    occupancy_rate = best_usage / knapsack_capacities
    print("\n各伺服器的負載率：")
    print(occupancy_rate)

    print("\n各目標函數的值：")
    print("延遲指標（最大負載率）：", [objective_latency(solution, best_usage[usage_index], nsga3.item_values, nsga3.knapsack_capacities) for usage_index, solution in enumerate(best_solution, 0)])
    print("吞吐量指標（負吞吐量）：", [objective_throughput(solution, best_usage[usage_index], nsga3.item_values, nsga3.knapsack_capacities) for usage_index, solution in enumerate(best_solution, 0)])
    print("資源均衡指標（負載率標準差）：", [objective_resource_utilization(solution, best_usage[usage_index], nsga3.item_values, nsga3.knapsack_capacities) for usage_index, solution in enumerate(best_solution, 0)])
    print("錯誤率指標：", [objective_error_rate(solution, best_usage[usage_index], nsga3.item_values, nsga3.knapsack_capacities) for usage_index, solution in enumerate(best_solution, 0)])
    print("成本指標（負載率總和）：", [objective_cost(solution, best_usage[usage_index], nsga3.item_values, nsga3.knapsack_capacities) for usage_index, solution in enumerate(best_solution, 0)])
    print("可用性指標（負最小剩餘比）：", [objective_availability(solution, best_usage[usage_index], nsga3.item_values, nsga3.knapsack_capacities) for usage_index, solution in enumerate(best_solution, 0)])
    print("負載均衡指標（最大與最小負載率差）：", [objective_load_balancing(solution, best_usage[usage_index], nsga3.item_values, nsga3.knapsack_capacities) for usage_index, solution in enumerate(best_solution, 0)])

    # 假設 best_solution 和 best_usage 已經從 NSGA-III 演化過程中獲得，
    # 並且 nsga3 以及各目標函數（objective_latency、objective_throughput、...）均已定義

    # 例如：
    # best_solution = [sol1, sol2, sol3, ...]  # 每個 sol 為一個分配向量
    # best_usage = [nsga3.compute_objective_usage(sol) for sol in best_solution]

    # 收集每個解的各目標值到列表中
    objectives_data = []
    for idx, solution in enumerate(best_solution):
        # 對應解的使用值
        usage = best_usage[idx]
        # 計算各目標的值
        latency_val = objective_latency(solution, usage, nsga3.item_values, nsga3.knapsack_capacities)
        throughput_val = objective_throughput(solution, usage, nsga3.item_values, nsga3.knapsack_capacities)
        resource_utilization_val = objective_resource_utilization(solution, usage, nsga3.item_values,
                                                                  nsga3.knapsack_capacities)
        error_rate_val = objective_error_rate(solution, usage, nsga3.item_values, nsga3.knapsack_capacities)
        cost_val = objective_cost(solution, usage, nsga3.item_values, nsga3.knapsack_capacities)
        availability_val = objective_availability(solution, usage, nsga3.item_values, nsga3.knapsack_capacities)
        load_balancing_val = objective_load_balancing(solution, usage, nsga3.item_values, nsga3.knapsack_capacities)

        # 將各目標結果存入字典中（解的編號作為分類標識）
        objectives_data.append({
            'Solution': f"Sol_{idx}",
            'Latency': latency_val,
            'Throughput': throughput_val,
            'ResourceUtilization': resource_utilization_val,
            'ErrorRate': error_rate_val,
            'Cost': cost_val,
            'Availability': availability_val,
            'LoadBalancing': load_balancing_val
        })

    # 將數據轉換成 DataFrame
    df = pd.DataFrame(objectives_data)
    print("各目標函數的彙總數據：")
    print(df)

    # --- 平行座標圖 ---
    plt.figure(figsize=(12, 6))
    # 以 'Solution' 作為類別標籤，每條線代表一個解
    parallel_coordinates(df, 'Solution', colormap=plt.get_cmap("Set2"))
    plt.title("Parallel Coordinates of Pareto Front Solutions (NSGA3)")
    plt.ylabel("Objective Function Values")
    plt.tight_layout()
    plt.show()


    def scatter_matrix_with_mean(df, figsize=(12, 12)):
        """
        自定義散點矩陣：
          - 非對角子圖：展示兩兩變數間的散點圖
          - 對角子圖：顯示該列數據的均值
        """
        num_vars = df.shape[1]
        fig, axes = plt.subplots(num_vars, num_vars, figsize=figsize)

        # 遍歷每一個子圖
        for i, col_i in enumerate(df.columns):
            for j, col_j in enumerate(df.columns):
                ax = axes[i, j]
                if i == j:
                    # 對角位置：顯示均值文本
                    mean_val = df[col_i].mean()
                    ax.text(0.5, 0.5, f"Mean: {mean_val:.2f}",
                            horizontalalignment='center', verticalalignment='center', fontsize=12)
                    # 去除刻度
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_frame_on(False)
                else:
                    # 非對角位置：繪製散點圖
                    ax.scatter(df[col_j], df[col_i], alpha=0.8, color='blue')
                    # 如非最外側圖形，隱藏刻度標籤
                    if i < num_vars - 1:
                        ax.set_xticks([])
                    if j > 0:
                        ax.set_yticks([])
                    # 為最外層添加坐標標籤
                    if i == num_vars - 1:
                        ax.set_xlabel(col_j)
                    if j == 0:
                        ax.set_ylabel(col_i)
        plt.tight_layout()
        plt.show()

    # --- Scatter Matrix ---
    df_plot = df.drop('Solution', axis=1)
    scatter_matrix_with_mean(df_plot, figsize=(12, 12))

