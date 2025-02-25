import numpy as np


class NSGA3_Knapsack:
    def __init__(self, num_knapsacks, num_items, population_size, generations, objective_functions, item_values,
                 knapsack_capacities):
        """
        初始化NSGA-III多目標背包問題（物品分配到背包，每個物品只會選擇一個背包）

        參數:
          num_knapsacks: 背包（節點）數量
          num_items: 物品（POD）數量
          population_size: 種群規模
          generations: 迭代代數
          objective_functions: 目標函數列表（數量與目標數量相同），格式為 f(solution, usage, item_values) -> float
          item_values: 每個物品在各目標下的價值，形狀為 (num_items, num_objectives)
          knapsack_capacities: 每個背包在各目標下的容量上限，形狀為 (num_knapsacks, num_objectives)
        """
        self.num_knapsacks = num_knapsacks
        self.num_items = num_items
        self.population_size = population_size
        self.generations = generations
        self.objective_functions = objective_functions
        self.item_values = np.array(item_values)
        self.knapsack_capacities = np.array(knapsack_capacities)
        # 初始種群以可行解生成，保證每個物品只分配到一個背包，且不違反容量約束
        self.population = np.array([self.generate_feasible_solution() for _ in range(population_size)])
        self.fitness_history = []

    def generate_feasible_solution(self):
        """
        生成一個可行解：
          對每個物品（依隨機順序），檢查哪些背包能接納該物品（即該背包目前累計使用值 + 該物品的價值不會超出容量）。
          若存在候選背包，隨機選一個；若沒有則拋出錯誤。
        回傳一個長度為 num_items 的整數向量，代表物品分配的背包編號。
        """
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
        """
        計算每個背包在各目標下的使用值：
          對於每個背包 j，累加所有分配到 j 的物品的價值向量，
          若計算結果違反約束（某背包任一目標使用值超出容量），則刪除此子代，
          並重新初始化一個符合約束的解，再計算並返回該新解的使用值。
        回傳 usage 矩陣，形狀為 (num_knapsacks, num_objectives)
        """
        num_objectives = self.item_values.shape[1]
        usage = np.zeros((self.num_knapsacks, num_objectives))
        for j in range(self.num_knapsacks):
            indices = (solution == j)
            if np.any(indices):
                usage[j] = np.sum(self.item_values[indices], axis=0)
        # 若違反容量約束，則認定該子代無效，重新生成一個可行解
        if np.any(usage > self.knapsack_capacities):
            # 可加上提示訊息
            print("發現不可行子代，重新初始化一個符合約束的解。")
            new_solution = self.generate_feasible_solution()
            return self.compute_objective_usage(new_solution)
        return usage

    def normalize_objectives(self, raw_values):
        if not self.fitness_history:
            return raw_values
        history_array = np.array(self.fitness_history)
        ideal_point = np.min(history_array, axis=0)
        translated_values = raw_values - ideal_point
        extreme_points = np.max(history_array, axis=0)
        intercepts = np.where(extreme_points - ideal_point == 0, 1, extreme_points - ideal_point)
        return translated_values / intercepts

    def fitness(self, solution):
        """
        計算解的適應度：
          1. 利用 compute_objective_usage() 計算各背包在各目標下的使用值
          2. 若有任何背包違反容量約束，compute_objective_usage() 會返回新生成的可行解的使用值
          3. 否則依次呼叫各目標函數計算目標值，並歸一化返回
        """
        usage = self.compute_objective_usage(solution)
        raw_values = np.array(
            [f(solution, usage, self.item_values, self.knapsack_capacities) for f in self.objective_functions])
        self.fitness_history.append(raw_values)
        if len(self.fitness_history) > 100:
            self.fitness_history.pop(0)
        return self.normalize_objectives(raw_values)

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

    def selection(self):
        fitness_values = np.array([self.fitness(sol) for sol in self.population])
        ranks = self.fast_non_dominated_sort(fitness_values)
        sorted_indices = np.argsort(ranks)
        return self.population[sorted_indices[:self.population_size]]

    def crossover(self, parent1, parent2):
        """
        單點交叉：
        隨機選取一個交叉點，前半部來自 parent1，後半部來自 parent2，
        交配後利用修正機制保證子代符合約束。
        """
        if np.random.rand() < 0.9:
            point = np.random.randint(1, self.num_items)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        # 利用 generate_feasible_solution() 修正子代（這裡可直接重新生成一個可行解）
        child1 = self.repair_solution(child1)
        child2 = self.repair_solution(child2)
        return child1, child2

    def mutation(self, solution):
        """
        變異操作：
        隨機選取一個物品，改為分配到另一個隨機背包，並修正以滿足約束。
        """
        sol = solution.copy()
        if np.random.rand() < 0.2:
            idx = np.random.randint(0, self.num_items)
            sol[idx] = np.random.randint(0, self.num_knapsacks)
        sol = self.repair_solution(sol)
        return sol

    def repair_solution(self, solution):
        """
        修正違反約束的解：
          若某背包使用值超過容量，則嘗試將該背包中的物品移到其他能接納的背包。
          若修正失敗，則重新生成一個可行解。
        """
        usage = self.compute_objective_usage(solution)
        if not np.any(usage > self.knapsack_capacities):
            return solution
        # 嘗試修正
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
                # 若無法修正，則重新生成一個可行解
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
        fitness_values = np.array([self.fitness(sol) for sol in self.population])
        ranks = self.fast_non_dominated_sort(fitness_values)
        best_index = np.argmin(ranks)
        return self.population[best_index]


# 以下定義2個目標函數（目標數量為2，因此每個物品有2個價值，每個背包的容量也為2維）
def objective_example1(solution, usage, item_values, knapsack_capacities):
    """
    目標函數1：使各背包使用值儘可能均衡，最小化使用值的標準差
    """
    return np.std(usage / knapsack_capacities)


def objective_example2(solution, usage, item_values, knapsack_capacities):
    """
    目標函數2：最小化所有背包的總使用值
    """
    return np.sum(usage / knapsack_capacities)


if __name__ == "__main__":
    # 參數設定
    num_knapsacks = 5  # 背包數量
    num_items = 50  # 物品（POD）數量
    population_size = 20
    generations = 50

    # 目標數量為2，則每個物品的價值為2維，每個背包的容量也為2維
    num_objectives = 1
    item_values = np.random.randint(1, 2, [num_items, num_objectives])  # shape: (num_items, 2)
    knapsack_capacities = np.random.randint(20, 21, [num_knapsacks, num_objectives])  # shape: (num_knapsacks, 2)

    # 目標函數列表（數量應與目標數量相同）
    objectives = [objective_example1, objective_example2]

    nsga3 = NSGA3_Knapsack(num_knapsacks, num_items, population_size, generations, objectives, item_values,
                           knapsack_capacities)
    best_solution = nsga3.evolve()

    print("最佳解（每個物品的分配向量）：")
    print(best_solution)

    # 計算最佳解下各背包在各目標下的使用值
    best_usage = nsga3.compute_objective_usage(best_solution)
    print("\n各背包在各目標下的使用值：")
    print(best_usage)

    print("\n各背包在各目標下的容量上限：")
    print(knapsack_capacities)

    # 計算並輸出每個背包的每個目標維度佔用率（使用值 / 該背包該目標容量）
    occupancy_rate = best_usage / knapsack_capacities
    print("\n每個背包的每個目標維度佔用率：")
    print(occupancy_rate)

    # # 計算各目標的總使用值及總容量（所有背包累加）
    # total_usage = np.sum(best_usage, axis=0)
    # total_capacity = np.sum(knapsack_capacities, axis=0)
    # usage_ratio = total_usage / total_capacity
    # print("\n各目標總使用值：")
    # print(total_usage)
    # print("\n各目標總容量上限：")
    # print(total_capacity)
    # print("\n各目標使用率（總使用值 / 總容量）：")
    # print(usage_ratio)
    print(np.std(best_usage / knapsack_capacities))
