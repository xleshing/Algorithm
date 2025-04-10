import numpy as np
import matplotlib.pyplot as plt


def dominates(u, v):
    """
    判斷多目標向量 u 是否支配 v
    u 支配 v 當且僅當 u 所有目標值均 ≤ v，且至少一個目標值 < v
    """
    return np.all(u <= v) and np.any(u < v)


def fast_non_dominated_sort(pop_objs):
    """
    快速多目標非支配排序 (fast non-dominated sort)

    輸入:
      pop_objs: (N x M) 陣列，N 為解數、M 為目標數
    回傳:
      fronts: 一個 front 列表，每個 front 為包含該 front 中解索引的列表
              第一個 front 為非支配解集合
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
    fronts.pop()  # 移除最後一個空的 front
    return fronts


class NSGA4:
    def __init__(self, FOBJ, VarMin, VarMax, population_size, generations):
        """
        初始化多目標 NSGA4 演算法

        參數:
          FOBJ: 多目標目標函式，傳入一個解向量並回傳目標向量
          VarMin, VarMax: 決策變數下界與上界 (numpy 陣列，維度 D)
          population_size: 種群規模
          generations: 迭代代數
        """
        self.FOBJ = FOBJ
        self.VarMin = np.array(VarMin)
        self.VarMax = np.array(VarMax)
        self.population_size = population_size
        self.generations = generations
        self.D = self.VarMin.size

        # 隨機產生初始種群，每個解為 D 維連續向量
        self.population = self.VarMin + np.random.rand(population_size, self.D) * (self.VarMax - self.VarMin)

    def evaluate_population(self):
        """
        計算全族群每個解的多目標值，回傳形狀為 (population_size x M) 的陣列
        """
        return np.array([self.FOBJ(sol) for sol in self.population])

    def selection(self):
        """
        NSGA4 的選擇操作：
          1. 依多目標非支配排序將種群分層
          2. 將前 50% 的個體標記為 Q1，接續個體標記為 Q2，直到候選集合規模約為 1.5 倍種群規模
          3. 計算候選集合內各解的決策空間距離，進行聚類移除，直到候選集合解數等於種群規模
        回傳新的候選集合（其規模為 population_size）
        """
        pop = self.population.copy()
        fitnesses = self.evaluate_population()
        fronts = fast_non_dominated_sort(fitnesses)

        data_PR = []  # 每個元素為 (解, 區域標記 "Q1" 或 "Q2")
        total = 0
        rank = 0
        # 將各前沿依序加入，前 50% 標記為 Q1
        while rank < len(fronts) and total + len(fronts[rank]) <= 0.5 * self.population_size:
            for idx in fronts[rank]:
                data_PR.append((pop[idx], "Q1"))
            total += len(fronts[rank])
            rank += 1
        # 剩餘的個體標記為 Q2，直到候選集合大小達 1.5 倍種群規模
        while rank < len(fronts) and total < int(1.5 * self.population_size):
            for idx in fronts[rank]:
                if total < int(1.5 * self.population_size):
                    data_PR.append((pop[idx], "Q2"))
                    total += 1
                else:
                    break
            rank += 1

        L = len(data_PR)
        # 計算候選集合中各解決策變數間的歐氏距離矩陣
        distance_matrix = np.zeros((L, L))
        for i in range(L):
            for j in range(i + 1, L):
                d = np.linalg.norm(data_PR[i][0] - data_PR[j][0])
                distance_matrix[i, j] = d
                distance_matrix[j, i] = d

        removed = set()
        # 聚類移除：反覆剔除解直到候選集合數量等於 population_size
        while L - len(removed) > self.population_size:
            min_d = float('inf')
            min_i, min_j = -1, -1
            for i in range(L):
                if i in removed:
                    continue
                for j in range(i + 1, L):
                    if j in removed:
                        continue
                    # 保護 Q1 區域內精英解：若兩解皆屬 Q1，則跳過
                    if data_PR[i][1] == "Q1" and data_PR[j][1] == "Q1":
                        continue
                    if distance_matrix[i, j] < min_d:
                        min_d = distance_matrix[i, j]
                        min_i, min_j = i, j
            # 決定刪除哪個解
            if data_PR[min_i][1] == "Q2" and data_PR[min_j][1] == "Q2":
                # 計算兩個候選解與其他解的最小距離，刪除較「擁擠」的那個
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
        new_population = np.array(new_population)
        # 若不足，則補足（一般狀況下不會發生）
        if new_population.shape[0] < self.population_size:
            deficit = self.population_size - new_population.shape[0]
            new_population = np.vstack([new_population, pop[:deficit]])
        return new_population

    def crossover(self, parent1, parent2):
        """
        算術交叉：以隨機權重線性組合產生兩個子代
        交叉機率設定為 0.9
        """
        if np.random.rand() < 0.9:
            alpha = np.random.rand()
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = alpha * parent2 + (1 - alpha) * parent1
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()
        child1 = np.maximum(child1, self.VarMin)
        child1 = np.minimum(child1, self.VarMax)
        child2 = np.maximum(child2, self.VarMin)
        child2 = np.minimum(child2, self.VarMax)
        return child1, child2

    def mutation(self, solution):
        """
        單維度隨機突變：以機率 0.2 對隨機一個維度賦予新的隨機值
        """
        sol = solution.copy()
        if np.random.rand() < 0.2:
            dim = np.random.randint(0, self.D)
            sol[dim] = self.VarMin[dim] + np.random.rand() * (self.VarMax[dim] - self.VarMin[dim])
        return sol

    def evolve(self):
        """
        NSGA4 演化流程：
          1. 使用 NSGA4 選擇操作從當前族群中挑選候選解
          2. 利用交叉與突變產生後代組成新種群
          3. 記錄每代的 Pareto 前沿（依非支配排序取得第一前沿）
          4. 重複上述過程直到達到最大迭代次數
        回傳最終族群的 Pareto 前沿與演化紀錄（archive）
        """
        archive = []
        for gen in range(self.generations):
            selected = self.selection()
            np.random.shuffle(selected)
            offspring = []
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % len(selected)]
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                offspring.append(child1)
                offspring.append(child2)
            self.population = np.array(offspring[:self.population_size])
            fitnesses = self.evaluate_population()
            fronts = fast_non_dominated_sort(fitnesses)
            pareto_front = self.population[fronts[0], :]
            archive.append(pareto_front)
            print(f"Generation {gen + 1}: Pareto front size = {len(fronts[0])}")
        # 演化結束後取得最終族群之 Pareto 前沿
        fitnesses = self.evaluate_population()
        fronts = fast_non_dominated_sort(fitnesses)
        final_pf = self.population[fronts[0], :]
        return final_pf, archive


# --------------------------
# 以下為範例使用 (main)
# --------------------------
if __name__ == "__main__":
    # 定義多目標測試函式
    # 目標1: f1 = sum(x^2)
    # 目標2: f2 = sum((x - 0.5)^2)
    def my_objectives(x):
        f1 = np.sum(x ** 2)
        f2 = np.sum((x - 0.5) ** 2)
        return np.array([f1, f2])


    # 問題設定
    D = 30
    VarMin = np.zeros(D)
    VarMax = np.ones(D)
    population_size = 50
    generations = 100

    # 執行 NSGA4 演化過程
    nsga4 = NSGA4(my_objectives, VarMin, VarMax, population_size, generations)
    final_pf, archive = nsga4.evolve()

    print("最終 Pareto 前沿解 (NSGA4):")
    print(final_pf)

    # 若要將最終 Pareto 解在目標空間中繪製出來
    pf_obj_values = np.array([my_objectives(sol) for sol in final_pf])
    plt.figure()
    plt.scatter(pf_obj_values[:, 0], pf_obj_values[:, 1], c='red', marker='o', edgecolors='k')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Final Pareto Front (NSGA4)')
    plt.grid(True)
    plt.show()

    # 依據 archive 建立所有解在目標空間的資料，並記錄它們所屬的迭代次數
    all_obj_values = []  # 存放所有解的目標值 (N, 2)
    iter_numbers = []  # 存放各解所屬的迭代次數 (N,)
    for i, pareto_front in enumerate(archive):
        # 計算每個 Pareto 解的目標值
        obj_vals = np.array([my_objectives(sol) for sol in pareto_front])
        all_obj_values.append(obj_vals)
        # 使用 i+1 表示第 i+1 代 (使迭代數從 1 開始)
        iter_numbers.append(np.full(obj_vals.shape[0], i + 1))

    # 合併所有代的資料
    all_obj_values = np.vstack(all_obj_values)
    iter_numbers = np.concatenate(iter_numbers)

    # 畫圖，並使用 colorbar 表示每個解的所屬迭代次數
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(all_obj_values[:, 0], all_obj_values[:, 1],
                     c=iter_numbers, cmap='jet', alpha=0.7, edgecolors='k')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Pareto Fronts')
    plt.grid(True)
    # 建立 colorbar 並加上標籤
    cbar = plt.colorbar(sc)
    cbar.set_label('Iteration')
    plt.show()
