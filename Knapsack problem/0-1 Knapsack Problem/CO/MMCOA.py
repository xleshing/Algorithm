import numpy as np


class Algorithm:

    def __init__(self, turn_node_on, d, value, weight, capacity, coyotes_per_group, n_groups, p_leave, max_iter,
                 max_delay, original_status):
        self.d = d
        self.value = value
        self.weight = weight
        self.capacity = capacity
        self.coyotes_per_group = coyotes_per_group
        self.n_groups = n_groups
        self.p_leave = p_leave
        self.max_iter = max_iter
        self.turn_node_on = turn_node_on
        self.max_delay = max_delay
        self.original_status = original_status

    def func(self, x):
        if np.any(self.weight / np.dot(x, self.value) * 100 > self.capacity):
            return -np.inf
        else:
            return self.weight / np.dot(x, self.value) * 100

    def mmco_initialize_population(self):
        """
        初始化 0/1 背包問題的族群，確保總重量不超過背包容量。

        參數:
        - n_groups: 群體數量
        - coyotes_per_group: 每個群體中的土狼數量
        - D: 問題維度 (物品數量)
        - weights: 每個物品的重量 (1D 陣列, shape = (D,))
        - capacity: 背包的最大容量 (scalar)

        回傳:
        - population: (total_coyotes, D) 的二進制陣列 (0 或 1)，確保不超重
        - groups: (n_groups, coyotes_per_group)，分配群體索引
        - population_age: 每隻土狼的年齡 (全設為 0)
        """
        total_coyotes = self.n_groups * self.coyotes_per_group

        # 隨機初始化 0/1 背包解
        population = np.random.randint(2, size=(total_coyotes, self.d))

        # 計算初始的總重量
        total_value = np.dot(population, self.value)  # total resource after pop some node

        # 重新生成所有超重的解
        for n in range(self.max_delay):
            if np.any([np.sum(abs(coyote) == 0) == self.d for coyote in population]):
                invalid_indices = np.where([np.sum(abs(coyote) == 0) == self.d for coyote in population])[0]  # 找出超重的索引
                population[invalid_indices] = np.random.randint(2, size=(len(invalid_indices), self.d))  # 重新生成
                total_value[invalid_indices] = np.dot(population[invalid_indices], self.value)  # 更新重量
            elif np.any(self.weight / total_value * 100 > self.capacity):
                invalid_indices = np.where(self.weight / total_value * 100 > self.capacity)[0]  # 找出超重的索引
                population[invalid_indices] = np.random.randint(2, size=(len(invalid_indices), self.d))  # 重新生成
                total_value[invalid_indices] = np.dot(population[invalid_indices], self.value)  # 更新重量
            else:
                break

        # 隨機分配群體
        indices = np.random.permutation(total_coyotes)
        groups = indices.reshape(self.n_groups, self.coyotes_per_group)

        # 初始化所有土狼的年齡 (全設為 0)
        population_age = np.zeros(total_coyotes, dtype=int)

        return population, groups, population_age

    def mmco_compute_cultural_tendency(self, sub_pop):
        """
        文化傾向: 以群體中位數作為文化基因 (仍為 0/1)
        """
        return np.round(np.median(sub_pop, axis=0))  # 確保仍為 0 或 1

    def update_coyote(self, i, sub_pop, alpha_coyote, cultural_tendency):
        qj1 = i  # 初始化為自己
        while qj1 == i:  # 當選到自己時，重新選擇
            qj1 = np.random.choice(self.coyotes_per_group)
        qj2 = i  # 初始化為自己
        while qj2 == i:  # 當選到自己時，重新選擇
            qj2 = np.random.choice(self.coyotes_per_group)

        delta1 = alpha_coyote - sub_pop[qj1, :]
        delta2 = cultural_tendency - sub_pop[qj2, :]
        # ka_1 = as_p + np.random.rand() * delta1 + np.random.rand() * delta2
        # ka_1 = np.clip(ka_1, 0, 1).astype(int)

        ka_1 = np.where(np.random.rand(self.d) < 0.5, abs(delta1), abs(delta2))

        # 加權組合兩個影響方向
        # ka_1 = np.round(np.random.rand(D) * abs(delta1) + np.random.rand(D) * abs(delta2)).astype(int)
        # ka_1 = np.clip(ka_1, 0, 1).astype(int)

        return ka_1

    def crossover(self, sub_pop):
        # 選擇雙親
        parents_idx = np.random.choice(self.coyotes_per_group, 2, replace=False)

        # 設定 crossover mask & 突變 mask
        mutation_prob = 1 / self.d
        parent_prob = (1 - mutation_prob) / 2

        # 產生隨機遮罩來決定基因來自父母 1、父母 2 或突變
        pdr = np.random.permutation(self.d)
        p1_mask = np.zeros(self.d, dtype=bool)
        p2_mask = np.zeros(self.d, dtype=bool)

        # 確保至少有 1 個基因來自父母 1，1 個來自父母 2
        p1_mask[pdr[0]] = True
        p2_mask[pdr[1]] = True

        # 其他維度的機率分配
        if self.d > 2:
            rand_vals = np.random.rand(self.d - 2)
            p1_mask[pdr[2:]] = (rand_vals < parent_prob)  # 來自父母 1
            p2_mask[pdr[2:]] = (rand_vals > (1 - parent_prob))  # 來自父母 2

        # 剩下沒被分配的基因 (來自突變)
        mut_mask = ~(p1_mask | p2_mask)

        # 產生 pup (後代)
        pup = (p1_mask * sub_pop[parents_idx[0], :]
               + p2_mask * sub_pop[parents_idx[1], :]
               + mut_mask * np.random.randint(2, size=self.d))  # 突變產生 0 或 1

        return pup

    def mmco_update_group(
            self, population, fitness, group_indices, population_age
    ):
        sub_pop = population[group_indices, :].copy()
        sub_fit = fitness[group_indices].copy()
        sub_age = population_age[group_indices].copy()
        coyotes_per_group = len(group_indices)

        # (1) 找 alpha
        alpha_idx = np.argmin(sub_fit)
        alpha_coyote = sub_pop[alpha_idx, :].copy()

        # (2) 文化傾向
        cultural_tendency = self.mmco_compute_cultural_tendency(sub_pop)

        # (3) 更新: 社會行為
        for i in range(coyotes_per_group):
            ka_1 = self.update_coyote(
                i, sub_pop, alpha_coyote, cultural_tendency
            )
            ka_1_value = np.dot(ka_1, self.value)
            for n in range(self.max_delay):
                if np.sum(abs(ka_1) == 0) == self.d:
                    ka_1 = self.update_coyote(
                        i, sub_pop, alpha_coyote, cultural_tendency
                    )
                    ka_1_value = np.dot(ka_1, self.value)
                elif np.any(self.weight / ka_1_value * 100 > self.capacity):
                    ka_1 = self.update_coyote(
                        i, sub_pop, alpha_coyote, cultural_tendency
                    )
                    ka_1_value = np.dot(ka_1, self.value)
                else:
                    break

            if np.sum(abs(ka_1) == 0) == self.d:
                pass
            elif np.any(self.weight / ka_1_value * 100 > self.capacity):
                pass
            else:
                ka_1_fit = self.func(ka_1)

                if ka_1_fit > sub_fit[i]:
                    sub_pop[i, :] = ka_1
                    sub_fit[i] = ka_1_fit

        # (4) Pup 生產 (Crossover)
        if self.d > 1:
            pup = self.crossover(
                sub_pop
            )
            pup_value = np.dot(pup, self.value)
            for n in range(self.max_delay):
                if np.sum(abs(pup) == 0) == self.d:
                    pup = self.crossover(
                        sub_pop
                    )
                    pup_value = np.dot(pup, self.value)
                elif np.any(self.weight / pup_value * 100 > self.capacity):
                    pup = self.crossover(
                        sub_pop
                    )
                    pup_value = np.dot(pup, self.value)
                else:
                    break

            if np.sum(abs(pup) == 0) == self.d:
                pass
            elif np.any(self.weight / pup_value * 100 > self.capacity):
                pass
            else:
                pup_fit = self.func(pup)

                # 替換最老且最差的個體
                candidate_mask = sub_fit < pup_fit
                if np.any(candidate_mask):
                    candidate_indices = np.where(candidate_mask)[0]
                    oldest_idx = np.argmin(sub_age[candidate_indices])  # 找最老的
                    to_replace = candidate_indices[oldest_idx]

                    sub_pop[to_replace, :] = pup
                    sub_fit[to_replace] = pup_fit
                    sub_age[to_replace] = 0  # 替換後年齡歸 0

        population[group_indices, :] = sub_pop
        fitness[group_indices] = sub_fit
        population_age[group_indices] = sub_age
        return population, fitness, population_age

    def mmco_coyote_exchange(self, groups):
        """
        依機率 p_leave, 隨機抽兩個不同群, 各自隨機選一隻 coyote 互換.
        使族群之間能有基因流動, 類似 COA eq.4.
        備註: 如果要做多次, 可在外圍再加 for 迴圈.
        """
        n_groups, coyotes_per_group = groups.shape
        p_leave = self.p_leave * (self.coyotes_per_group ** 2)

        if n_groups < 2:
            return groups  # 只有 1 群, 無法交換

        # 只做一次嘗試
        if np.random.rand() < p_leave:
            # 選兩個不同群
            g1, g2 = np.random.choice(n_groups, 2, replace=False)
            c1 = np.random.randint(coyotes_per_group)
            c2 = np.random.randint(coyotes_per_group)
            tmp = groups[g1, c1]
            groups[g1, c1] = groups[g2, c2]
            groups[g2, c2] = tmp

        return groups

    def MMCO_main(self):
        # 1) 初始化
        population, groups, population_age = self.mmco_initialize_population()
        fitness = np.array([self.func(coyote) for coyote in population])

        # 2) 找初始最佳
        best_idx = np.argmax(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        convergence = [best_fitness]

        # 3) 迭代
        for iteration in range(self.max_iter):
            # (a) 更新每個群 (含出生)
            for g in range(self.n_groups):
                group_indices = groups[g, :]
                population, fitness, population_age = self.mmco_update_group(
                    population, fitness, group_indices, population_age
                )

            # (b) 群間交換 (脫離狼群)
            groups = self.mmco_coyote_exchange(groups)

            # (c) 年齡更新 (所有土狼年齡 +1)
            population_age += 1

            # (d) 更新全域最佳
            current_best_idx = np.argmax(fitness)
            current_best_fit = fitness[current_best_idx]
            if current_best_fit > best_fitness:
                best_fitness = current_best_fit
                best_solution = population[current_best_idx].copy()

            convergence.append(best_fitness)

        if (self.weight / np.dot(self.original_status, self.value) * 100 < best_fitness <= self.capacity or
                -np.inf != best_fitness <= self.capacity < self.weight / np.dot(self.original_status, self.value) * 100):
            pass
        else:
            best_solution = self.original_status
            best_fitness = self.func(best_solution)  # re-calculate fitness with original status

        return np.array(best_solution), best_fitness, convergence


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    v = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, ]  # each node resource

    w = np.sum([500, 0, 0, 0, 0, 0, 500, 0, 0, 0, ])  # each node resource usage

    c = 55  # SLA

    algorithm = Algorithm(
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
        original_status=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, ]
    )

    best_sol, best_fit, curve = algorithm.MMCO_main()

    print("Best Solution =", best_sol)
    print("Best Fitness  =", best_fit)

    plt.plot(curve)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("MMCO_Enhance with Crossover & Group Exchange")
    plt.grid(True)
    plt.show()
