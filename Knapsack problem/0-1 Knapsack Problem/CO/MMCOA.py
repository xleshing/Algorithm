import numpy as np
from data import Answer


def mmco_initialize_population(n_groups, coyotes_per_group, D, weights, capacity):
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
    total_coyotes = n_groups * coyotes_per_group

    # 隨機初始化 0/1 背包解
    population = np.random.randint(2, size=(total_coyotes, D))

    # 計算初始的總重量
    total_weight = np.dot(population, weights)

    # 重新生成所有超重的解，直到符合約束
    while np.any(total_weight > capacity):
        invalid_indices = np.where(total_weight > capacity)[0]  # 找出超重的索引
        population[invalid_indices] = np.random.randint(2, size=(len(invalid_indices), D))  # 重新生成
        total_weight[invalid_indices] = np.dot(population[invalid_indices], weights)  # 更新重量

    # 隨機分配群體
    indices = np.random.permutation(total_coyotes)
    groups = indices.reshape(n_groups, coyotes_per_group)

    # 初始化所有土狼的年齡 (全設為 0)
    population_age = np.zeros(total_coyotes, dtype=int)

    return population, groups, population_age

def mmco_evaluate_population(FOBJ, population):
    """
    回傳: fitness (shape=(N,))
    """
    fitness = np.array([FOBJ(ind) for ind in population])
    return fitness


def mmco_compute_cultural_tendency(sub_pop):
    """
    文化傾向: 以群體中位數作為文化基因 (仍為 0/1)
    """
    return np.round(np.median(sub_pop, axis=0))  # 確保仍為 0 或 1

def update_coyote(i, coyotes_per_group, sub_pop, alpha_coyote, cultural_tendency, D, FOBJ, sub_fit):
    as_p = sub_pop[i, :].copy()
    qj1 = i  # 初始化為自己
    while qj1 == i:  # 當選到自己時，重新選擇
        qj1 = np.random.choice(coyotes_per_group)
    qj2 = i  # 初始化為自己
    while qj2 == i:  # 當選到自己時，重新選擇
        qj2 = np.random.choice(coyotes_per_group)

    delta1 = alpha_coyote - sub_pop[qj1, :]
    delta2 = cultural_tendency - sub_pop[qj2, :]
    # KA_1 = as_p + np.random.rand() * delta1 + np.random.rand() * delta2
    # KA_1 = np.clip(KA_1, 0, 1).astype(int)

    KA_1 = np.where(np.random.rand(D) < 0.5, abs(delta1), abs(delta2))

    # 加權組合兩個影響方向
    # KA_1 = np.round(np.random.rand(D) * abs(delta1) + np.random.rand(D) * abs(delta2)).astype(int)
    # KA_1 = np.clip(KA_1, 0, 1).astype(int)

    return KA_1

def crossover(coyotes_per_group, D, sub_pop):
    # 選擇雙親
    parents_idx = np.random.choice(coyotes_per_group, 2, replace=False)

    # 設定 crossover mask & 突變 mask
    mutation_prob = 1 / D
    parent_prob = (1 - mutation_prob) / 2

    # 產生隨機遮罩來決定基因來自父母 1、父母 2 或突變
    pdr = np.random.permutation(D)
    p1_mask = np.zeros(D, dtype=bool)
    p2_mask = np.zeros(D, dtype=bool)
    mut_mask = np.zeros(D, dtype=bool)

    # 確保至少有 1 個基因來自父母 1，1 個來自父母 2
    p1_mask[pdr[0]] = True
    p2_mask[pdr[1]] = True

    # 其他維度的機率分配
    if D > 2:
        rand_vals = np.random.rand(D - 2)
        p1_mask[pdr[2:]] = (rand_vals < parent_prob)   # 來自父母 1
        p2_mask[pdr[2:]] = (rand_vals > (1 - parent_prob))  # 來自父母 2

    # 剩下沒被分配的基因 (來自突變)
    mut_mask = ~(p1_mask | p2_mask)

    # 產生 pup (後代)
    pup = (p1_mask * sub_pop[parents_idx[0], :]
           + p2_mask * sub_pop[parents_idx[1], :]
           + mut_mask * np.random.randint(2, size=D))  # 突變產生 0 或 1

    return pup


def mmco_update_group(
        FOBJ, population, fitness, group_indices, D, population_age, weight
):
    sub_pop = population[group_indices, :].copy()
    sub_fit = fitness[group_indices].copy()
    sub_age = population_age[group_indices].copy()
    coyotes_per_group = len(group_indices)

    # (1) 找 alpha
    alpha_idx = np.argmin(sub_fit)
    alpha_coyote = sub_pop[alpha_idx, :].copy()

    # (2) 文化傾向
    cultural_tendency = mmco_compute_cultural_tendency(sub_pop)

    # (3) 更新: 社會行為
    for i in range(coyotes_per_group):
        KA_1 = update_coyote(
            i, coyotes_per_group, sub_pop, alpha_coyote, cultural_tendency, D, FOBJ, sub_fit
        )
        KA_1_weight = np.dot(KA_1, weight)
        while KA_1_weight > capacity:
            KA_1 = update_coyote(
                i, coyotes_per_group, sub_pop, alpha_coyote, cultural_tendency, D, FOBJ, sub_fit
            )
            KA_1_weight = np.dot(KA_1, weight)
        KA_1_fit = FOBJ(KA_1)
        if KA_1_fit > sub_fit[i]:
            sub_pop[i, :] = KA_1
            sub_fit[i] = KA_1_fit

    # (4) Pup 生產 (Crossover)
    pup = crossover(
        coyotes_per_group, D, sub_pop
    )
    pup_weight = np.dot(pup, weight)
    while np.any(pup_weight > capacity):
        pup = crossover(
            coyotes_per_group, D, sub_pop
        )
        pup_weight = np.dot(pup, weight)

    pup_fit = FOBJ(pup)

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

def mmco_coyote_exchange(groups, p_leave):
    """
    依機率 p_leave, 隨機抽兩個不同群, 各自隨機選一隻 coyote 互換.
    使族群之間能有基因流動, 類似 COA eq.4.
    備註: 如果要做多次, 可在外圍再加 for 迴圈.
    """
    n_groups, coyotes_per_group = groups.shape

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


# -------------------------------------------------------------------
# 主函式: MMCO_main
# -------------------------------------------------------------------
def MMCO_main(FOBJ,
              n_groups, coyotes_per_group,
              D,
              weight,
              capacity,
              max_iter,
              p_leave):
    # 1) 初始化
    population, groups, population_age = mmco_initialize_population(n_groups, coyotes_per_group, D, weight, capacity)
    fitness = mmco_evaluate_population(FOBJ, population)

    # 2) 找初始最佳
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    convergence = [best_fitness]

    # 3) 迭代
    for iteration in range(max_iter):
        # (a) 更新每個群 (含出生)
        for g in range(n_groups):
            group_indices = groups[g, :]
            population, fitness, population_age = mmco_update_group(
                FOBJ, population, fitness, group_indices, D, population_age, weight
            )

        # (b) 群間交換 (脫離狼群)
        groups = mmco_coyote_exchange(groups, p_leave * (coyotes_per_group ** 2))

        # (c) 年齡更新 (所有土狼年齡 +1)
        population_age += 1

        # (d) 更新全域最佳
        current_best_idx = np.argmax(fitness)
        current_best_fit = fitness[current_best_idx]
        if current_best_fit > best_fitness:
            best_fitness = current_best_fit
            best_solution = population[current_best_idx].copy()

        convergence.append(best_fitness)

    return best_solution, best_fitness, convergence


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    answer = Answer("p08_c.txt", "p08_p.txt", "p08_w.txt", "p08_s.txt")

    values = answer.answer()[0]

    weight = answer.answer()[1]

    capacity = answer.answer()[2]

    def my_objective(x):
        return np.sum(x * values)

    best_sol, best_fit, curve = MMCO_main(
        FOBJ=my_objective,
        n_groups=10,
        coyotes_per_group=10,
        D=len(values),
        weight=weight,
        capacity=capacity,
        max_iter=100,
        p_leave=0.02,  # 群間交換機
    )

    print("Best Solution =", best_sol)
    print("Best Fitness  =", best_fit)
    print(answer.answer()[3])
    print(np.sum(answer.answer()[3] * values))

    plt.plot(curve)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("MMCO_Enhance with Crossover & Group Exchange")
    plt.grid(True)
    plt.show()


