import numpy as np

# -------------------------------------------------------------------
# 子函式 1) 初始化族群
# -------------------------------------------------------------------
def mmco_initialize_population(n_groups, coyotes_per_group, D, lower_bound, upper_bound):
    total_coyotes = n_groups * coyotes_per_group

    if np.isscalar(lower_bound):
        lower_bound = np.full(D, lower_bound, dtype=float)
    if np.isscalar(upper_bound):
        upper_bound = np.full(D, upper_bound, dtype=float)

    population = lower_bound + np.random.rand(total_coyotes, D) * (upper_bound - lower_bound)

    # 分配群組 (打亂索引後 reshape)
    indices = np.random.permutation(total_coyotes)
    groups = indices.reshape(n_groups, coyotes_per_group)

    # 新增: 初始化所有土狼的年齡 (全設為 0)
    population_age = np.zeros(total_coyotes, dtype=int)

    return population, groups, lower_bound, upper_bound, population_age


# -------------------------------------------------------------------
# 子函式 2) 計算整個 population 的適應度
# -------------------------------------------------------------------
def mmco_evaluate_population(FOBJ, population):
    """
    回傳: fitness (shape=(N,))
    """
    fitness = np.array([FOBJ(ind) for ind in population])
    return fitness


# -------------------------------------------------------------------
# 子函式 3) 計算某群的文化傾向切換中位數
# -------------------------------------------------------------------
def mmco_compute_cultural_tendency(sub_pop):
    return np.median(sub_pop, axis=0)


# -------------------------------------------------------------------
# 子函式 4) 單一群內部行為 (更新 + crossover/pup)
# -------------------------------------------------------------------
def mmco_update_group(
        FOBJ, population, fitness, group_indices,
        lower_bound, upper_bound, D, population_age
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
        as_p = sub_pop[i, :].copy()
        qj1 = i  # 初始化為自己
        while qj1 == i:  # 當選到自己時，重新選擇
            qj1 = np.random.choice(coyotes_per_group)
        qj2 = i  # 初始化為自己
        while qj2 == i:  # 當選到自己時，重新選擇
            qj2 = np.random.choice(coyotes_per_group)
        KA_1 = as_p + np.random.rand() * (alpha_coyote - sub_pop[qj1, :]) + np.random.rand() * (cultural_tendency - sub_pop[qj2, :])
        KA_1 = np.clip(KA_1, lower_bound, upper_bound)
        KA_1_fit = FOBJ(KA_1)
        if KA_1_fit < sub_fit[i]:
            sub_pop[i, :] = KA_1
            sub_fit[i] = KA_1_fit

    # (4) Pup 生產 (Crossover)
    parents_idx = np.random.choice(coyotes_per_group, 2, replace=False)
    p1_rate = p2_rate = (1 - 1 / D) / 2
    pdr = np.random.permutation(D)
    p1_mask = np.zeros(D, dtype=bool)
    p2_mask = np.zeros(D, dtype=bool)
    p1_mask[pdr[0]] = True
    p2_mask[pdr[1]] = True
    if D > 2:
        r = np.random.rand(D - 2)
        p1_mask[pdr[2:]] = (r < p1_rate)
        p2_mask[pdr[2:]] = (r > (1 - p2_rate))
    n_mask = ~(p1_mask | p2_mask)

    pup = (p1_mask * sub_pop[parents_idx[0], :]
           + p2_mask * sub_pop[parents_idx[1], :]
           + n_mask * (lower_bound + np.random.rand(D) * (upper_bound - lower_bound)))

    pup = np.clip(pup, lower_bound, upper_bound)
    pup_fit = FOBJ(pup)

    # 替換最老且最差的個體
    candidate_mask = sub_fit > pup_fit
    if np.any(candidate_mask):
        candidate_indices = np.where(candidate_mask)[0]
        oldest_idx = np.argmax(sub_age[candidate_indices])  # 找最老的
        to_replace = candidate_indices[oldest_idx]

        sub_pop[to_replace, :] = pup
        sub_fit[to_replace] = pup_fit
        sub_age[to_replace] = 0  # 替換後年齡歸 0

    population[group_indices, :] = sub_pop
    fitness[group_indices] = sub_fit
    population_age[group_indices] = sub_age
    return population, fitness, population_age


# -------------------------------------------------------------------
# 子函式 5) 群間交換 (土狼脫離某群 -> 加入另一群)
# -------------------------------------------------------------------
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
              D, lower_bound, upper_bound,
              max_iter,
              p_leave):
    # 1) 初始化
    population, groups, lb, ub, population_age = mmco_initialize_population(
        n_groups, coyotes_per_group, D, lower_bound, upper_bound
    )
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
                FOBJ, population, fitness, group_indices,
                lb, ub, D, population_age
            )

        # (b) 群間交換 (脫離狼群)
        groups = mmco_coyote_exchange(groups, p_leave * (coyotes_per_group ** 2))

        # (c) 年齡更新 (所有土狼年齡 +1)
        population_age += 1

        # (d) 更新全域最佳
        current_best_idx = np.argmin(fitness)
        current_best_fit = fitness[current_best_idx]
        if current_best_fit < best_fitness:
            best_fitness = current_best_fit
            best_solution = population[current_best_idx].copy()

        convergence.append(best_fitness)

    return best_solution, best_fitness, convergence


# --------------------------
# 以下示範如何使用 (main)
# --------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt


    def my_objective(x):
        return np.sum(x ** 2)


    best_sol, best_fit, curve = MMCO_main(
        FOBJ=my_objective,
        n_groups=10,
        coyotes_per_group=10,
        D=30,
        lower_bound=0,
        upper_bound=1,
        max_iter=200,
        p_leave=0.005,  # 群間交換機
    )

    print("Best Solution =", best_sol)
    print("Best Fitness  =", best_fit)

    plt.plot(curve)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("MMCO_Enhance with Crossover & Group Exchange")
    plt.grid(True)
    plt.show()
