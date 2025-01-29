import numpy as np
import matplotlib.pyplot as plt

def initialize_population(lu, n_packs, n_coy):
    """
    初始化族群:
    1. 產生所有 coyote 參數
    2. 建立 packs 結構 (隨機將所有 coyote 分配到各個 pack)
    3. 設定每隻 coyote 年齡(ages)
    4. 回傳 (coyotes, packs, ages, coypack)
    """
    VarMin = lu[0, :]
    VarMax = lu[1, :]
    D = VarMin.size

    # 總體 coyote 數
    pop_total = n_packs * n_coy

    # 隨機產生所有土狼
    coyotes = VarMin + np.random.rand(pop_total, D) * (VarMax - VarMin)
    # 年齡初始化
    ages = np.zeros(pop_total)

    # 產生隨機索引，並分組
    indices = np.random.permutation(pop_total)
    packs = indices.reshape(n_packs, n_coy)

    # 每群 coyote 數量 (此處固定 n_coy)
    coypack = np.full(n_packs, n_coy)

    return coyotes, packs, ages, coypack

def evaluate_population(FOBJ, coyotes):
    """
    計算所有 coyote 的成本(適應度)
    回傳 costs (1-D numpy array)
    """
    pop_total = coyotes.shape[0]
    costs = np.zeros(pop_total)
    for i in range(pop_total):
        costs[i] = FOBJ(coyotes[i, :])
    return costs

def update_pack(FOBJ, coyotes, costs, ages, p, packs, coypack, VarMin, VarMax, Ps, nfeval):
    """
    對第 p 群的 coyote 進行群內行為 (對應 pseudo code 裡的 eq.5, eq.6, eq.12, eq.7等):
      1. 找 alpha
      2. 計算群傾向 (tendency)
      3. 逐隻 coyote 嘗試更新 (社交互動)
      4. 生育 pup，若較佳則取代群中最老的差勁 coyote
    回傳更新後的 coyotes, costs, ages 及 nfeval (增加評估次數)
    """
    D = VarMin.size

    # 取出此群的 coyote 索引
    c_indices = packs[p, :]
    # 擷取當前群資料
    coyotes_aux = coyotes[c_indices, :].copy()
    costs_aux = costs[c_indices].copy()
    ages_aux = ages[c_indices].copy()
    n_coy_aux = coypack[p]

    # (Eq. 5) 根據成本排序，找出 alpha (最低成本者)
    sort_idx = np.argsort(costs_aux)
    costs_aux = costs_aux[sort_idx]
    coyotes_aux = coyotes_aux[sort_idx, :]
    ages_aux = ages_aux[sort_idx]

    # alpha
    c_alpha = coyotes_aux[0, :]

    # (Eq. 6) 群傾向 (每個維度取中位數)
    tendency = np.median(coyotes_aux, axis=0)

    # (Eq. 12) 嘗試更新每一隻 coyote (含 alpha 也可能嘗試，但 alpha 通常最優)
    for c in range(n_coy_aux):
        rc1 = c
        while rc1 == c:
            rc1 = np.random.randint(n_coy_aux)
        rc2 = c
        while rc2 == c or rc2 == rc1:
            rc2 = np.random.randint(n_coy_aux)

        new_c = (coyotes_aux[c, :]
                 + np.random.rand() * (c_alpha - coyotes_aux[rc1, :])
                 + np.random.rand() * (tendency - coyotes_aux[rc2, :]))

        # 邊界處理
        new_c = np.maximum(new_c, VarMin)
        new_c = np.minimum(new_c, VarMax)

        # 評估新成本
        new_cost = FOBJ(new_c)
        nfeval += 1

        # 若變好，則接受新解
        if new_cost < costs_aux[c]:
            costs_aux[c] = new_cost
            coyotes_aux[c, :] = new_c

    # (Eq. 7, Alg. 1) 生育一隻新的 coyote (pup)
    parents = np.random.choice(n_coy_aux, size=2, replace=False)
    prob1 = (1 - Ps) / 2
    prob2 = prob1

    # 隨機打亂維度
    pdr = np.random.permutation(D)
    p1 = np.zeros(D, dtype=bool)
    p2 = np.zeros(D, dtype=bool)

    # 保證至少兩維度分別來自不同父母
    p1[pdr[0]] = True
    p2[pdr[1]] = True

    # 其餘維度用亂數決定是否來自父母
    if D > 2:
        r = np.random.rand(D - 2)
        p1[pdr[2:]] = (r < prob1)
        p2[pdr[2:]] = (r > (1 - prob2))

    # n = 既不來自 p1 也不來自 p2，則隨機在 VarMin ~ VarMax
    n_mask = ~(p1 | p2)

    pup = (p1 * coyotes_aux[parents[0], :]
           + p2 * coyotes_aux[parents[1], :]
           + n_mask * (VarMin + np.random.rand(D) * (VarMax - VarMin)))

    # 評估 pup 成本
    pup_cost = FOBJ(pup)
    nfeval += 1

    # 如果 pup_cost 比群裡某些 coyotes 更優，就取代最老的那一隻
    candidate_mask = (pup_cost < costs_aux)
    if np.any(candidate_mask):
        worst_indices = np.where(candidate_mask)[0]
        # 找出最老的那隻(ages_aux 最大)
        older_idx = np.argsort(ages_aux[worst_indices])[::-1]  # 大到小
        which = worst_indices[older_idx[0]]
        # 用 pup 取代
        coyotes_aux[which, :] = pup
        costs_aux[which] = pup_cost
        ages_aux[which] = 0

    # 把更新後資料放回原結構
    coyotes[c_indices, :] = coyotes_aux
    costs[c_indices] = costs_aux
    ages[c_indices] = ages_aux

    return coyotes, costs, ages, nfeval

def coyote_exchange(p_leave, packs, n_packs, n_coy):
    """
    依機率 p_leave，隨機挑兩個不同的群各自交換一隻 coyote
    """
    if n_packs > 1:
        if np.random.rand() < p_leave:
            rp = np.random.choice(n_packs, size=2, replace=False)
            rc1 = np.random.randint(n_coy)
            rc2 = np.random.randint(n_coy)
            tmp = packs[rp[0], rc1]
            packs[rp[0], rc1] = packs[rp[1], rc2]
            packs[rp[1], rc2] = tmp
    return packs

def COA(FOBJ, lu, nfevalMAX, n_packs, n_coy, max_iter):
    """
    Coyote Optimization Algorithm (COA) 主函式:
      - 使用上述各子函式執行初始化, 評估, 年度迭代更新, 交換等。
      - 回傳最佳解 (GlobalParams), 最佳成本 (GlobalMin), 以及演化紀錄 (crgy)。
    """
    # 參數檢查
    if n_coy < 3:
        raise ValueError("至少需要 3 隻土狼 (n_coy >= 3)")

    # 問題維度
    D = lu.shape[1]
    VarMin = lu[0, :]
    VarMax = lu[1, :]

    # (Eq. 4) 離開群的機率
    p_leave = 0.005 * (n_coy ** 2)
    # pup 基因組合參數
    Ps = 1.0 / D

    # 1) 初始化族群
    coyotes, packs, ages, coypack = initialize_population(lu, n_packs, n_coy)

    # 2) 初始評估
    costs = evaluate_population(FOBJ, coyotes)
    nfeval = len(costs)

    # 3) 設定初始最佳解
    GlobalMin = np.min(costs)
    ibest = np.argmin(costs)
    GlobalParams = coyotes[ibest, :].copy()

    # 演化紀錄
    crgy = []
    year = 0

    # 4) 主要迭代
    # while nfeval < nfevalMAX:
    for iteration in range(max_iter):
        year += 1
        # 針對每個群執行內部行為
        for p in range(n_packs):
            coyotes, costs, ages, _ = update_pack(
                FOBJ, coyotes, costs, ages, p, packs, coypack, VarMin, VarMax, Ps, nfeval
            )

        # (Eq. 4) 交換 coyote (在不同群間)
        packs = coyote_exchange(p_leave, packs, n_packs, n_coy)

        # 年齡 +1
        ages += 1

        # 更新全域最佳解
        current_min = np.min(costs)
        if current_min < GlobalMin:
            GlobalMin = current_min
            ibest = np.argmin(costs)
            GlobalParams = coyotes[ibest, :].copy()

        crgy.append(GlobalMin)

    return GlobalParams, GlobalMin, crgy


# --------------------------
# 以下為範例使用 (main) 示意
# --------------------------
if __name__ == "__main__":
    # 定義目標函式
    def my_objective(x):
        return np.sum(x**2)

    # 問題設定
    D = 30
    lu = np.array([np.zeros(D), np.ones(D)])  # 搜尋空間 [0, 1]^D
    nfevalMAX = 500 * D
    n_packs = 10
    n_coy = 10

    # 執行 COA
    best_params, best_cost, record = COA(my_objective, lu, nfevalMAX, n_packs, n_coy, 200)
    print("最佳解參數 = ", best_params)
    print("最佳成本值 = ", best_cost)

    # 繪製收斂曲線
    plt.plot(record)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Coyote Optimization Algorithm Process')
    plt.grid(True)
    plt.show()
