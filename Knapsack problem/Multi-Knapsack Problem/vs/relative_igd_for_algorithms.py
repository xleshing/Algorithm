import numpy as np

# --- 基礎：IGD（向量化） ---
def igd(reference: np.ndarray, approx: np.ndarray) -> float:
    """
    Inverted Generational Distance:
    平均每個 reference 點到 approx 集合的最近歐氏距離。
    reference: [N_ref, M]
    approx:    [N_alg, M]
    """
    reference = np.asarray(reference, dtype=float)
    approx = np.asarray(approx, dtype=float)
    if reference.size == 0:
        return np.nan
    if approx.size == 0:
        return np.inf
    dists = np.linalg.norm(reference[:, None, :] - approx[None, :, :], axis=2)  # [N_ref, N_alg]
    return float(np.min(dists, axis=1).mean())

# --- 工具：非支配過濾（最小化） ---
def nondominated_filter(points: np.ndarray) -> np.ndarray:
    """
    回傳非支配點的布林遮罩（True=保留）。O(N^2) 實作，N 中等可用。
    points: [N, M]
    """
    pts = np.asarray(points, dtype=float)
    N = pts.shape[0]
    keep = np.ones(N, dtype=bool)
    for i in range(N):
        if not keep[i]:
            continue
        # 若存在 j 支配 i，則丟掉 i
        dominated_i = np.any((pts <= pts[i]).all(axis=1) & (pts < pts[i]).any(axis=1))
        # 上面也會把自己判定進去，所以要排除自己
        dominated_i = dominated_i and not ((pts[i] <= pts[i]).all() and (pts[i] < pts[i]).any())
        if dominated_i:
            keep[i] = False
    return keep

# --- 工具：建立「相對」reference（合併 + 去重 + 非支配） ---
def build_relative_reference(list_of_fronts, dedup_tol=1e-12):
    """
    list_of_fronts: list[np.ndarray]，每個是某演算法的 Pareto 目標值矩陣 [Ni, M]
    流程：縱向堆疊 -> 近似去重 -> 非支配過濾
    """
    if not list_of_fronts:
        return np.empty((0, 0))
    stacked = np.vstack([np.asarray(F, dtype=float) for F in list_of_fronts if len(F)])
    if stacked.size == 0:
        return stacked

    # 近似去重（避免重複點影響效率）
    # 用四捨五入到一定小數位的方式，等價群組後取唯一
    # 你也可以改成用 KDTree/哈希等更精細作法
    scale = 1.0 / max(dedup_tol, 1e-12)
    keys = np.round(stacked * scale).astype(np.int64)
    _, uniq_idx = np.unique(keys, axis=0, return_index=True)
    uniq = stacked[np.sort(uniq_idx)]

    # 非支配過濾
    keep = nondominated_filter(uniq)
    return uniq[keep]

# --- 主函式：計算多演算法的「相對 IGD」 ---
def relative_igd_for_algorithms(alg_obj_fronts: dict,
                                normalize=False) -> dict:
    """
    alg_obj_fronts: {name: obj_front}，obj_front 為 [N, M] 的目標值矩陣（全為最小化）。
      例：{"NSGA4": Y_nsga4, "NSCO": Y_nsco}
          其中 Y_nsga4 = np.array([nsga4.fitness(sol) for sol in pf_nsga4])
               Y_nsco  = np.array([nsco.fitness(sol)  for sol in pf_nsco])

    normalize: 是否對所有點做 min-max normalization（跨尺度目標建議開啟）

    回傳: {name: igd_value}
    """
    # 建 reference（相對）：合併所有前沿後取非支配集
    fronts = [v for v in alg_obj_fronts.values() if v is not None and len(v)]
    if not fronts:
        return {k: np.nan for k in alg_obj_fronts.keys()}

    all_points = np.vstack(fronts)
    if normalize:
        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)
        denom = np.where(maxs > mins, maxs - mins, 1.0)
        # 正規化各集合
        norm_fronts = {k: (np.asarray(v) - mins) / denom for k, v in alg_obj_fronts.items()}
        reference = build_relative_reference(list(norm_fronts.values()))
        return {k: igd(reference, norm_fronts[k]) for k in alg_obj_fronts.keys()}
    else:
        reference = build_relative_reference(fronts)
        return {k: igd(reference, np.asarray(v)) for k, v in alg_obj_fronts.items()}
