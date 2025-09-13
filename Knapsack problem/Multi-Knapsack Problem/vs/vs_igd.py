import os, glob
import numpy as np
import pandas as pd

# ===== 這三個目標函式要與 NSGA4/NSCO 使用的一致 =====
def objective_latency(solution, usage, item_values, server_capacities):
    load_ratios = usage.flatten()/server_capacities.flatten()
    return np.max(load_ratios)

def objective_cost(solution, usage, item_values, server_capacities):
    load_ratios = usage.flatten()/server_capacities.flatten()
    return np.sum(load_ratios**2)

def objective_resource_utilization(solution, usage, item_values, server_capacities):
    load_ratios = usage.flatten()/server_capacities.flatten()
    return np.std(load_ratios)

OBJECTIVES = [objective_resource_utilization, objective_latency, objective_cost]

# ====== 相對 IGD 小工具（與前面一致） ======
def igd(reference: np.ndarray, approx: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    approx = np.asarray(approx, dtype=float)
    if reference.size == 0:
        return np.nan
    if approx.size == 0:
        return np.inf
    dists = np.linalg.norm(reference[:, None, :] - approx[None, :, :], axis=2)
    return float(np.min(dists, axis=1).mean())

def nondominated_filter(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    N = pts.shape[0]
    keep = np.ones(N, dtype=bool)
    for i in range(N):
        if not keep[i]:
            continue
        # 是否存在其他點嚴格支配 i
        for j in range(N):
            if i == j or not keep[j]:
                continue
            if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                keep[i] = False
                break
    return keep

def build_relative_reference(list_of_fronts, dedup_tol=1e-12):
    if not list_of_fronts:
        return np.empty((0,0))
    stacked = np.vstack([np.asarray(F, dtype=float) for F in list_of_fronts if len(F)])
    if stacked.size == 0:
        return stacked
    scale = 1.0 / max(dedup_tol, 1e-12)
    keys = np.round(stacked * scale).astype(np.int64)
    _, uniq_idx = np.unique(keys, axis=0, return_index=True)
    uniq = stacked[np.sort(uniq_idx)]
    keep = nondominated_filter(uniq)
    return uniq[keep]

def relative_igd_for_algorithms(alg_obj_fronts: dict, normalize=True) -> dict:
    fronts = [v for v in alg_obj_fronts.values() if v is not None and len(v)]
    if not fronts:
        return {k: np.nan for k in alg_obj_fronts.keys()}
    all_points = np.vstack(fronts)
    if normalize:
        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)
        denom = np.where(maxs > mins, maxs - mins, 1.0)
        norm_fronts = {k: (np.asarray(v) - mins) / denom for k, v in alg_obj_fronts.items()}
        reference = build_relative_reference(list(norm_fronts.values()))
        return {k: igd(reference, norm_fronts[k]) for k in alg_obj_fronts.keys()}
    else:
        reference = build_relative_reference(fronts)
        return {k: igd(reference, np.asarray(v)) for k, v in alg_obj_fronts.items()}

# ====== 實驗跑一次：給定 n，回傳 NSGA4/NSCO 的相對 IGD 並存成 CSV ======
def run_once(num_servers, num_requests, population_size, generations,
             nsga4_cls, nsco_cls, outdir=".", seed=1234):

    rng = np.random.default_rng(seed + num_requests)  # 依 n 變動，確保同 n 兩法共用同資料
    item_values = rng.integers(1, 5, size=(num_requests, 1))
    server_caps = rng.integers(50, 61, size=(num_servers, 1))

    # 建演算法物件
    nsga4 = nsga4_cls(num_servers, num_requests, population_size, generations,
                      OBJECTIVES, item_values, server_caps)
    nsco  = nsco_cls (num_servers, num_requests, population_size, generations,
                      OBJECTIVES, item_values, server_caps,
                          coyotes_per_group=5, n_groups=4, p_leave=0.1)

    # 跑最終前緣
    pf_nsga4, _ = nsga4.evolve()
    pf_nsco,  _ = nsco.evolve()

    # 轉成目標值矩陣
    Y_nsga4 = np.array([nsga4.fitness(sol) for sol in pf_nsga4])
    Y_nsco  = np.array([nsco.fitness(sol)  for sol in pf_nsco])

    # 相對 IGD（用兩者合併的非支配作 reference）
    rel_igd = relative_igd_for_algorithms({"NSGA4": Y_nsga4, "NSCO": Y_nsco}, normalize=True)

    # 存檔：每個 n 一個 CSV，兩列（NSGA4/NSCO）
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame([
        {"num_requests": num_requests, "Algorithm": "NSGA4", "RelativeIGD": rel_igd["NSGA4"]},
        {"num_requests": num_requests, "Algorithm": "NSCO",  "RelativeIGD": rel_igd["NSCO"]},
    ])
    outpath = os.path.join(outdir, f"metrics_n{num_requests}.csv")
    df.to_csv(outpath, index=False)
    print(f"Saved: {outpath}")
    return df

if __name__ == "__main__":
    # 你自己的類別（把 import 換成你的模組路徑）
    from NSGA4_LoadBalancing import NSGA4_LoadBalancing as NSGA4
    from NSCO_LoadBalancing import NSCO_LoadBalancing as NSCO

    num_servers    = 15
    generations    = 100
    population     = 20  # 建議讓 NSCO 的 n_groups*coyotes_per_group == population，或在 NSCO 內已自動對齊
    outdir         = "./out_metrics"

    for n in range(10, 21, 5):
        run_once(num_servers, n, population, generations, NSGA4, NSCO, outdir=outdir, seed=42)
