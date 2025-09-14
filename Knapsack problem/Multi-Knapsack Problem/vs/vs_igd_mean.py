import os, glob
import numpy as np
import pandas as pd

# ===== 目標函式（與兩演算法一致） =====
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

# ===== 相對 IGD 小工具 =====
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
        if not keep[i]: continue
        for j in range(N):
            if i == j or not keep[j]: continue
            if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                keep[i] = False
                break
    return keep

def build_relative_reference(list_of_fronts, dedup_tol=1e-12):
    if not list_of_fronts: return np.empty((0,0))
    stacked = np.vstack([np.asarray(F, dtype=float) for F in list_of_fronts if len(F)])
    if stacked.size == 0: return stacked
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


# ====== 固定問題集生成 ======
def generate_benchmark_instances(num_servers, request_list, outdir="./instances", seed=42):
    rng = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)
    instances = {}
    for n in request_list:
        item_values = rng.integers(1, 5, size=(n, 1))
        server_caps = rng.integers(50, 61, size=(num_servers, 1))
        instances[n] = (item_values, server_caps)
        # 存檔方便後續重複使用
        np.savez(os.path.join(outdir, f"instance_n{n}.npz"),
                 item_values=item_values, server_caps=server_caps)
    return instances

def load_instance(n, outdir="./instances"):
    data = np.load(os.path.join(outdir, f"instance_n{n}.npz"))
    return data["item_values"], data["server_caps"]

# ====== 實驗（用固定實例） ======
def run_once(num_servers, num_requests, population_size, generations,
             nsga4_cls, nsco_cls, seed, instance_dir="./instances"):
    # 載入固定實例
    item_values, server_caps = load_instance(num_requests, instance_dir)

    # 建立演算法物件
    nsga4 = nsga4_cls(num_servers, num_requests, population_size, generations,
                      OBJECTIVES, item_values, server_caps)
    nsco  = nsco_cls (num_servers, num_requests, population_size, generations,
                      OBJECTIVES, item_values, server_caps,
                      coyotes_per_group=5, n_groups=4, p_leave=0.1)

    # 演化
    pf_nsga4, _ = nsga4.evolve()
    pf_nsco,  _ = nsco.evolve()

    # 轉目標值
    Y_nsga4 = np.array([nsga4.fitness(sol) for sol in pf_nsga4])
    Y_nsco  = np.array([nsco.fitness(sol)  for sol in pf_nsco])

    # 相對 IGD
    rel_igd = relative_igd_for_algorithms({"NSGA4": Y_nsga4, "NSCO": Y_nsco}, normalize=True)
    return rel_igd

# ====== 主程式 ======
if __name__ == "__main__":
    from NSGA4_LoadBalancing import NSGA4_LoadBalancing as NSGA4
    from NSCO_LoadBalancing import NSCO_LoadBalancing as NSCO

    num_servers  = 15
    generations  = 100
    population   = 20
    outdir       = "./out_metrics_mean"
    trials       = 3
    request_list = range(10, 101, 5)

    # 1. 先生成固定實例
    generate_benchmark_instances(num_servers, request_list, outdir="./instances", seed=42)

    # 2. 跑實驗
    summary_rows = []
    for n in request_list:
        trial_rows = []
        for t in range(trials):
            seed = 1000 + t  # 演算法隨機性
            scores = run_once(num_servers, n, population, generations, NSGA4, NSCO, seed)
            trial_rows.append({"num_requests": n, "trial": t+1, "Algorithm": "NSGA4", "RelativeIGD": scores["NSGA4"]})
            trial_rows.append({"num_requests": n, "trial": t+1, "Algorithm": "NSCO",  "RelativeIGD": scores["NSCO"]})

        df_trials = pd.DataFrame(trial_rows)
        df_trials.to_csv(os.path.join(outdir, f"metrics_n{n}.csv"), index=False)

        g = df_trials.groupby("Algorithm")["RelativeIGD"]
        for alg, s in g:
            summary_rows.append({
                "num_requests": n,
                "Algorithm": alg,
                "RelativeIGD_mean": s.mean(),
                "RelativeIGD_std": s.std(ddof=1)
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(outdir, "metrics_summary.csv"), index=False)
    print("Saved:", os.path.join(outdir, "metrics_summary.csv"))