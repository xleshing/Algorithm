# main.py
import argparse
import os
import math
import numpy as np
import pandas as pd
from typing import List, Tuple

# === 匯入演算法類別 ===
try:
    from ns_algorithms import NSGA4_LoadBalancing, NSCO_LoadBalancing
except Exception:
    pass


# -------------------------
# 讀取 vOptLib UKP .dat 格式
# -------------------------
def read_voptlib_ukp(dat_path: str) -> Tuple[int, int, int, List[np.ndarray], np.ndarray, float]:
    lines = []
    with open(dat_path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)

    it = iter(lines)
    n = int(next(it))
    p = int(next(it))
    k = int(next(it))
    assert k == 1, "UKP 應為單一容量約束（k=1）"

    C = []
    for _ in range(p):
        coeffs = [float(next(it)) for _ in range(n)]
        C.append(np.array(coeffs, dtype=float))

    w = np.array([float(next(it)) for _ in range(n)], dtype=float)
    W = float(next(it))
    return n, p, k, C, w, W


# -------------------------
# 建立目標函式（最大化→最小化）
# -------------------------
def build_objectives_from_dataset(C: List[np.ndarray]):
    objectives = []

    def make_obj_r(r: int):
        def f(solution, usage, item_values, server_capacities):
            chosen = (solution == 0)
            val = float(np.sum(C[r][chosen]))
            return -val
        return f

    for r in range(len(C)):
        objectives.append(make_obj_r(r))
    return objectives


# -------------------------
# 儲存每代前沿 CSV
# -------------------------
def save_generation_results(algo_name, generation_pareto_fronts, ns_algo, p: int, filename=None):
    def _row_from_vals(vals: np.ndarray):
        vals = list(map(float, vals))
        if p == 3:
            return {
                "LoadBalance": vals[0],
                "Average Delay": vals[1],
                "Cost": vals[2],
            }
        else:
            return {f"Obj{i+1}": vals[i] for i in range(p)}

    rows = []
    for gen_idx, front in enumerate(generation_pareto_fronts):
        for sol in front:
            vals = ns_algo.fitness(sol)
            base = {"Generation": gen_idx}
            base.update(_row_from_vals(vals))
            rows.append(base)

    df = pd.DataFrame(rows)
    if filename is None:
        filename = f"{algo_name}_generation_solutions.csv"
    df.to_csv(filename, index=False)
    print(f"[{algo_name}] 已輸出到 {filename}")
    return df


def save_final_front(name, algo, front, p: int, outname: str):
    rows = []
    for sol in front:
        vals = list(map(float, algo.fitness(sol)))
        if p == 3:
            rows.append({
                "LoadBalance": vals[0],
                "Average Delay": vals[1],
                "Cost": vals[2],
            })
        else:
            rows.append({f"Obj{i+1}": vals[i] for i in range(p)})
    pd.DataFrame(rows).to_csv(outname, index=False)
    print(f"[{name}] 最終前沿已輸出到 {outname}")


# -------------------------
# NSCO 族群結構自動分解
# -------------------------
def factor_groups_coyotes(pop: int):
    g = int(math.sqrt(pop))
    while g > 1 and pop % g != 0:
        g -= 1
    if pop % g == 0:
        return g, pop // g
    return pop, 1


# -------------------------
# 主程式
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch run NSGA4/NSCO on multiple UKP datasets")
    parser.add_argument("--data_dir", default="data", help="資料集資料夾 (e.g., data/)")
    parser.add_argument("--pop", type=int, default=50, help="population size")
    parser.add_argument("--gen", type=int, default=100, help="generations")
    parser.add_argument("--repeat", type=int, default=10, help="每個資料集重複次數")
    parser.add_argument("--seed", type=int, default=42, help="初始亂數種子")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # === 指定要跑的 .dat 清單 ===
    dat_files = [f"2KP{i}-1A.dat" for i in range(50, 501, 50)]
    dat_paths = [os.path.join(args.data_dir, f) for f in dat_files]

    for dat_path in dat_paths:
        if not os.path.exists(dat_path):
            print(f"[警告] 找不到檔案：{dat_path}，略過")
            continue

        base = os.path.splitext(os.path.basename(dat_path))[0]
        print(f"\n=== 處理資料集：{base} ===")

        # 讀資料一次（固定內容）
        n, p, k, C, w, W = read_voptlib_ukp(dat_path)
        item_values = w.reshape(-1, 1)
        server_capacities = np.array([[W], [float(np.sum(w) + 1e-6)]], dtype=float)
        objectives = build_objectives_from_dataset(C)

        num_servers = 2
        num_requests = n
        population_size = args.pop
        generations = args.gen
        n_groups, coyotes_per_group = factor_groups_coyotes(population_size)

        for r in range(args.repeat):
            print(f"\n--- 實驗 {r+1}/{args.repeat} ---")
            np.random.seed(args.seed + r)

            # === NSGA4 ===
            nsga4 = NSGA4_LoadBalancing(
                num_servers=num_servers,
                num_requests=num_requests,
                population_size=population_size,
                generations=generations,
                objective_functions=objectives,
                item_values=item_values,
                server_capacities=server_capacities,
                divisions=4
            )
            pareto4, gen_fronts4 = nsga4.evolve()
            save_generation_results(
                "NSGA4",
                gen_fronts4,
                nsga4,
                p=p,
                filename=f"generation/NSGA4_{base}_run{r+1}_generations.csv"
            )
            save_final_front(
                "NSGA4",
                nsga4,
                pareto4,
                p,
                outname=f"front/NSGA4_{base}_run{r+1}_final_front.csv"
            )

            # === NSCO ===
            nsco = NSCO_LoadBalancing(
                num_servers=num_servers,
                num_requests=num_requests,
                population_size=population_size,
                generations=generations,
                objective_functions=objectives,
                item_values=item_values,
                server_capacities=server_capacities,
                coyotes_per_group=coyotes_per_group,
                n_groups=n_groups,
                p_leave=0.1,
                max_delay=200
            )
            paretoC, gen_frontsC = nsco.evolve()
            save_generation_results(
                "NSCO",
                gen_frontsC,
                nsco,
                p=p,
                filename=f"generation/NSCO_{base}_run{r+1}_generations.csv"
            )
            save_final_front(
                "NSCO",
                nsco,
                paretoC,
                p,
                outname=f"front/NSCO_{base}_run{r+1}_final_front.csv"
            )


if __name__ == "__main__":
    main()
