import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

def load_objectives(dir_path):
    """讀取資料夾內所有 *_<node>.csv，回傳 {node: ndarray(n_solutions, 3)}"""
    data = {}
    pattern = re.compile(r'_(\d+)\.csv$')
    for fname in os.listdir(dir_path):
        if m := pattern.search(fname):
            node = int(m.group(1))
            df = pd.read_csv(os.path.join(dir_path, fname))
            pts = np.column_stack([
                df['LoadBalance'],
                df['Average Delay'],
                -df['Throughput']  # 轉為最小化
            ])
            data.setdefault(node, []).append(pts)
    return {node: np.vstack(pts) for node, pts in data.items()}

def pareto_front(points):
    """回傳 minimization 的 Pareto 前沿"""
    is_eff = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        if not is_eff[i]: continue
        is_eff[np.all(points >= p, axis=1) & np.any(points > p, axis=1)] = False
    return points[is_eff]

def igd(ref, approx):
    """計算 Inverted Generational Distance"""
    return np.mean([np.min(np.linalg.norm(approx - r, axis=1)) for r in ref])

# ========== 主程式 ==========
if __name__ == "__main__":
    base_paths = ["NSCO/CSV", 'NSGA3/nsga3csv', 'NSGA4/nsga4csv']
    replicates = [f"data{i}" for i in range(1, 5)]
    igd_all = [{} for _ in base_paths]

    for rep in replicates:
        d_list = [load_objectives(os.path.join(base, rep)) for base in base_paths]
        common_nodes = sorted(reduce(set.intersection, (set(d.keys()) for d in d_list)))

        for node in common_nodes:
            merged = np.vstack([d[node] for d in d_list])
            ref = pareto_front(merged)
            for i, d in enumerate(d_list):
                igd_all[i].setdefault(node, []).append(igd(ref, d[node]))

    # 計算平均 IGD 並繪圖
    nodes = sorted(igd_all[0])
    igd_mean = [[np.mean(igd_all[i][n]) for n in nodes] for i in range(len(base_paths))]

    plt.figure(figsize=(8, 5))
    for i, mean in enumerate(igd_mean):
        plt.plot(nodes, mean, marker='o', label=f'{base_paths[i].split("/")[0]} Avg IGD')
    plt.xlabel('Node Count')
    plt.ylabel('Mean IGD')
    plt.title('Avg IGD Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
