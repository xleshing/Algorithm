import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

def load_objectives(dir_path):
    """讀取一個 dataX 資料夾內所有 *_<node>.csv，回傳 dict[node] = ndarray((n_solutions,3))."""
    data = {}
    pattern = re.compile(r'_(\d+)\.csv$')
    for fname in os.listdir(dir_path):
        if not fname.endswith('.csv'):
            continue
        m = pattern.search(fname)
        if not m:
            continue
        node = int(m.group(1))
        df = pd.read_csv(os.path.join(dir_path, fname))
        pts = np.column_stack([
            df['LoadBalance'].values,
            df['Average Delay'].values,
            -df['Throughput'].values   # 轉成最小化
        ])
        data.setdefault(node, []).append(pts)
    # 對於同一 node，如果有多個檔案（應該只有一個），直接合併
    return {node: np.vstack(arrays) for node, arrays in data.items()}

def pareto_front(points):
    """回傳 Pareto 前沿 (minimization) 的子集陣列。"""
    is_eff = np.ones(points.shape[0], dtype=bool)
    for i, p in enumerate(points):
        if not is_eff[i]:
            continue
        # 找出支配 p 的點，保留這些點
        better = np.all(points <= p, axis=1) & np.any(points < p, axis=1)
        is_eff[better] = True
        # 找出被 p 支配的點，剔除
        worse = np.all(points >= p, axis=1) & np.any(points > p, axis=1)
        is_eff[worse] = False
    return points[is_eff]

def igd(reference, approx):
    """計算 IGD: 每個 reference 點到 approx 的最短距離取平均。"""
    dists = [np.min(np.linalg.norm(approx - r, axis=1)) for r in reference]
    return np.mean(dists)

# 主程式
if __name__ == "__main__":
    base = ["NSCO/CSV", 'NSGA3/nsga3csv', 'NSGA4/nsga4csv']
    replicates = [f"data{i}" for i in range(1, 5)]
    igd_all = [{} for _ in range(len(base))]

    for rep in replicates:
        d = [load_objectives(os.path.join(base_item, rep)) for base_item in base]
        nodes = sorted(reduce(lambda a, b: a & b, (set(d_item.keys()) for d_item in d)))

        for node in nodes:
            merged = np.vstack([d_item[node] for d_item in d])
            ref = pareto_front(merged)
            igd_val = [igd(ref, d_item[node]) for d_item in d]
            for i in range(len(igd_all)):
                igd_all[i].setdefault(node, []).append(igd_val[i])

    # 計算平均 IGD
    nodes = sorted(igd_all[0].keys())
    igd_mean = [[np.mean(igd_all_item[n]) for n in nodes] for igd_all_item in igd_all]

    # 繪圖
    plt.figure(figsize=(8, 5))
    for i in range(len(base)):
        plt.plot(nodes, igd_mean[i], marker='o', label=f'{base[i].split("/")[0]} Avg IGD')
    plt.xlabel('Node Count')
    plt.ylabel('Mean IGD')
    plt.title('Avg IGD Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
