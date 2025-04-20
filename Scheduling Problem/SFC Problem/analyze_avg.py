import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    base3 = 'NSGA3/nsga3csv'
    base4 = 'NSGA4/nsga4csv'
    replicates = [f"data{i}" for i in range(1, 11)]

    # 收集每次 replicate 算出的 IGD
    igd3_all = {}
    igd4_all = {}

    for rep in replicates:
        d3 = load_objectives(os.path.join(base3, rep))
        d4 = load_objectives(os.path.join(base4, rep))
        nodes = sorted(set(d3.keys()) & set(d4.keys()))

        for node in nodes:
            merged = np.vstack([d3[node], d4[node]])
            ref = pareto_front(merged)
            igd3_val = igd(ref, d3[node])
            igd4_val = igd(ref, d4[node])
            igd3_all.setdefault(node, []).append(igd3_val)
            igd4_all.setdefault(node, []).append(igd4_val)

    # 計算平均 IGD
    nodes = sorted(igd3_all.keys())
    igd3_mean = [np.mean(igd3_all[n]) for n in nodes]
    igd4_mean = [np.mean(igd4_all[n]) for n in nodes]

    # 繪圖
    plt.figure(figsize=(8,5))
    plt.plot(nodes, igd3_mean, marker='o', label='NSGA3 Avg IGD')
    plt.plot(nodes, igd4_mean, marker='s', label='NSGA4 Avg IGD')
    plt.xlabel('Node Count')
    plt.ylabel('Mean IGD')
    plt.title('NSGA3 vs NSGA4 Avg IGD Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
