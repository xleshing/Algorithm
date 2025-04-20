import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_objectives(dir_path):
    data = {}
    pattern = re.compile(r'_(\d+)\.csv$')
    for fname in os.listdir(dir_path):
        if fname.endswith('.csv'):
            m = pattern.search(fname)
            if m:
                node = int(m.group(1))
                df = pd.read_csv(os.path.join(dir_path, fname))
                pts = np.column_stack([
                    df['LoadBalance'].values,
                    df['Average Delay'].values,
                    -df['Throughput'].values
                ])
                data[node] = pts
    return data


def pareto_front(points):
    """
    Identify Pareto-efficient points (minimization for all objectives).
    """
    is_efficient = np.ones(points.shape[0], dtype=bool)
    for i, p in enumerate(points):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(points[is_efficient] < p, axis=1)
            is_efficient[i] = True
    return points[is_efficient]


def igd(reference, approx):
    """
    Compute Inverted Generational Distance (IGD):
    average Euclidean distance from each reference point to the nearest point in approx.
    """
    dists = []
    for r in reference:
        dists.append(np.min(np.linalg.norm(approx - r, axis=1)))
    return np.mean(dists)


# Paths to data directories
dir3 = 'NSGA3/nsga3csv/data1'
dir4 = 'NSGA4/nsga4csv/data1'

# Load objectives
nsga3_data = load_objectives(dir3)
nsga4_data = load_objectives(dir4)

# Node counts present in both methods
nodes = sorted(set(nsga3_data.keys()) & set(nsga4_data.keys()))

# Compute IGD for NSGA3 and NSGA4
igd3 = []
igd4 = []
for node in nodes:
    merged = np.vstack([nsga3_data[node], nsga4_data[node]])
    reference = pareto_front(merged)
    igd3.append(igd(reference, nsga3_data[node]))
    igd4.append(igd(reference, nsga4_data[node]))

# Plot IGD vs Node Count
plt.figure()
plt.plot(nodes, igd3, marker='o', label='NSGA3')
plt.plot(nodes, igd4, marker='s', label='NSGA4')
plt.xlabel('Node Count')
plt.ylabel('IGD')
plt.title('IGD Comparison Across Node Counts')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
