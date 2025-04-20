import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # for 3D projection

# -------------------------------
# Step 1. 讀取並解析 CSV 檔案
# -------------------------------
csv_path = 'csv/3D/NSGA4_generation_solutions_data2_100.csv'
df = pd.read_csv(csv_path)

solutions_data = []
for _, row in df.iterrows():
    gen = row['Generation']
    try:
        sol_list = ast.literal_eval(row['sol'])
    except Exception as e:
        print(f"第 {gen} 代解讀錯誤：{e}")
        continue
    for sol in sol_list:
        sol['Generation'] = gen
        solutions_data.append(sol)

solutions_df = pd.DataFrame(solutions_data)

# -------------------------------
# 共用 colormap 與 Normalize
# -------------------------------
gens = solutions_df['Generation'].unique()
norm = mcolors.Normalize(vmin=gens.min(), vmax=gens.max())
cmap = plt.cm.viridis

# -------------------------------
# Step 2. 3D 散點圖
# -------------------------------
fig = plt.figure(figsize=(10, 8))
ax3d = fig.add_subplot(111, projection='3d')
sc3d = ax3d.scatter(
    solutions_df['LoadBalance'],
    solutions_df['Average Delay'],
    solutions_df['Throughput'],
    c=solutions_df['Generation'],
    cmap=cmap,
    norm=norm,
    s=50,
    edgecolor='k'
)
ax3d.set_xlabel('LoadBalance')
ax3d.set_ylabel('Average Delay')
ax3d.set_zlabel('Throughput')
ax3d.set_title('NSGA4 3D Scatter')
cbar3d = fig.colorbar(sc3d, ax=ax3d, pad=0.1)
cbar3d.set_label('Generation')
plt.tight_layout()
plt.show()

# -------------------------------
# Step 3. 各別顯示 2D 投影圖
# -------------------------------
pairs = [
    ('LoadBalance', 'Average Delay', 'Average Delay vs LoadBalance'),
    ('LoadBalance', 'Throughput', 'Throughput vs LoadBalance'),
    ('Average Delay', 'Throughput', 'Throughput vs Average Delay')
]

for x_key, y_key, title in pairs:
    plt.figure(figsize=(6, 5))
    sc2d = plt.scatter(
        solutions_df[x_key],
        solutions_df[y_key],
        c=solutions_df['Generation'],
        cmap=cmap,
        norm=norm,
        s=40,
        edgecolor='k',
        alpha=0.8
    )
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    cbar2d = plt.colorbar(sc2d, label='Generation', pad=0.02)
    plt.tight_layout()
    plt.show()
