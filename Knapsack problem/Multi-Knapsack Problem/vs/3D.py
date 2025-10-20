import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 讀檔
csv_path = 'NSCO_generation_solutions.csv'
df = pd.read_csv(csv_path)

# 解析 sol（此時內容已是純 float）
rows = []
for _, row in df.iterrows():
    gen = int(row['Generation'])
    sol_list = ast.literal_eval(row['sol'])  # e.g. [{'LoadBalance':0.1,'Average Delay':2.6,'Cost':2.6}, ...]
    for d in sol_list:
        rows.append({
            'Generation': gen,
            'LoadBalance': float(d['LoadBalance']),
            'AverageDelay': float(d['Average Delay']),
            'Cost': float(d['Cost']),
        })

solutions_df = pd.DataFrame(rows)

# 準備資料
X = solutions_df['LoadBalance'].values
Y = solutions_df['AverageDelay'].values
Z = solutions_df['Cost'].values
G = solutions_df['Generation'].values

# 顏色依世代
norm = Normalize(vmin=G.min(), vmax=G.max())
cmap = plt.cm.viridis

# 繪圖
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(X, Y, Z, c=G, cmap=cmap, norm=norm, s=50, edgecolor='k', alpha=0.85)

ax.set_xlabel('LoadBalance')
ax.set_ylabel('Average Delay')
ax.set_zlabel('Cost')  # <- 修正
ax.set_title('NSCO')

# colorbar 顯示世代
cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('Generation')

plt.tight_layout()
plt.show()