import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ===== 設定要讀的 CSV 檔 =====
csv_path = 'test/NSCO_2KP500-1A_generations.csv'  # 也可換成 NSCO_2KP100-1A_generations.csv
algo_name = os.path.basename(csv_path).split('_')[0]
print(f"顯示：{csv_path}")

# ===== 讀檔 =====
df = pd.read_csv(csv_path)

# ===== 解析成扁平表格：Generation, X, Y[, Z] =====
def flatten_df(df: pd.DataFrame) -> pd.DataFrame:
    # 1) 新格式：已扁平，可能是 Obj1/Obj2 或 Obj1/Obj2/Obj3 或 LoadBalance/...
    if 'sol' not in df.columns:
        # 先找出除 Generation 外的目標欄
        obj_cols = [c for c in df.columns if c.lower() != 'generation']
        if len(obj_cols) < 2:
            raise ValueError(f"找不到至少兩個目標欄位：{obj_cols}")

        # 優先順序：三目標常見欄名；否則用前兩/三欄
        preferred3 = ['LoadBalance', 'Average Delay', 'Cost']
        if all(c in df.columns for c in preferred3):
            X, Y, Z = preferred3
            out = df[['Generation', X, Y, Z]].copy()
            out.columns = ['Generation', 'X', 'Y', 'Z']
            return out

        preferred3b = ['Obj1', 'Obj2', 'Obj3']
        if all(c in df.columns for c in preferred3b):
            out = df[['Generation', 'Obj1', 'Obj2', 'Obj3']].copy()
            out.columns = ['Generation', 'X', 'Y', 'Z']
            return out

        # 僅兩目標（最常見：Obj1, Obj2）
        preferred2 = ['Obj1', 'Obj2']
        if all(c in df.columns for c in preferred2):
            out = df[['Generation', 'Obj1', 'Obj2']].copy()
            out.columns = ['Generation', 'X', 'Y']
            return out

        # 回退：依照出現順序拿前 3（或 2）欄
        if len(obj_cols) >= 3:
            out = df[['Generation', obj_cols[0], obj_cols[1], obj_cols[2]]].copy()
            out.columns = ['Generation', 'X', 'Y', 'Z']
            return out
        else:
            out = df[['Generation', obj_cols[0], obj_cols[1]]].copy()
            out.columns = ['Generation', 'X', 'Y']
            return out

    # 2) 舊格式：有一欄 sol，裡頭是 list of dicts
    rows = []
    for _, row in df.iterrows():
        gen = int(row['Generation'])
        sol_list = ast.literal_eval(row['sol'])
        for d in sol_list:
            # 先試三目標常見鍵
            if all(k in d for k in ['LoadBalance', 'Average Delay', 'Cost']):
                rows.append({'Generation': gen,
                             'X': float(d['LoadBalance']),
                             'Y': float(d['Average Delay']),
                             'Z': float(d['Cost'])})
            else:
                # 再試 Obj1~3
                if 'Obj3' in d:
                    rows.append({'Generation': gen,
                                 'X': float(d.get('Obj1', 0.0)),
                                 'Y': float(d.get('Obj2', 0.0)),
                                 'Z': float(d.get('Obj3', 0.0))})
                else:
                    rows.append({'Generation': gen,
                                 'X': float(d.get('Obj1', d.get('LoadBalance', 0.0))),
                                 'Y': float(d.get('Obj2', d.get('Average Delay', 0.0)))})
    return pd.DataFrame(rows)

df_plot = flatten_df(df)

# ===== 繪圖（自動 2D / 3D） =====
G = df_plot['Generation'].values
norm = Normalize(vmin=G.min(), vmax=G.max())
cmap = plt.cm.viridis

if 'Z' in df_plot.columns:
    # ---- 3D ----
    X = df_plot['X'].values
    Y = df_plot['Y'].values
    Z = df_plot['Z'].values

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(X, Y, Z, c=G, cmap=cmap, norm=norm, s=40, edgecolor='k', alpha=0.85)

    ax.set_xlabel('Obj1 / LoadBalance')
    ax.set_ylabel('Obj2 / Average Delay')
    ax.set_zlabel('Obj3 / Cost')
    ax.set_title(f'{algo_name} - Generations (3D)')
    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label('Generation')
else:
    # ---- 2D ----
    X = df_plot['X'].values
    Y = df_plot['Y'].values

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(X, Y, c=G, cmap=cmap, norm=norm, s=40, edgecolor='k', alpha=0.9)
    ax.set_xlabel('Obj1 / LoadBalance')
    ax.set_ylabel('Obj2 / Average Delay')
    ax.set_title(f'{algo_name} - Generations (2D)')
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Generation')

plt.tight_layout()
plt.show()
