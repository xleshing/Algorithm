# merge_and_plot_generations.py
import os
import glob
import ast
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

IN_DIR = "generation"
OUT_DIR = "out_generation_merged"

NSGA4_PATTERN = os.path.join(IN_DIR, "NSGA4_2KP*-1A_run*_generations.csv")
NSCO_PATTERN  = os.path.join(IN_DIR, "NSCO_2KP*-1A_run*_generations.csv")

# --------------------------
# 檔名解析：<ALG>_<BASE>_run<r>_generations.csv
# --------------------------
def parse_meta_from_filename(path: str):
    bn = os.path.basename(path)
    m = re.match(r"^(NSGA4|NSCO)_(.+)_run(\d+)_generations\.csv$", bn, re.IGNORECASE)
    if not m:
        alg = bn.split("_")[0]
        base = bn.split("_", 1)[1].split("_run")[0]
        run  = re.findall(r"_run(\d+)_", bn)
        run  = int(run[0]) if run else np.nan
        return alg.upper(), base, run
    alg, base, run = m.group(1).upper(), m.group(2), int(m.group(3))
    return alg, base, run

# --------------------------
# 解析為統一欄位：Generation, X, Y[, Z]
# 支援兩種來源格式（扁平 / sol=list[dict]）
# --------------------------
def flatten_df(df: pd.DataFrame) -> pd.DataFrame:
    # 已扁平
    if 'sol' not in df.columns:
        obj_cols = [c for c in df.columns if c.lower() != 'generation']
        if len(obj_cols) < 2:
            raise ValueError(f"找不到至少兩個目標欄位：{obj_cols}")

        pref3 = ['LoadBalance', 'Average Delay', 'Cost']
        if all(c in df.columns for c in pref3):
            out = df[['Generation', *pref3]].copy()
            out.columns = ['Generation', 'X', 'Y', 'Z']
            return out

        pref3b = ['Obj1', 'Obj2', 'Obj3']
        if all(c in df.columns for c in pref3b):
            out = df[['Generation', *pref3b]].copy()
            out.columns = ['Generation', 'X', 'Y', 'Z']
            return out

        pref2 = ['Obj1', 'Obj2']
        if all(c in df.columns for c in pref2):
            out = df[['Generation', *pref2]].copy()
            out.columns = ['Generation', 'X', 'Y']
            return out

        # 回退：除去 Generation 之外，照出現順序取前 3 或前 2 欄
        obj_cols = [c for c in df.columns if c.lower() != 'generation']
        if len(obj_cols) >= 3:
            out = df[['Generation', obj_cols[0], obj_cols[1], obj_cols[2]]].copy()
            out.columns = ['Generation', 'X', 'Y', 'Z']
            return out
        else:
            out = df[['Generation', obj_cols[0], obj_cols[1]]].copy()
            out.columns = ['Generation', 'X', 'Y']
            return out

    # 舊格式：有 'sol' 欄位（list[dict]）
    rows = []
    for _, row in df.iterrows():
        gen = int(row['Generation'])
        try:
            sol_list = ast.literal_eval(str(row['sol']))
        except Exception:
            continue
        if not isinstance(sol_list, (list, tuple)):
            continue
        for d in sol_list:
            if all(k in d for k in ['LoadBalance', 'Average Delay', 'Cost']):
                rows.append({'Generation': gen,
                             'X': float(d['LoadBalance']),
                             'Y': float(d['Average Delay']),
                             'Z': float(d['Cost'])})
            else:
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

def load_and_flatten_one(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    flat = flatten_df(df)
    alg, base, run = parse_meta_from_filename(path)
    flat.insert(0, "Run", run)
    flat.insert(0, "Base", base)
    flat.insert(0, "Algorithm", alg)
    return flat

# --------------------------
# 圖：每個 (Algorithm, Base) 一張
# --------------------------
def plot_one_group(sub: pd.DataFrame, out_png: str):
    if sub.empty:
        return
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    G = sub["Generation"].to_numpy()
    norm = Normalize(vmin=np.min(G), vmax=np.max(G))
    cmap = plt.cm.viridis

    # 3D
    if "Z" in sub.columns:
        X = sub["X"].to_numpy()
        Y = sub["Y"].to_numpy()
        Z = sub["Z"].to_numpy()
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(X, Y, Z, c=G, cmap=cmap, norm=norm, s=22, edgecolor='k', alpha=0.9)
        ax.set_xlabel("Obj1 / LoadBalance")
        ax.set_ylabel("Obj2 / Average Delay")
        ax.set_zlabel("Obj3 / Cost")
        ttl = f"{sub['Algorithm'].iloc[0]} – {sub['Base'].iloc[0]} (3D, all runs)"
        ax.set_title(ttl)
        cbar = plt.colorbar(sc, ax=ax, pad=0.10)
        cbar.set_label("Generation")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return

    # 2D
    X = sub["X"].to_numpy()
    Y = sub["Y"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(X, Y, c=G, cmap=cmap, norm=norm, s=28, edgecolor='k', alpha=0.9)
    ax.set_xlabel("Obj1 / LoadBalance")
    ax.set_ylabel("Obj2 / Average Delay")
    ttl = f"{sub['Algorithm'].iloc[0]} – {sub['Base'].iloc[0]} (2D, all runs)"
    ax.set_title(ttl)
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Generation")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def main():
    paths = sorted(glob.glob(NSGA4_PATTERN)) + sorted(glob.glob(NSCO_PATTERN))
    if not paths:
        print("找不到任何 generations 檔案，請確認 generation/ 目錄。")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    merged = []
    for p in paths:
        try:
            flat = load_and_flatten_one(p)
            merged.append(flat)
        except Exception as e:
            print(f"[skip] 讀取失敗：{p} -> {e}")

    if not merged:
        print("沒有可合併的資料。")
        return

    df_all = pd.concat(merged, ignore_index=True)

    # 存合併檔
    out_all = os.path.join(OUT_DIR, "all_generations_merged.csv")
    df_all.to_csv(out_all, index=False)
    print(f"[OK] 存檔：{out_all}")

    for alg in ["NSGA4", "NSCO"]:
        sub = df_all[df_all["Algorithm"] == alg]
        if not sub.empty:
            out_alg = os.path.join(OUT_DIR, f"{alg}_generations_merged.csv")
            sub.to_csv(out_alg, index=False)
            print(f"[OK] 存檔：{out_alg}")

    # 逐 (Algorithm, Base) 繪圖
    plot_dir = os.path.join(OUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for (alg, base), sub in df_all.groupby(["Algorithm", "Base"]):
        safe_base = base.replace("/", "_")
        suffix = "3d" if "Z" in sub.columns else "2d"
        out_png = os.path.join(plot_dir, f"{alg}_{safe_base}_{suffix}.png")
        try:
            plot_one_group(sub, out_png)
            print(f"[plot] {alg} {base} -> {out_png}")
        except Exception as e:
            print(f"[plot skip] {alg} {base}: {e}")

if __name__ == "__main__":
    main()
