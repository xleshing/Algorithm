# analyze_avg_runs.py
# 以 run1~run10 先做平均，再用平均後的曲線計算：
# (1) 最終前沿相對 IGD（逐 base、逐法平均 ±std）
# (2) 各 base 的平均收斂曲線（相對 IGD vs 代數，±1σ）
# (3) 90% 收斂速度（Gen@90%）
# (4) AUC（normalized anytime curve）
# (5) 固定預算 FBP（B=10/25/50/100 的 best-so-far IGD）
# (6) 多門檻 FHT（幾何級數門檻，含避零）
#
# 主要輸出：
#   out_metrics_avg/front_mean_igd.csv
#   out_metrics_avg/plots/front_{base}_merged.png
#   out_metrics_avg/plots/IGD_vs_items.png
#   out_convergence_avg/conv_mean_{base}.csv
#   out_convergence_avg/plots/conv_{base}_mean.png
#   out_convergence_avg/convergence_speed_90.csv
#   out_convergence_avg/plots/speed90_vs_items.png
#   out_convergence_avg/plots/speed90_winner.png
#   out_convergence_avg/auc_summary.csv
#   out_convergence_avg/plots/auc_vs_items.png
#   out_convergence_avg/fbp_summary.csv
#   out_convergence_avg/plots/fbp_vs_items.png
#   out_convergence_avg/fht_summary.csv
#   out_convergence_avg/plots/fht_ecdf.png

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===================== IGD 工具（2 目標，容忍空集/NaN） =====================
def igd(reference: np.ndarray, approx: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    approx = np.asarray(approx, dtype=float)
    if reference.size == 0:
        return np.nan
    if approx.size == 0:
        return np.inf
    ref = reference[~np.isnan(reference).any(axis=1)]
    app = approx[~np.isnan(approx).any(axis=1)]
    if ref.size == 0:
        return np.nan
    if app.size == 0:
        return np.inf
    d = np.linalg.norm(ref[:, None, :] - app[None, :, :], axis=2)
    return float(np.min(d, axis=1).mean())


def nondominated_filter(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros((0,), dtype=bool)
    valid = ~np.isnan(pts).any(axis=1)
    pts = pts[valid]
    N = pts.shape[0]
    keep = np.ones(N, dtype=bool)
    for i in range(N):
        if not keep[i]:
            continue
        for j in range(N):
            if i == j or not keep[j]:
                continue
            if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                keep[i] = False
                break
    out = np.zeros(valid.shape[0], dtype=bool)
    out[np.where(valid)[0][keep]] = True
    return out


def build_reference(fronts_list):
    fronts = [np.asarray(F, float) for F in fronts_list if F is not None and len(F)]
    if not fronts:
        return np.empty((0, 2))
    stacked = np.vstack(fronts)
    stacked = stacked[~np.isnan(stacked).any(axis=1)]
    if stacked.size == 0:
        return np.empty((0, 2))
    keys = np.round(stacked / 1e-12).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    uniq = stacked[np.sort(idx)]
    keep = nondominated_filter(uniq)
    return uniq[keep]


def relative_igd(A: np.ndarray, B: np.ndarray, normalize=True) -> dict:
    """回傳 {'NSGA4': igd_A, 'NSCO': igd_B}，參考集以 A、B 合併而成。"""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    non_empty = [x for x in (A, B) if x.size > 0]
    if not non_empty:
        return {"NSGA4": np.nan, "NSCO": np.nan}
    if normalize:
        stacked = np.vstack(non_empty)
        stacked = stacked[~np.isnan(stacked).any(axis=1)]
        mins = stacked.min(axis=0)
        maxs = stacked.max(axis=0)
        denom = np.where(maxs > mins, maxs - mins, 1.0)
        A_ = (A - mins) / denom if A.size else np.empty((0, 2))
        B_ = (B - mins) / denom if B.size else np.empty((0, 2))
        ref = build_reference([A_, B_])
        return {"NSGA4": igd(ref, A_), "NSCO": igd(ref, B_)}
    else:
        ref = build_reference([A, B])
        return {"NSGA4": igd(ref, A), "NSCO": igd(ref, B)}


# ===================== 讀檔 (固定使用 Obj1, Obj2) =====================
def load_front_csv(path: str) -> np.ndarray:
    if not os.path.exists(path):
        return np.empty((0, 2))
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c.lower().startswith("obj")]
    use = [c for c in cols if c.lower() in ("obj1", "obj2")]
    if len(use) < 2:
        use = [c for c in df.columns if c.lower() != "generation"][:2]
        if len(use) < 2:
            return np.empty((0, 2))
    arr = df[use].to_numpy(float)
    return arr.reshape(-1, 2)


def load_generation_csv(path: str) -> dict:
    """回傳 {gen: ndarray(N,2)}"""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if "Generation" not in df.columns:
        return {}
    cols = [c for c in df.columns if c.lower().startswith("obj")]
    use = [c for c in cols if c.lower() in ("obj1", "obj2")]
    if len(use) < 2:
        use = [c for c in df.columns if c.lower() != "generation"][:2]
        if len(use) < 2:
            return {}
    out = {}
    for g, sub in df.groupby("Generation"):
        arr = sub[use].to_numpy(float).reshape(-1, 2)
        out[int(g)] = arr
    return out


# ===================== 檔案索引 =====================
def collect_bases_runs(front_dir="front", gen_dir="generation"):
    bases = {}
    def scan(pattern):
        for p in glob.glob(pattern):
            bn = os.path.basename(p)
            try:
                base = bn.split("_", 1)[1].split("_run")[0]
                run  = int(bn.split("_run")[1].split("_")[0])
            except Exception:
                continue
            bases.setdefault(base, set()).add(run)
    scan(os.path.join(front_dir, "NSGA4_2KP*-1A_run*_final_front.csv"))
    scan(os.path.join(front_dir, "NSCO_2KP*-1A_run*_final_front.csv"))
    scan(os.path.join(gen_dir,   "NSGA4_2KP*-1A_run*_generations.csv"))
    scan(os.path.join(gen_dir,   "NSCO_2KP*-1A_run*_generations.csv"))
    return {b: sorted(list(v)) for b, v in bases.items()}


# ===================== 小工具 =====================
def extract_item_count(base_name: str) -> float:
    """從 base 文字（如 2KP100-1A）解析物品量 100；若無 KP，退而取最長數字串。"""
    m = re.search(r'KP(\d+)', base_name)
    if m:
        return int(m.group(1))
    nums = re.findall(r'(\d+)', base_name)
    return int(max(nums, key=len)) if nums else np.nan


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def best_so_far(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    return np.minimum.accumulate(y)


# ===================== 繪圖 =====================
def plot_front_avg(base, A_pts, B_pts, out_png):
    ensure_dir(out_png)
    plt.figure(figsize=(7, 6))
    if A_pts.size:
        ka = nondominated_filter(A_pts); Ha = A_pts[ka]
        plt.scatter(A_pts[:, 0], A_pts[:, 1], s=22, alpha=0.4, label="NSGA4 (all runs)")
        if len(Ha) >= 2:
            Ha = Ha[np.argsort(Ha[:, 0])]
            plt.plot(Ha[:, 0], Ha[:, 1], '-', lw=1.3, label="NSGA4 ND-edge")
    if B_pts.size:
        kb = nondominated_filter(B_pts); Hb = B_pts[kb]
        plt.scatter(B_pts[:, 0], B_pts[:, 1], s=22, alpha=0.4, label="NSCO (all runs)")
        if len(Hb) >= 2:
            Hb = Hb[np.argsort(Hb[:, 0])]
            plt.plot(Hb[:, 0], Hb[:, 1], '--', lw=1.3, label="NSCO ND-edge")
    plt.xlabel("Obj1"); plt.ylabel("Obj2")
    plt.title(f"Final fronts (all runs merged) – {base}")
    plt.grid(True, alpha=0.25); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()


def plot_mean_convergence(base, df_mean, out_png):
    ensure_dir(out_png)
    plt.figure(figsize=(7, 5))
    for alg, style in [("NSGA4", "-"), ("NSCO", "--")]:
        sub = df_mean[df_mean["Algorithm"] == alg].sort_values("Generation")
        if sub.empty:
            continue
        x = sub["Generation"].values
        y = sub["MeanIGD"].values
        s = sub["StdIGD"].values
        plt.plot(x, y, style, marker='o', label=f"{alg} mean")
        if np.any(np.isfinite(s)):
            plt.fill_between(x, y - s, y + s, alpha=0.15, linewidth=0)
        best = np.nanmin(y)
        if np.isfinite(best):
            plt.axhline(best, linestyle=":", lw=1, label=f"{alg} best={best:.4f}")
    plt.xlabel("Generation"); plt.ylabel("Relative IGD (mean ± 1σ)")
    plt.title(f"Mean convergence over runs – {base}")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()


# ===================== 高階收斂指標 =====================
def fixed_budget_metrics(df_mean: pd.DataFrame, budgets=(10, 25, 50, 100)):
    """回傳 list[dict] 每個 B 對應的 {Base, Algorithm, Budget, IGD}（平均曲線 best-so-far）。"""
    rows = []
    for alg in ["NSGA4", "NSCO"]:
        sub = df_mean[df_mean["Algorithm"] == alg].sort_values("Generation")
        if sub.empty:
            for B in budgets:
                rows.append({"Algorithm": alg, "Budget": B, "IGD": np.nan})
            continue
        x = sub["Generation"].to_numpy(int)
        y = best_so_far(sub["MeanIGD"].to_numpy(float))
        for B in budgets:
            if B in x:
                val = y[np.where(x == B)[0][0]]
            else:
                if B < x.min():
                    val = y[0]
                elif B > x.max():
                    val = y[-1]
                else:
                    i = np.searchsorted(x, B) - 1
                    t = (B - x[i]) / (x[i + 1] - x[i])
                    val = (1 - t) * y[i] + t * y[i + 1]
            rows.append({"Algorithm": alg, "Budget": B, "IGD": float(val)})
    return rows


def auc_anytime_joint(df_mean: pd.DataFrame, eps: float = 1e-12):
    """
    用『原始平均曲線 y_t』（非 best-so-far）計算 AUC，
    用該 base 內「NSGA4 與 NSCO 的 y_t 聯合 min/max」做正規化。
    對 NaN/空序列/零範圍做防護；少於 2 個有效點回傳 NaN。
    回傳 list[dict]: [{'Algorithm':..., 'AUC':...}, ...]
    """
    # 收集兩法的原始平均曲線（僅取有限值）
    curves = {}
    for alg in ["NSGA4", "NSCO"]:
        sub = df_mean[df_mean["Algorithm"] == alg].sort_values("Generation")
        if sub.empty:
            curves[alg] = None
        else:
            y = sub["MeanIGD"].to_numpy(float)
            mask = np.isfinite(y)
            y = y[mask]
            curves[alg] = y if y.size >= 2 else None  # 少於2點，不足以積分

    # 聯合 min/max（只看有效序列）
    ys = [v for v in curves.values() if v is not None and v.size]
    if not ys:
        return [{"Algorithm": "NSGA4", "AUC": np.nan},
                {"Algorithm": "NSCO",  "AUC": np.nan}]

    y_all = np.concatenate(ys)
    if not np.any(np.isfinite(y_all)) or y_all.size < 2:
        return [{"Algorithm": "NSGA4", "AUC": np.nan},
                {"Algorithm": "NSCO",  "AUC": np.nan}]

    ymax = float(np.nanmax(y_all))
    ymin = float(np.nanmin(y_all))
    denom = max(ymax - ymin, eps)  # 防 0

    rows = []
    for alg in ["NSGA4", "NSCO"]:
        y = curves.get(alg)
        if y is None or y.size < 2:
            rows.append({"Algorithm": alg, "AUC": np.nan})
            continue
        # 轉成『品質越大越好』的 z_t，再做等步長梯形積分
        z = (ymax - y) / denom
        z = np.clip(z, 0.0, 1.0)  # 保險
        auc = float(np.trapezoid(z, dx=1.0))
        rows.append({"Algorithm": alg, "AUC": auc})
    return rows

def safe_geom_targets(ymin: float, ymax: float, n=8, eps=1e-12):
    """建立避免 0 的幾何級數門檻（由較易→較難）。"""
    lo = max(min(ymin, ymax), eps)
    hi = max(max(ymin, ymax), eps)
    # 讓 start 靠近較難（接近 hi），end 靠近較易（接近 lo）
    start_val = max(0.9 * hi + 0.1 * lo, eps)
    end_val = max(1.02 * lo, eps)
    if end_val >= start_val:
        # 防止相等或反向
        start_val *= 1.1
        end_val *= 0.9
    return np.geomspace(start_val, end_val, n)


def fht_multitarget(df_mean: pd.DataFrame, targets: np.ndarray):
    """
    多門檻 first-hitting time：回傳 list[dict]: {Algorithm, Target, Gen}
    使用平均曲線 best-so-far IGD。
    """
    rows = []
    for alg in ["NSGA4", "NSCO"]:
        sub = df_mean[df_mean["Algorithm"] == alg].sort_values("Generation")
        if sub.empty:
            for T in targets:
                rows.append({"Algorithm": alg, "Target": float(T), "Gen": np.nan})
            continue
        x = sub["Generation"].to_numpy(int)
        y = best_so_far(sub["MeanIGD"].to_numpy(float))
        for T in targets:
            idx = np.where(y <= T)[0]
            gen = int(x[idx[0]]) if len(idx) else np.nan
            rows.append({"Algorithm": alg, "Target": float(T), "Gen": gen})
    return rows


# ===================== 主分析 =====================
def main():
    FRONT_DIR = "front"
    GEN_DIR   = "generation"
    OUT_MET   = "out_metrics_avg"
    OUT_GEN   = "out_convergence_avg"
    NORMALIZE = True
    FBP_BUDGETS = (10, 25, 50, 100)

    os.makedirs(OUT_MET, exist_ok=True)
    os.makedirs(OUT_GEN, exist_ok=True)

    base_runs = collect_bases_runs(FRONT_DIR, GEN_DIR)
    if not base_runs:
        print("找不到檔案，請確認 front/ 與 generation/ 內容。")
        return

    # ---------- 1) final-front：先逐 run 計算 IGD，再做平均；畫合併點雲 ----------
    front_summary_rows = []
    for base, runs in sorted(base_runs.items()):
        igds4, igdsc = [], []
        mergeA, mergeB = [], []
        for r in runs:
            f4 = os.path.join(FRONT_DIR, f"NSGA4_{base}_run{r}_final_front.csv")
            fc = os.path.join(FRONT_DIR, f"NSCO_{base}_run{r}_final_front.csv")
            A = load_front_csv(f4); B = load_front_csv(fc)
            if A.size: mergeA.append(A)
            if B.size: mergeB.append(B)
            if A.size == 0 and B.size == 0:
                continue
            m = relative_igd(A, B, normalize=NORMALIZE)
            igds4.append(m["NSGA4"]); igdsc.append(m["NSCO"])
        if igds4 or igdsc:
            front_summary_rows += [
                {"Base": base, "Algorithm": "NSGA4",
                 "MeanIGD": np.nanmean(igds4) if igds4 else np.nan,
                 "StdIGD":  np.nanstd(igds4, ddof=1) if len(igds4) > 1 else np.nan,
                 "Runs":    len(igds4)},
                {"Base": base, "Algorithm": "NSCO",
                 "MeanIGD": np.nanmean(igdsc) if igdsc else np.nan,
                 "StdIGD":  np.nanstd(igdsc, ddof=1) if len(igdsc) > 1 else np.nan,
                 "Runs":    len(igdsc)}
            ]
            # 合併點雲圖（視覺參考）
            A_all = np.vstack(mergeA) if mergeA else np.empty((0, 2))
            B_all = np.vstack(mergeB) if mergeB else np.empty((0, 2))
            plot_front_avg(base, A_all, B_all,
                           os.path.join(OUT_MET, "plots", f"front_{base}_merged.png"))

    if front_summary_rows:
        df_front = pd.DataFrame(front_summary_rows)
        df_front.to_csv(os.path.join(OUT_MET, "front_mean_igd.csv"), index=False)
        print(f"[front] 平均 IGD 已輸出：{os.path.join(OUT_MET, 'front_mean_igd.csv')}")

        # IGD vs items（平均 ±1σ）
        ensure_dir(os.path.join(OUT_MET, "plots", "IGD_vs_items.png"))
        fig = plt.figure(figsize=(8, 5))
        df_front["Items"] = df_front["Base"].apply(extract_item_count)
        df_front = df_front.dropna(subset=["Items"])
        for alg, (ls, mk) in [("NSGA4", ("-", "o")), ("NSCO", ("--", "s"))]:
            sub = df_front[df_front["Algorithm"] == alg].sort_values("Items")
            if sub.empty:
                continue
            x = sub["Items"].values
            y = sub["MeanIGD"].values
            plt.plot(x, y, linestyle=ls, marker=mk, label=f"{alg} mean IGD")
            if "StdIGD" in sub.columns and np.any(np.isfinite(sub["StdIGD"].values)):
                plt.fill_between(x, y - sub["StdIGD"].values, y + sub["StdIGD"].values, alpha=0.15)
        plt.xlabel("Number of Items")
        plt.ylabel("Relative IGD (mean ± 1σ)")
        plt.title("Mean Relative IGD vs Item Count")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_MET, "plots", "IGD_vs_items.png"), dpi=150)
        plt.close(fig)

    # ---------- 2) generation：逐 run、逐代算 IGD → 同代平均；輸出平均曲線 ----------
    conv90_rows = []   # Gen@90%
    auc_rows = []      # AUC
    fbp_rows = []      # FBP rows
    fht_rows_total = []  # FHT rows

    for base, runs in sorted(base_runs.items()):
        curves = {"NSGA4": {}, "NSCO": {}}
        gens_all = set()
        for r in runs:
            g4 = os.path.join(GEN_DIR, f"NSGA4_{base}_run{r}_generations.csv")
            gc = os.path.join(GEN_DIR, f"NSCO_{base}_run{r}_generations.csv")
            G4 = load_generation_csv(g4); GC = load_generation_csv(gc)
            gens = sorted(set(G4.keys()) | set(GC.keys()))
            if not gens:
                continue
            for g in gens:
                A = G4.get(g, np.empty((0, 2)))
                B = GC.get(g, np.empty((0, 2)))
                m = relative_igd(A, B, normalize=NORMALIZE)
                curves["NSGA4"].setdefault(g, []).append(m["NSGA4"])
                curves["NSCO"].setdefault(g, []).append(m["NSCO"])
                gens_all.add(g)

        if not gens_all:
            continue

        rows = []
        for alg in ["NSGA4", "NSCO"]:
            for g in sorted(gens_all):
                arr = np.array(curves[alg].get(g, []), float)
                rows.append({
                    "Base": base, "Algorithm": alg, "Generation": g,
                    "MeanIGD": np.nanmean(arr) if arr.size else np.nan,
                    "StdIGD":  np.nanstd(arr, ddof=1) if arr.size > 1 else np.nan,
                    "N": int(arr.size)
                })
        df_mean = pd.DataFrame(rows).dropna(subset=["MeanIGD"])
        df_mean.to_csv(os.path.join(OUT_GEN, f"conv_mean_{base}.csv"), index=False)

        # 平均曲線圖（±1σ）
        plot_mean_convergence(base, df_mean, os.path.join(OUT_GEN, "plots", f"conv_{base}_mean.png"))

        # Gen@90%（使用累積最小）
        for alg in ["NSGA4", "NSCO"]:
            sub = df_mean[df_mean["Algorithm"] == alg].sort_values("Generation")
            if sub.empty:
                conv90_rows.append({"Base": base, "Algorithm": alg, "Gen@90%": np.nan})
                continue
            x = sub["Generation"].to_numpy(int)
            y = best_so_far(sub["MeanIGD"].to_numpy(float))
            start, end = y[0], np.nanmin(y)
            drop = start - end
            if not np.isfinite(drop) or drop <= 0:
                conv90_rows.append({"Base": base, "Algorithm": alg, "Gen@90%": np.nan})
                continue
            target = end + 0.10 * drop
            idx = np.where(y <= target)[0]
            gen90 = int(x[idx[0]]) if len(idx) else np.nan
            conv90_rows.append({"Base": base, "Algorithm": alg, "Gen@90%": gen90})

        # AUC（normalized anytime curve）
        for row in auc_anytime_joint(df_mean):
            row["Base"] = base
            auc_rows.append(row)

        # FBP（固定預算）
        for row in fixed_budget_metrics(df_mean, FBP_BUDGETS):
            row["Base"] = base
            fbp_rows.append(row)

        # FHT（多門檻）：以兩法的 best-so-far 合併，產生避免 0 的幾何級數門檻
        y_all = []
        for alg in ["NSGA4", "NSCO"]:
            sub = df_mean[df_mean["Algorithm"] == alg].sort_values("Generation")
            if not sub.empty:
                y_all.append(best_so_far(sub["MeanIGD"].to_numpy(float)))
        if y_all:
            yy = np.concatenate(y_all)
            ymin, ymax = float(np.nanmin(yy)), float(np.nanmax(yy))
            targets = safe_geom_targets(ymin, ymax, n=8)  # 避免零
            for row in fht_multitarget(df_mean, targets):
                row["Base"] = base
                fht_rows_total.append(row)

    # === 輸出：Gen@90% ===
    if conv90_rows:
        df_conv90 = pd.DataFrame(conv90_rows)
        out_csv = os.path.join(OUT_GEN, "convergence_speed_90.csv")
        df_conv90.to_csv(out_csv, index=False)
        print(f"[gen] 90% 收斂速度已輸出：{out_csv}")

        # 視覺化（兩法對齊 x；缺值以 NR 標記）
        df_speed = df_conv90.copy()
        df_speed["Items"] = df_speed["Base"].apply(extract_item_count)
        df_speed = df_speed.dropna(subset=["Items"])
        all_items = np.sort(df_speed["Items"].unique())

        fig, ax = plt.subplots(figsize=(9, 6))
        for alg, (ls, mk) in [("NSGA4", ("-", "o")), ("NSCO", ("--", "s"))]:
            sub = df_speed[df_speed["Algorithm"] == alg]
            y_map = dict(zip(sub["Items"], sub["Gen@90%"]))
            y = np.array([y_map.get(x, np.nan) for x in all_items], dtype=float)
            mask = np.isfinite(y)
            if np.any(mask):
                ax.plot(all_items[mask], y[mask], linestyle=ls, marker=mk, label=alg)
            miss = ~mask
            if np.any(miss):
                ymax = np.nanmax(y) if np.any(mask) else 1.0
                y_nr = np.full(miss.sum(), ymax * 1.05)
                ax.scatter(all_items[miss], y_nr, marker='x', color='gray', zorder=3)
                for xi, yi in zip(all_items[miss], y_nr):
                    ax.annotate("NR", (xi, yi), xytext=(3, 3),
                                textcoords="offset points", fontsize=8, color='gray')
        ax.set_xlabel("Number of Items")
        ax.set_ylabel("Generations to 90% convergence")
        ax.set_title("Convergence Speed (90%) vs Item Count")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ensure_dir(os.path.join(OUT_GEN, "plots", "speed90_vs_items.png"))
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_GEN, "plots", "speed90_vs_items.png"), dpi=150)
        plt.close(fig)

        # 贏家視覺化
        winners = []
        for base in sorted(df_speed["Base"].unique(), key=extract_item_count):
            s = df_speed[df_speed["Base"] == base]
            g4s = s[s["Algorithm"] == "NSGA4"]["Gen@90%"].values
            gcs = s[s["Algorithm"] == "NSCO"]["Gen@90%"].values
            g4 = g4s[0] if len(g4s) else np.nan
            gc = gcs[0] if len(gcs) else np.nan
            if np.isfinite(g4) and (not np.isfinite(gc) or g4 < gc):
                winners.append((extract_item_count(base), g4, "NSGA4"))
            elif np.isfinite(gc) and (not np.isfinite(g4) or gc < g4):
                winners.append((extract_item_count(base), gc, "NSCO"))
            else:
                best = np.nanmin([g4, gc])
                winners.append((extract_item_count(base), best, "TIE"))

        if winners:
            items = [w[0] for w in winners]
            gens  = [w[1] for w in winners]
            labs  = [w[2] for w in winners]
            colors = {"NSGA4": "#1f77b4", "NSCO": "#ff7f0e", "TIE": "#7f7f7f"}

            fig = plt.figure(figsize=(9, 6))
            plt.scatter(items, gens, s=60,
                        c=[colors[l] for l in labs],
                        edgecolors='k', linewidths=0.6)
            for i, lab in enumerate(labs):
                if not np.isfinite(gens[i]):
                    continue
                plt.annotate(lab, (items[i], gens[i]),
                             xytext=(5, 5), textcoords="offset points", fontsize=8)
            plt.xlabel("Number of Items")
            plt.ylabel("Generations to 90% convergence (winner)")
            plt.title("Winner (90% Convergence) vs Item Count")
            plt.grid(True, alpha=0.3)
            ensure_dir(os.path.join(OUT_GEN, "plots", "speed90_winner.png"))
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_GEN, "plots", "speed90_winner.png"), dpi=150)
            plt.close(fig)

    # === 輸出：AUC ===
    if auc_rows:
        df_auc = pd.DataFrame(auc_rows)
        df_auc.to_csv(os.path.join(OUT_GEN, "auc_summary.csv"), index=False)

        # 轉 Items，過濾 AUC/Items 的 NaN
        df_auc["Items"] = df_auc["Base"].apply(extract_item_count)
        df_auc = df_auc[np.isfinite(df_auc["Items"])]
        df_auc = df_auc[np.isfinite(df_auc["AUC"])]

        fig = plt.figure(figsize=(8, 5))
        plotted_any = False
        for alg, (ls, mk) in [("NSGA4", ("-", "o")), ("NSCO", ("--", "s"))]:
            sub = df_auc[df_auc["Algorithm"] == alg].sort_values("Items")
            if sub.empty:
                continue
            plt.plot(sub["Items"].values, sub["AUC"].values,
                     linestyle=ls, marker=mk, label=alg)
            plotted_any = True

        plt.xlabel("Number of Items")
        plt.ylabel("AUC of normalized anytime curve (↑ better)")
        plt.title("AUC vs Item Count")
        plt.grid(True, alpha=0.3)
        if plotted_any:
            plt.legend()
        else:
            print("[warn] AUC: 沒有任何有效點可畫（請檢查 conv_mean_* 是否有資料）")

        out_png = os.path.join(OUT_GEN, "plots", "auc_vs_items.png")
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

    # === 輸出：FBP ===
    if fbp_rows:
        df_fbp = pd.DataFrame(fbp_rows)
        df_fbp.to_csv(os.path.join(OUT_GEN, "fbp_summary.csv"), index=False)
        # FBP 圖：對每個 Budget 畫 NSGA4/NSCO 兩條線
        df_fbp["Items"] = df_fbp["Base"].apply(extract_item_count)
        df_fbp = df_fbp.dropna(subset=["Items"])
        budgets = sorted(df_fbp["Budget"].unique())
        fig = plt.figure(figsize=(10, 6))
        for B in budgets:
            for alg, (ls, mk) in [("NSGA4", ("-", "o")), ("NSCO", ("--", "s"))]:
                sub = df_fbp[(df_fbp["Algorithm"] == alg) & (df_fbp["Budget"] == B)].sort_values("Items")
                if sub.empty:
                    continue
                plt.plot(sub["Items"], sub["IGD"], linestyle=ls, marker=mk, label=f"{alg} B={B}")
        plt.xlabel("Number of Items")
        plt.ylabel("Best-so-far IGD @ Budget")
        plt.title("Fixed-Budget Performance vs Item Count")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2)
        ensure_dir(os.path.join(OUT_GEN, "plots", "fbp_vs_items.png"))
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_GEN, "plots", "fbp_vs_items.png"), dpi=150)
        plt.close(fig)

    # === 輸出：FHT 多門檻 ===
    if fht_rows_total:
        df_fht = pd.DataFrame(fht_rows_total)
        df_fht.to_csv(os.path.join(OUT_GEN, "fht_summary.csv"), index=False)
        # ECDF 圖：合併所有 base，對每個門檻畫命中機率曲線（平均視角）
        fig = plt.figure(figsize=(10, 6))
        targets = sorted(df_fht["Target"].unique(), reverse=True)  # 從難到易
        for alg, (ls, mk) in [("NSGA4", ("-", "o")), ("NSCO", ("--", "s"))]:
            # 將所有 base 的 Gen 合併做 ECDF（代數越小越好）
            gens = df_fht[df_fht["Algorithm"] == alg]["Gen"].dropna().to_numpy(dtype=float)
            if gens.size == 0:
                continue
            xs = np.sort(gens)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            plt.step(xs, ys, where="post", linestyle=ls, marker=mk, label=alg)
        plt.xlabel("Generations (first-hitting time across thresholds/bases)")
        plt.ylabel("ECDF (hit probability)")
        plt.title("ECDF of First-Hitting Time (aggregated)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        ensure_dir(os.path.join(OUT_GEN, "plots", "fht_ecdf.png"))
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_GEN, "plots", "fht_ecdf.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
