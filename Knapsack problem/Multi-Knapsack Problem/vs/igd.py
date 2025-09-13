import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_all_metrics(indir="./out_metrics"):
    files = sorted(glob.glob(os.path.join(indir, "metrics_n*.csv")))
    if not files:
        raise FileNotFoundError("找不到任何 metrics_n*.csv，先跑完批次實驗再來聚合～")
    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    # 確保排序
    df = df.sort_values(["Algorithm", "num_requests"]).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = load_all_metrics("./out_metrics")
    # 標準化欄位型別
    df["num_requests"] = df["num_requests"].astype(int)
    df["RelativeIGD"]  = df["RelativeIGD"].astype(float)

    # 取出兩條線資料
    algos = df["Algorithm"].unique()
    plt.figure(figsize=(8,5))
    for alg in algos:
        sub = df[df["Algorithm"] == alg].sort_values("num_requests")
        plt.plot(sub["num_requests"].values, sub["RelativeIGD"].values, label=alg, marker='o')
    plt.xlabel("num_requests")
    plt.ylabel("Relative IGD")
    plt.title("Relative IGD vs. num_requests (NSGA4 vs NSCO)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("relative_igd_vs_n.png", dpi=150)
    plt.show()
