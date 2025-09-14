import pandas as pd
import matplotlib.pyplot as plt

# 讀總表
df = pd.read_csv("./out_metrics_mean/metrics_summary.csv")

# 畫圖
plt.figure(figsize=(8,6))
for alg, group in df.groupby("Algorithm"):
    plt.errorbar(
        group["num_requests"],
        group["RelativeIGD_mean"],
        yerr=group["RelativeIGD_std"],
        label=alg,
        marker='o',
        capsize=4,
        linestyle='-'
    )

plt.xlabel("Number of Requests", fontsize=12)
plt.ylabel("Relative IGD (mean ± std)", fontsize=12)
plt.title("Relative IGD Comparison (NSGA4 vs NSCO)", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
