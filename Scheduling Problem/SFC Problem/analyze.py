import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re


def extract_node_from_filename(filename):
    """
    從檔名中使用正規表示法提取節點數
    假設檔名格式為: nsgaX_node.csv，例如 nsga3_25.csv 或 nsga4_50.csv
    """
    match = re.search(r'_(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return None


def compute_combined_std(df):
    """
    計算 CSV 中三個目標的標準差，並以歐幾里得方式合成綜合標準差:
        combined_std = sqrt(std(LoadBalance)^2 + std(Delay)^2 + std(Throughput)^2)
    """
    std_load = df['LoadBalance'].std(ddof=0)
    std_delay = df['Average Delay'].std(ddof=0)
    std_through = df['Throughput'].std(ddof=0)
    return np.sqrt(std_load ** 2 + std_delay ** 2 + std_through ** 2)


def load_and_process_files(file_pattern):
    """
    根據 file_pattern 讀取所有 CSV 檔案，並回傳一個 dict {node: combined_std}
    """
    std_dict = {}
    for filepath in glob.glob(file_pattern):
        node_count = extract_node_from_filename(filepath)
        if node_count is None:
            raise ValueError()
        df = pd.read_csv(filepath)
        # 計算綜合標準差
        combined_std = compute_combined_std(df)
        std_dict[node_count] = combined_std
    return std_dict

nsga3_std_sum = {}
nsga4_std_sum = {}

for i in range(1, 11):
    # 讀取 nsga3 與 nsga4 的檔案，請根據實際路徑調整 pattern
    nsga3_std = load_and_process_files(f"nsga3csv/data{i}/NSGA3_solutions_data_*.csv")
    nsga4_std = load_and_process_files(f"nsga4csv/data{i}/NSGA4_solutions_data_*.csv")

    # 依節點數排序
    nodes_nsga3 = sorted(nsga3_std.keys())
    nodes_nsga4 = sorted(nsga4_std.keys())
    if i > 1:
        for j in nodes_nsga3:
            nsga3_std_sum[j] += nsga3_std[j]
        for k in nodes_nsga3:
            nsga4_std_sum[k] += nsga4_std[k]
    else:
        for j in nodes_nsga3:
            nsga3_std_sum[j] = nsga3_std[j]
        for k in nodes_nsga3:
            nsga4_std_sum[k] = nsga4_std[k]

plt.figure(figsize=(8, 6))
plt.plot(sorted(nsga3_std_sum.keys()), [nsga3_std_sum[n] / 10 for n in sorted(nsga3_std_sum.keys())],
         marker='o', label='NSGA3')
plt.plot(sorted(nsga4_std_sum.keys()), [nsga4_std_sum[n] / 10 for n in sorted(nsga4_std_sum.keys())],
         marker='s', label='NSGA4')
plt.xlabel('Node Num')
plt.ylabel('Average Combined STD')
plt.title('NSGA3 vs NSGA4 with different Node Num')
plt.legend()
plt.grid(True)
plt.show()
