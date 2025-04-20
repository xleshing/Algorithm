import pandas as pd
import matplotlib.pyplot as plt

# 1. 定义节点规模和数据文件夹列表
node_counts = list(range(15, 101, 5))
data_dirs = [f"data{i}" for i in range(1, 11)]

# 2. 用来存放“十组数据再平均”后的指标列表
load_balance_means = []
avg_delay_means = []
throughput_means = []

# 3. 对每个规模循环
for cnt in node_counts:
    # 临时列表：存放 data1~data10 对应文件的单组平均
    lb_tmp = []
    ad_tmp = []
    tp_tmp = []

    # 遍历十个文件夹
    for d in data_dirs:
        fn = f"nsga3csv/{d}/NSGA3_solutions_data_{cnt}.csv"
        df = pd.read_csv(fn)

        # 各自计算这组文件内所有解的平均
        lb_tmp.append(df['LoadBalance'].mean())
        ad_tmp.append(df['Average Delay'].mean())
        tp_tmp.append(df['Throughput'].mean())

    # 对十个平均值再算一次平均
    load_balance_means.append(sum(lb_tmp) / len(lb_tmp))
    avg_delay_means.append(sum(ad_tmp) / len(ad_tmp))
    throughput_means.append(sum(tp_tmp) / len(tp_tmp))

# 4. 绘图（与之前完全相同，只是 y 数据换成新的列表）
plt.figure()
plt.bar(node_counts, load_balance_means)
plt.xlabel('Node Count')
plt.ylabel('LoadBalance')
plt.title('Avg LoadBalance across data1~data10')
plt.show()

plt.figure()
plt.bar(node_counts, avg_delay_means)
plt.xlabel('Node Count')
plt.ylabel('Average Delay')
plt.title('Avg Delay across data1~data10')
plt.show()

plt.figure()
plt.bar(node_counts, throughput_means)
plt.xlabel('Node Count')
plt.ylabel('Throughput')
plt.title('Avg Throughput across data1~data10')
plt.show()
