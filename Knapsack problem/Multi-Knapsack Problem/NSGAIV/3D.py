import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from mpl_toolkits.mplot3d import Axes3D  # 為確保 3D 支援

# -------------------------------
# Step 1. 讀取並解析 CSV 檔案
# -------------------------------
csv_path = 'NSGA4_generation_solutions.csv'
df = pd.read_csv(csv_path)

# 解析 "sol" 欄位 (儲存一代內所有解答的字串)
solutions_data = []
for index, row in df.iterrows():
    generation = row['Generation']
    sol_str = row['sol']
    try:
        sol_list = ast.literal_eval(sol_str)
    except Exception as e:
        print(f"轉換第 {generation} 代解答資料失敗：{e}")
        continue
    # 為每個解答加入世代資訊
    for sol in sol_list:
        sol['Generation'] = generation
        solutions_data.append(sol)

solutions_df = pd.DataFrame(solutions_data)
print("解析後的解答資料：")
print(solutions_df.head())

# -------------------------------
# Step 2. 建立 3D 散點圖
# -------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 依世代排序並為各世代指派不同顏色
generations = sorted(solutions_df['Generation'].unique())
colors = plt.cm.viridis(np.linspace(0, 1, len(generations)))

for gen, color in zip(generations, colors):
    gen_data = solutions_df[solutions_df['Generation'] == gen]
    ax.scatter(
        gen_data['LoadBalance'],
        gen_data['Average Delay'],
        gen_data['Cost'],
        color=color,
        label=f'Generation {gen}',
        s=50,
        edgecolor='k'
    )

ax.set_xlabel('LoadBalance')
ax.set_ylabel('Average Delay')
ax.set_zlabel('Throughput')
ax.set_title('NSGA3 各世代解答三維目標圖')

# -------------------------------
# Step 3. 將 legend 另外顯示並分成四列(4個欄位)豎直顯示
# -------------------------------
# 將 legend 放置於圖外，並設定 ncol=4 讓 legend 分成 4 個欄位（豎直排列）
# ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=4, fontsize='small', markerscale=0.6)

plt.tight_layout()  # 自動調整布局以避免圖例和圖形重疊
plt.show()
