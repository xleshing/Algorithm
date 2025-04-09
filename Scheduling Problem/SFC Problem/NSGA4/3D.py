import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import matplotlib.colors as colorscm
import matplotlib.cm as cm

# -------------------------------
# Step 1. 讀取並解析 CSV 檔案
# -------------------------------
csv_path = 'csv/3D/NSGA4_generation_solutions_data1_100.csv'
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
# 為了後續建立 colorbar，我們先建立一個對應的 Normalize 物件
norm = colorscm.Normalize(vmin=min(generations), vmax=max(generations))

for gen, color in zip(generations, colors):
    gen_data = solutions_df[solutions_df['Generation'] == gen]
    ax.scatter(
        gen_data['LoadBalance'],
        gen_data['Average Delay'],
        gen_data['Throughput'],
        color=color,
        label=f'Generation {gen}',
        s=50,
        edgecolor='k'
    )

ax.set_xlabel('LoadBalance')
ax.set_ylabel('Average Delay')
ax.set_zlabel('Throughput')
ax.set_title('NSGA4')

# -------------------------------
# Step 3. 將 legend 另外顯示並分成四列(4個欄位)豎直顯示
# -------------------------------
# 加上 colorbar 表示 Generation 數值
sm = cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
sm.set_array([])  # 空陣列即可
cbar = plt.colorbar(sm, ax=ax, pad=0.1)
cbar.set_label('Generation')

plt.tight_layout()  # 自動調整布局以避免圖例和圖形重疊
plt.show()



#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import ast
# from mpl_toolkits.mplot3d import Axes3D  # 3D 支援
# import matplotlib.cm as cm
# import matplotlib.colors as colorscm
#
# # -------------------------------
# # Step 1. 讀取並解析 10 個 CSV 檔案，並合併所有解答資料
# # -------------------------------
# solutions_data_all = []
#
# for dataset in range(1, 11):
#     csv_path = f'csv/3D/NSGA4_generation_solutions_data{dataset}_100.csv'
#     try:
#         df = pd.read_csv(csv_path)
#     except Exception as e:
#         print(f"讀取 {csv_path} 失敗：{e}")
#         continue
#
#     # 解析每筆資料中的 "sol" 欄位（字串形式，內含列表字典）
#     for index, row in df.iterrows():
#         generation = row['Generation']  # 假設已記錄世代
#         sol_str = row['sol']
#         try:
#             sol_list = ast.literal_eval(sol_str)
#         except Exception as e:
#             print(f"轉換第 {generation} 代 (data{dataset}) 解答失敗：{e}")
#             continue
#         # 為每個解答加入世代資訊（也可以加入資料集編號，但這裡不使用不同資料集顏色）
#         for sol in sol_list:
#             sol['Generation'] = generation
#             # 若有需要，也可加入 dataset 編號 (但後續不採用此欄位做顏色區分)
#             sol['Dataset'] = dataset
#             solutions_data_all.append(sol)
#
# # 合併所有資料成一個 DataFrame
# solutions_df = pd.DataFrame(solutions_data_all)
# print("合併後的解答資料（前 5 筆）：")
# print(solutions_df.head())
#
# # -------------------------------
# # Step 2. 建立 3D 散點圖，僅以 Generation 區分顏色
# # -------------------------------
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # 取得所有不同疊代次數（世代）
# generations = sorted(solutions_df['Generation'].unique())
# # 利用 viridis colormap 為不同世代產生一組顏色
# color_values = plt.cm.viridis(np.linspace(0, 1, len(generations)))
#
# # 為了後續建立 colorbar，我們先建立一個對應的 Normalize 物件
# norm = colorscm.Normalize(vmin=min(generations), vmax=max(generations))
#
# # 由於 100 代較多，這裡建議不要用 legend (會出現 100 多個圖例項)
# # 改以依世代分組繪製散點，並用 colorbar 呈現「Generation」資訊
# for gen, color in zip(generations, color_values):
#     gen_data = solutions_df[solutions_df['Generation'] == gen]
#     ax.scatter(
#         gen_data['LoadBalance'],
#         gen_data['Average Delay'],
#         gen_data['Throughput'],
#         color=color,
#         s=30,
#         edgecolor='k',
#         alpha=0.7
#     )
#
# ax.set_xlabel('LoadBalance')
# ax.set_ylabel('Average Delay')
# ax.set_zlabel('Throughput')
# ax.set_title('NSGA3 / NSGA4 各疊代解答合併展示')
#
# # 加上 colorbar 表示 Generation 數值
# sm = cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
# sm.set_array([])  # 空陣列即可
# cbar = plt.colorbar(sm, ax=ax, pad=0.1)
# cbar.set_label('Generation')
#
# plt.tight_layout()
# plt.show()



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import ast
# from mpl_toolkits.mplot3d import Axes3D  # 支持 3D 繪圖
# import matplotlib.colors as mcolors
#
# # -------------------------------
# # Step 1. 讀取並解析 10 個 CSV 檔案，並合併所有解答資料
# # -------------------------------
# solutions_data_all = []
#
# for dataset in range(1, 11):
#     csv_path = f'csv/3D/NSGA4_generation_solutions_data{dataset}_100.csv'
#     try:
#         df = pd.read_csv(csv_path)
#     except Exception as e:
#         print(f"讀取 {csv_path} 失敗：{e}")
#         continue
#
#     # 對每個檔案，解析 "sol" 欄位（存放一代內多個解的字串）
#     for index, row in df.iterrows():
#         generation = row['Generation']  # 假設每行都有 Generation 欄位
#         sol_str = row['sol']
#         try:
#             sol_list = ast.literal_eval(sol_str)
#         except Exception as e:
#             print(f"轉換第 {generation} 代 (data{dataset}) 解答失敗：{e}")
#             continue
#         # 為每個解答加入世代與資料集標識（這裡 Dataset 標識僅做參考）
#         for sol in sol_list:
#             sol['Generation'] = generation
#             sol['Dataset'] = dataset
#             solutions_data_all.append(sol)
#
# # 將所有資料合併成一個 DataFrame
# solutions_df = pd.DataFrame(solutions_data_all)
# print("合併後的解答資料（前 5 筆）：")
# print(solutions_df.head())
#
# # -------------------------------
# # Step 2. 對 10 組資料依 Generation 求平均
# # -------------------------------
# # 針對 LoadBalance, Average Delay, Throughput 這三個目標欄位
# grouped_df = solutions_df.groupby('Generation')[['LoadBalance', 'Average Delay', 'Throughput']].mean().reset_index()
# print("各 Generation 平均後的資料（前 5 筆）：")
# print(grouped_df.head())
#
# # -------------------------------
# # Step 3. 以 3D 散點圖展示平均值，依 Generation 著色
# # -------------------------------
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # 取得各 Generation 數值並依據 colormap 映射顏色
# norm = mcolors.Normalize(vmin=grouped_df['Generation'].min(), vmax=grouped_df['Generation'].max())
# cmap = plt.cm.viridis
# colors = cmap(norm(grouped_df['Generation']))
#
# sc = ax.scatter(
#     grouped_df['LoadBalance'],
#     grouped_df['Average Delay'],
#     grouped_df['Throughput'],
#     c=grouped_df['Generation'],    # 以 Generation 數值上色
#     cmap='viridis',
#     s=50,
#     edgecolors='k'
# )
#
# ax.set_xlabel('LoadBalance')
# ax.set_ylabel('Average Delay')
# ax.set_zlabel('Throughput')
# ax.set_title('Generation Mean')
#
# # 加入 colorbar 表示 Generation
# cbar = plt.colorbar(sc, ax=ax, pad=0.1)
# cbar.set_label('Generation')
#
# plt.tight_layout()
# plt.show()
