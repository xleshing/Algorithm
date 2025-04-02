import networkx as nx
import matplotlib.pyplot as plt
from csv2list import csv2list

# 定義網路節點
# network_nodes = [
#     {'id': 'A', 'vnf_types': ['0', '1'], 'neighbors': ['B', 'C'],
#      'load_per_vnf': {'0': 0.5, '1': 0.7},
#      'processing_delay': {'0': 2, '1': 3}},
#     {'id': 'B', 'vnf_types': ['0', '2', '3'], 'neighbors': ['A', 'D', 'E'],
#      'load_per_vnf': {'0': 0.6, '2': 0.8, '3': 1.0},
#      'processing_delay': {'0': 2.5, '2': 2, '3': 2}},
#     {'id': 'C', 'vnf_types': ['0', '3', '2'], 'neighbors': ['A', 'D', 'G', 'F'],
#      'load_per_vnf': {'0': 0.55, '3': 0.65, '2': 0.75},
#      'processing_delay': {'0': 3, '3': 1.5, '2': 2.5}},
#     {'id': 'D', 'vnf_types': ['0', '2', '3'], 'neighbors': ['B', 'C', 'E', 'G'],
#      'load_per_vnf': {'0': 0.6, '2': 0.85, '3': 0.95},
#      'processing_delay': {'0': 3, '2': 1.8, '3': 2}},
#     {'id': 'E', 'vnf_types': ['3', '1'], 'neighbors': ['B', 'D', 'H'],
#      'load_per_vnf': {'3': 0.5, '1': 0.6},
#      'processing_delay': {'3': 3, '1': 1.8}},
#     {'id': 'F', 'vnf_types': ['1', '3'], 'neighbors': ['C', 'I', 'J'],
#      'load_per_vnf': {'1': 0.7, '3': 0.8},
#      'processing_delay': {'1': 3, '3': 1.8}},
#     {'id': 'G', 'vnf_types': ['1', '2'], 'neighbors': ['C', 'D', 'I', 'K', 'H'],
#      'load_per_vnf': {'1': 0.65, '2': 0.75},
#      'processing_delay': {'1': 3, '2': 1.8}},
#     {'id': 'H', 'vnf_types': ['0', '2', '3'], 'neighbors': ['E', 'G'],
#      'load_per_vnf': {'0': 0.55, '2': 0.65, '3': 0.75},
#      'processing_delay': {'0': 3, '2': 1.8, '3': 2}},
#     {'id': 'I', 'vnf_types': ['0', '2'], 'neighbors': ['F', 'G', 'K'],
#      'load_per_vnf': {'0': 0.6, '2': 0.7},
#      'processing_delay': {'0': 3, '2': 1.8}},
#     {'id': 'J', 'vnf_types': ['2', '1'], 'neighbors': ['F', 'K'],
#      'load_per_vnf': {'2': 0.7, '1': 0.8},
#      'processing_delay': {'2': 3, '1': 1.8}},
#     {'id': 'K', 'vnf_types': ['1', '3'], 'neighbors': ['G', 'I', 'J'],
#      'load_per_vnf': {'1': 0.55, '3': 0.65},
#      'processing_delay': {'1': 1.8, '3': 2}},
# ]
#
# # 邊資訊 (無向邊)，數值可代表帶寬或連接權重
# edges = {
#     ('A', 'B'): 100,
#     ('A', 'C'): 80,
#     ('C', 'F'): 90,
#     ('F', 'J'): 70,
#     ('J', 'K'): 60,
#     ('K', 'G'): 60,
#     ('G', 'H'): 70,
#     ('H', 'E'): 80,
#     ('E', 'B'): 60,
#     ('D', 'B'): 40,
#     ('D', 'C'): 100,
#     ('D', 'G'): 40,
#     ('D', 'E'): 70,
#     ('C', 'G'): 50,
#     ('I', 'F'): 70,
#     ('I', 'G'): 80,
#     ('I', 'K'): 70,
# }

# 創建無向圖

c2l = csv2list()
network_nodes = c2l.nodes(f"./problem/data5/nodes/nodes_15.csv")
edges = c2l.edges(f"./problem/data5/edges/edges_15.csv")

G = nx.Graph()

# 將節點加入圖中，並保存相關屬性
for node in network_nodes:
    G.add_node(node['id'],
               vnf_types=node['vnf_types'],
               load_per_vnf=node['load_per_vnf'],
               processing_delay=node['processing_delay'])

# 將邊加入圖中，並設定權重
for (u, v), weight in edges.items():
    G.add_edge(u, v, weight=weight)

# 計算節點位置
pos = nx.spring_layout(G, seed=42)

# 建立一個圖形，左邊顯示拓樸圖，右邊顯示節點資料表
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig, (ax1) = plt.subplots(1, figsize=(8, 8))

# ----- 左側：畫出網路拓樸 -----
nx.draw(G, pos, with_labels=True, node_size=1500, node_color="lightblue", font_size=10, ax=ax1)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', ax=ax1)
ax1.set_title("Network Topology")
ax1.axis("off")

# ----- 右側：建立節點資料表 -----
# 定義表格標題與資料
# columns = ["Node", "VNF Types", "Load per VNF", "Processing Delay"]
# table_data = []
# for node in network_nodes:
#     table_data.append([
#         node['id'],
#         ", ".join(node['vnf_types']),
#         ", ".join([f"{k}:{v}" for k,v in node['load_per_vnf'].items()]),
#         ", ".join([f"{k}:{v}" for k,v in node['processing_delay'].items()])
#     ])

# 隱藏軸
# ax2.axis('tight')
# ax2.axis('off')
# 建立表格
# table = ax2.table(cellText=table_data, colLabels=columns, loc='center')
# table.auto_set_font_size(False)
# table.set_fontsize(10)
# table.scale(1, 2)
# ax2.set_title("Node Data Table", fontweight="bold", pad=20)

plt.tight_layout()
plt.show()
