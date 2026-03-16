import networkx as nx
import os

# 1. 定義聚落大小（總和為 200）
# 設計五個大小不一的聚落：一個大、兩個中、兩個小
cluster_sizes = [80, 50, 40, 20, 10] 
p_in = 0.4   # 聚落內部連線機率
p_out = 0.01 # 聚落間連線機率

# 2. 生成隨機分區圖
G_clustered = nx.random_partition_graph(cluster_sizes, p_in, p_out, seed=42)

# 3. 確保目錄存在
output_dir = "實驗0118"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 4. 儲存檔案
file_path = os.path.join(output_dir, "facebook_combined.txt")
with open(file_path, "wb") as f:
    nx.write_edgelist(G_clustered, f, data=False)

print(f"五個大小不一的聚落已生成：")
print(f"- 總節點：{G_clustered.number_of_nodes()} 節點")
print(f"- 總邊數：{G_clustered.number_of_edges()} 條邊")
print(f"- 聚落大小分布：{cluster_sizes}")
print(f"- 檔案儲存於：{file_path}")