import networkx as nx
import pandas as pd
import matplotlib
# 強制使用 Agg 後端以避免 Qt 插件遺失的錯誤
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os

# =========================================================
# 1. 配置區域：定義要處理的 10 個網路 ID
# =========================================================
data_path = "facebook"  # 請確保此目錄下有對應的 .edges 與 .circles 檔案
node_list = ["0", "107", "348", "414", "686", "698", "1684", "1912", "3437", "3980"]

# 用於儲存所有網路總結數據的清單
summary_report = []

print(f"{'Node_ID':<10} | {'Global_Density':<15} | {'Max_g_ij':<10} | {'CV (不平均度)':<15}")
print("-" * 60)

for node_id in node_list:
    edge_file = f"{data_path}/{node_id}.edges"
    circle_file = f"{data_path}/{node_id}.circles"
    
    # 檢查檔案是否存在
    if not os.path.exists(edge_file):
        print(f"跳過 {node_id}: 找不到 edges 檔案")
        continue

    # =========================================================
    # 2. 建立網路並計算全域指標
    # =========================================================
    G = nx.read_edgelist(edge_file, nodetype=int)
    n_total = G.number_of_nodes()
    e_total = G.number_of_edges()
    global_density = e_total / (n_total * (n_total - 1) / 2) if n_total > 1 else 0

    # =========================================================
    # 3. 讀取社群並計算局部密度 
    # =========================================================
    analysis_results = []
    if os.path.exists(circle_file):
        with open(circle_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                circle_id = parts[0]
                members = [int(node) for node in parts[1:] if int(node) in G]
                
                if len(members) < 2:
                    continue
                
                subgraph = G.subgraph(members)
                n_local = len(members)
                e_local = subgraph.number_of_edges()
                
                # 局部密度計算 
                local_density = e_local / (n_local * (n_local - 1) / 2)
                # 密度偏離因子 g_ij 
                g_ij = local_density / global_density if global_density > 0 else 1.0
                
                analysis_results.append({
                    "Circle": circle_id,
                    "Density": local_density,
                    "g_ij": g_ij
                })

    if not analysis_results:
        continue

    df = pd.DataFrame(analysis_results)
    
    # 計算該網路的統計特徵
    cv = df['Density'].std() / df['Density'].mean()
    max_g = df['g_ij'].max()
    
    # 輸出表格到 Terminal
    print(f"{node_id:<10} | {global_density:<15.4f} | {max_g:<10.2f}x | {cv:<15.4f}")

    # =========================================================
    # 4. 繪製並儲存圖表 (檔名後綴 _{node_id})
    # =========================================================
    plt.figure(figsize=(12, 6))
    plt.bar(df['Circle'], df['Density'], color='royalblue', alpha=0.8, label='Local Density (D_ij)')
    plt.axhline(y=global_density, color='red', linestyle='--', linewidth=2, label=f'Global Avg')
    
    plt.title(f'Density Heterogeneity - Node {node_id}', fontsize=14)
    plt.ylabel('Density')
    plt.xlabel('Social Circles')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # 依照要求命名圖片
    output_filename = f"real_world_density_heterogeneity_{node_id}.png"
    plt.savefig(output_filename)
    plt.close() # 釋放記憶體

print("-" * 60)
print("批次處理完成，所有圖檔已儲存。")