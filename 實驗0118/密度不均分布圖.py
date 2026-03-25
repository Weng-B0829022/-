import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_density_pdf(node_id, data_path):
    # 1. 載入資料
    edge_file = f"{data_path}/{node_id}.edges"
    if not os.path.exists(edge_file):
        print(f"跳過節點 {node_id}: 檔案不存在")
        return None
    
    G = nx.read_edgelist(edge_file, nodetype=int)
    edges = list(G.edges())
    
    # 2. 計算所有邊的局部密度 Dij (k=1)
    # 這裡直接呼叫您提供的邏輯核心
    d_vals = []
    for u, v in edges:
        # 計算局部密度: (2 * 實際邊數) / (N*(N-1))
        # 範圍為 u, v 的鄰居聯集
        s_ij_1 = set(G.neighbors(u)) | set(G.neighbors(v))
        s_ij_1.discard(u)
        s_ij_1.discard(v)
        
        n_k = len(s_ij_1)
        if n_k > 1:
            d_ij = (2 * G.subgraph(s_ij_1).number_of_edges()) / (n_k * (n_k - 1))
        else:
            d_ij = 0
        d_vals.append(d_ij)
    
    d_vals = np.array(d_vals)
    
    # 3. 繪圖
    plt.figure(figsize=(8, 5))
    counts, bin_edges = np.histogram(d_vals, bins=50, density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.plot(bin_centres, counts, label=f'Node {node_id} Density PDF', color='teal', lw=2)
    plt.fill_between(bin_centres, counts, alpha=0.2, color='teal')
    
    # 標註平均值與異質性
    mean_d = np.mean(d_vals)
    cv = np.std(d_vals) / mean_d if mean_d > 0 else 0
    plt.axvline(mean_d, color='red', linestyle='--', label=f'Mean: {mean_d:.3f}')
    
    plt.title(f'Local Density PDF (Ego: {node_id}) | CV: {cv:.2f}')
    plt.xlabel('Local Subgraph Density ($D_{ij}$)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 儲存
    save_name = f"實驗0118/facebook/density_pdf_{node_id}.png"
    plt.savefig(save_name)
    plt.close()
    return cv

# 執行
data_folder = "facebook"
ego_nodes = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]

print("開始生成密度分佈圖...")
for node in ego_nodes:
    cv_val = generate_density_pdf(node, data_folder)
    if cv_val is not None:
        print(f"節點 {node} 完成，變異係數 (CV): {cv_val:.2f}")