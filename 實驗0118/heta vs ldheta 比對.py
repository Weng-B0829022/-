import networkx as nx
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# --- 1. LDHETA 核心演算法函數 ---

def get_k_layer_neighbors(G, node, k, exclude_node=None):
    """獲取節點的第 k 層鄰居"""
    distances = nx.single_source_shortest_path_length(G, node, cutoff=k)
    k_neighbors = {n for n, d in distances.items() if d == k}
    if exclude_node in k_neighbors: 
        k_neighbors.remove(exclude_node)
    return k_neighbors

def calculate_R_ij_k(G, u, v, k):
    """計算節點 u, v 在第 k 層的邊可靠性 (Edge Reliability)"""
    V_uk = get_k_layer_neighbors(G, u, k, v)
    V_vk = get_k_layer_neighbors(G, v, k, u)
    
    if k == 1:
        if not V_uk or not V_vk: return 0
        return len(V_uk & V_vk) / min(len(V_uk), len(V_vk))
    else:
        V_uk_1 = get_k_layer_neighbors(G, u, k-1, v)
        V_vk_1 = get_k_layer_neighbors(G, v, k-1, u)
        # 分子：跨層級的共同鄰居集合
        num_set = (V_uk & V_vk) | (V_uk_1 & V_vk) | (V_uk & V_vk_1)
        # 分母：最小集合大小之和
        den = min(len(V_uk), len(V_vk)) + min(len(V_uk_1), len(V_vk)) + min(len(V_uk), len(V_vk_1))
        return len(num_set) / den if den > 0 else 0

def calculate_local_density_ij_k(G, u, v, k):
    """計算邊 (u, v) 在第 k 層的局部密度"""
    S_ij_k = get_k_layer_neighbors(G, u, k, v) | get_k_layer_neighbors(G, v, k, u)
    N_k = len(S_ij_k)
    if N_k <= 1: return 0
    return (2 * G.subgraph(S_ij_k).number_of_edges()) / (N_k * (N_k - 1))

def run_analysis_logic(G, alpha=-1.5):
    """執行 HETA 與 LDHETA 邏輯並回傳四種邊的分類"""
    # 計算平均路徑長度以決定 k_max
    if nx.is_connected(G):
        avg_path = nx.average_shortest_path_length(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        avg_path = nx.average_shortest_path_length(G.subgraph(largest_cc))
    
    k_max = max(1, int(np.floor(avg_path / 2)))
    edges = [tuple(sorted(e)) for e in G.edges()]
    
    # 預計算 R 值、局部密度與全域密度
    R_vals = {k: {e: calculate_R_ij_k(G, e[0], e[1], k) for e in edges} for k in range(1, k_max + 1)}
    ld_vals = {k: {e: calculate_local_density_ij_k(G, e[0], e[1], k) for e in edges} for k in range(1, k_max + 1)}
    g_avg_dens = {k: np.mean(list(ld_vals[k].values())) if ld_vals[k] else 0 for k in range(1, k_max + 1)}
    T_E_base = {k: np.mean(list(R_vals[k].values())) + 0.5 * np.std(list(R_vals[k].values())) for k in range(1, k_max + 1)}

    def classify(method='LDHETA'):
        # 初始全部設為 Global (綠色)
        res = {e: 'Global' for e in edges}
        
        # 1. 識別 Silk (黃色): 度為 1 的邊
        for e in edges:
            if G.degree(e[0]) == 1 or G.degree(e[1]) == 1:
                res[e] = 'Silk'
        
        # 針對非 Silk 的邊進行 k-layer 判定
        for k in range(1, k_max + 1):
            for e in edges:
                if res[e] == 'Silk': continue
                
                # 計算動態門檻
                if method == 'LDHETA' and g_avg_dens[k] > 0:
                    local_ratio = ld_vals[k][e] / g_avg_dens[k]
                    threshold = T_E_base[k] * (1 + alpha * (local_ratio - 1))
                else:
                    threshold = T_E_base[k]
                
                # 2. 識別 Bond (紅色): 高於門檻 (且不被後續覆蓋)
                if R_vals[k][e] >= threshold:
                    res[e] = 'Bond'
                # 3. 識別 Local (藍色): 有共同鄰居但未達 Bond (且不覆蓋已有的 Bond)
                elif R_vals[k][e] > 0 and res[e] != 'Bond':
                    res[e] = 'Local'
        return res

    return classify('HETA'), classify('LDHETA')

# --- 2. 繪圖函數 ---

def plot_three_views(G, h_res, ld_res, node_id, alpha, save_path):
    pos = nx.spring_layout(G, seed=42)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    color_map = {
        'Bond': '#e41a1c',   # 紅
        'Local': '#377eb8',  # 藍
        'Global': '#4daf4a', # 綠
        'Silk': '#ffff33'    # 黃
    }
    
    titles = ['Original Network', 'HETA Classification', f'LDHETA (alpha={alpha})']
    
    for i, (ax, title) in enumerate(zip(axes, titles)):
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=25, node_color='black', alpha=0.6)
        
        if i == 0:
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.2)
        else:
            res = h_res if i == 1 else ld_res
            edge_list = [tuple(sorted(e)) for e in G.edges()]
            colors = [color_map.get(res.get(e, 'Global')) for e in edge_list]
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color=colors, width=1.5, alpha=0.8)
            
            # 計算各比例用於標籤
            counts = pd.Series(res.values()).value_counts(normalize=True) * 100
            stats_text = "\n".join([f"{k}: {v:.1f}%" for k, v in counts.items()])
            ax.text(0.05, 0.05, stats_text, transform=ax.transAxes, fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.7))

        ax.set_title(title, fontsize=16)
        ax.axis('off')
    
    plt.suptitle(f"Facebook Ego Network: {node_id}\nRed: Bond | Blue: Local | Green: Global | Yellow: Silk", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()

# --- 3. 批量執行 ---

if __name__ == "__main__":
    # 請確保資料夾路徑正確
    data_folder = "實驗0118/facebook"
    output_folder = "analysis_results"
    os.makedirs(output_folder, exist_ok=True)
    
    configs = [
        (0, -1.56), (107, -1.27), (348, -0.96), (414, -0.45), (686, -1.03),
        (698, -0.58), (1684, -1.00), (1912, -1.00), (3437, -1.03), (3980, -0.84)
    ]
    
    for node_id, a_val in configs:
        file_path = os.path.join(data_folder, f"{node_id}.edges")
        if not os.path.exists(file_path):
            print(f"Skipping {node_id}: File not found.")
            continue
            
        print(f"Processing Node {node_id}...")
        G = nx.read_edgelist(file_path, nodetype=int)
        
        h_res, ld_res = run_analysis_logic(G, alpha=a_val)
        
        save_name = os.path.join(output_folder, f"four_colors_analysis_{node_id}.png")
        plot_three_views(G, h_res, ld_res, node_id, a_val, save_name)
        print(f"Saved: {save_name}")