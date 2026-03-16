import networkx as nx
import numpy as np
import pandas as pd
import os
from sklearn.metrics import normalized_mutual_info_score

# ==========================================
# 1. LDHETA 核心運算函數
# ==========================================

def get_k_layer_neighbors(G, node, k, exclude_node=None):
    """獲取第 k 層鄰居"""
    distances = nx.single_source_shortest_path_length(G, node, cutoff=k)
    k_neighbors = {n for n, d in distances.items() if d == k}
    if exclude_node in k_neighbors: 
        k_neighbors.remove(exclude_node)
    return k_neighbors

def calculate_R_ij_k(G, u, v, k):
    """計算共同鄰居比例 R_ij"""
    V_uk = get_k_layer_neighbors(G, u, k, v)
    V_vk = get_k_layer_neighbors(G, v, k, u)
    if k == 1:
        if not V_uk or not V_vk: return 0
        return len(V_uk & V_vk) / min(len(V_uk), len(V_vk))
    else:
        V_uk_1 = get_k_layer_neighbors(G, u, k-1, v)
        V_vk_1 = get_k_layer_neighbors(G, v, k-1, u)
        num_set = (V_uk & V_vk) | (V_uk_1 & V_vk) | (V_uk & V_vk_1)
        den = min(len(V_uk), len(V_vk)) + min(len(V_uk_1), len(V_vk)) + min(len(V_uk), len(V_vk_1))
        return len(num_set) / den if den > 0 else 0

def calculate_LD_ij_k(G, u, v, k):
    """計算局部子圖密度 LD_ij"""
    S_ij_k = get_k_layer_neighbors(G, u, k, v) | get_k_layer_neighbors(G, v, k, u)
    N_k = len(S_ij_k)
    if N_k <= 1: return 0
    return (2 * G.subgraph(S_ij_k).number_of_edges()) / (N_k * (N_k - 1))

def analyze_network_heterogeneity(G):
    """計算網路異質性 (CV) 並建議 Alpha"""
    degrees = [d for n, d in G.degree()]
    if not degrees: return 0.0, 0.0
    cv = np.std(degrees) / np.mean(degrees) if np.mean(degrees) > 0 else 0
    # Alpha 映射邏輯 (CV 越高，Alpha 補正越強)
    suggested_alpha = np.clip(0.0 + ((cv - 0.4) / (1.2 - 0.4)) * 2.0, 0.0, 2.0)
    return round(cv, 4), round(suggested_alpha, 2)

# ==========================================
# 2. 分類與評估邏輯
# ==========================================

def run_heta_ldheta_comparison(G, alpha=0.1):
    """執行 HETA 與 LDHETA 分類並返回結果"""
    # 計算 k_max (平均路徑長度的一半)
    if nx.is_connected(G):
        avg_path = nx.average_shortest_path_length(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        avg_path = nx.average_shortest_path_length(G.subgraph(largest_cc))
    k_max = max(1, int(np.floor(avg_path / 2)))
    
    edges = [tuple(sorted(e)) for e in G.edges()]
    R_vals = {k: {e: calculate_R_ij_k(G, e[0], e[1], k) for e in edges} for k in range(1, k_max + 1)}
    D_vals = {k: {e: calculate_LD_ij_k(G, e[0], e[1], k) for e in edges} for k in range(1, k_max + 1)}
    Avg_D = {k: np.mean(list(D_vals[k].values())) if D_vals[k] else 0 for k in range(1, k_max + 1)}
    # 設定基礎門檻 T_E
    T_E_base = {k: np.mean(list(R_vals[k].values())) + 0.5 * np.std(list(R_vals[k].values())) for k in range(1, k_max + 1)}

    def classify(method='HETA'):
        res = {e: 'Bridge' for e in edges}
        for e in edges:
            if G.degree(e[0]) == 1 or G.degree(e[1]) == 1: res[e] = 'Silk'
        
        remaining = [e for e in edges if res[e] == 'Bridge']
        for k in range(1, k_max + 1):
            bonds = []
            for e in remaining:
                g_ij = D_vals[k][e] / Avg_D[k] if Avg_D[k] > 0 else 1
                multiplier = (1 + alpha * (g_ij - 1)) if method == 'LDHETA' else 1
                if R_vals[k][e] >= T_E_base[k] * multiplier:
                    res[e] = 'Bond'; bonds.append(e)
            remaining = [e for e in remaining if e not in bonds]
        return res

    return classify('HETA'), classify('LDHETA')

def evaluate_nmi(G, res_dict, ground_truth):
    """計算 NMI (以 Bond Link 形成的連通分量為社群)"""
    bond_edges = [e for e, t in res_dict.items() if t == 'Bond']
    tmp_G = nx.Graph()
    tmp_G.add_nodes_from(G.nodes())
    tmp_G.add_edges_from(bond_edges)
    detected_communities = [list(c) for c in nx.connected_components(tmp_G)]
    
    nodes = list(G.nodes())
    def get_labels(communities):
        labels = {node: -1 for node in nodes}
        for i, comm in enumerate(communities):
            for node in comm: labels[node] = i
        return [labels[n] for n in nodes]
    
    return normalized_mutual_info_score(get_labels(ground_truth), get_labels(detected_communities))

# ==========================================
# 3. 數據載入與主執行流程
# ==========================================

def load_data(node_id, folder):
    edge_f = f"{folder}/{node_id}.edges"
    circle_f = f"{folder}/{node_id}.circles"
    G = nx.read_edgelist(edge_f, nodetype=int)
    ground_truth = []
    if os.path.exists(circle_f):
        with open(circle_f, 'r') as f:
            for line in f:
                ground_truth.append(set(map(int, line.strip().split('\t')[1:])))
    return G, ground_truth

def main():
    data_folder = "實驗0118/facebook"
    ego_nodes = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]
    
    header = f"{'NodeID':<8} | {'CV':<6} | {'Alpha':<6} | {'HETA NMI':<10} | {'LDHETA NMI':<10} | {'Improve'}"
    print("\n" + header)
    print("-" * len(header))

    for node_id in ego_nodes:
        try:
            G, ground_truth = load_data(node_id, data_folder)
            cv_val, suggest_alpha = analyze_network_heterogeneity(G)
            
            # 執行運算
            heta_res, ldheta_res = run_heta_ldheta_comparison(G, alpha=suggest_alpha)
            
            # 計算 NMI
            nmi_heta = evaluate_nmi(G, heta_res, ground_truth)
            nmi_ldheta = evaluate_nmi(G, ldheta_res, ground_truth)
            
            diff = nmi_ldheta - nmi_heta
            print(f"{node_id:<8} | {cv_val:<6.2f} | {suggest_alpha:<6.2f} | {nmi_heta:<10.4f} | {nmi_ldheta:<10.4f} | {diff:>+7.4f}")
            
        except Exception as e:
            print(f"{node_id:<8} | 錯誤: {e}")

if __name__ == "__main__":
    main()