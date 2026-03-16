import networkx as nx
import numpy as np
import pandas as pd
import os
from sklearn.metrics import normalized_mutual_info_score

# --- 1. LDHETA 核心演算法函數 ---

def get_k_layer_neighbors(G, node, k, exclude_node=None):
    distances = nx.single_source_shortest_path_length(G, node, cutoff=k)
    k_neighbors = {n for n, d in distances.items() if d == k}
    if exclude_node in k_neighbors: k_neighbors.remove(exclude_node)
    return k_neighbors

def calculate_R_ij_k(G, u, v, k):
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
    S_ij_k = get_k_layer_neighbors(G, u, k, v) | get_k_layer_neighbors(G, v, k, u)
    N_k = len(S_ij_k)
    if N_k <= 1: return 0
    return (2 * G.subgraph(S_ij_k).number_of_edges()) / (N_k * (N_k - 1))

def run_analysis_logic(G, alpha=0.1): 
    """執行 HETA 與 LDHETA 的分類運算 (引入平滑因子 alpha)"""
    if nx.is_connected(G):
        avg_path = nx.average_shortest_path_length(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        avg_path = nx.average_shortest_path_length(G.subgraph(largest_cc))
        
    k_max = max(1, int(np.floor(avg_path / 2)))
    edges = [tuple(sorted(e)) for e in G.edges()]
    
    R_vals = {k: {e: calculate_R_ij_k(G, e[0], e[1], k) for e in edges} for k in range(1, k_max + 1)}
    D_vals = {k: {e: calculate_LD_ij_k(G, e[0], e[1], k) for e in edges} for k in range(1, k_max + 1)}
    Avg_D = {k: np.mean(list(D_vals[k].values())) for k in range(1, k_max + 1)}
    T_E_base = {k: np.mean(list(R_vals[k].values())) + 0.5 * np.std(list(R_vals[k].values())) for k in range(1, k_max + 1)}

    def classify(method='HETA'):
        res = {e: 'Global Bridge' for e in edges}
        for e in edges:
            if G.degree(e[0]) == 1 or G.degree(e[1]) == 1: res[e] = 'Silk'
        
        remaining = [e for e in edges if res[e] == 'Global Bridge']
        for k in range(1, k_max + 1):
            bonds = []
            for e in remaining:
                if method == 'LDHETA' and Avg_D[k] > 0:
                    g_ij = D_vals[k][e] / Avg_D[k]
                    threshold_multiplier = (1 + alpha * (g_ij - 1))
                else:
                    threshold_multiplier = 1
                
                if R_vals[k][e] >= T_E_base[k] * threshold_multiplier:
                    res[e] = 'Bond'; bonds.append(e)
            remaining = [e for e in remaining if e not in bonds]
        return res

    return classify('HETA'), classify('LDHETA')

# --- 2. 準確度評估工具 ---

def load_facebook_data(node_id):
    edge_file = f"實驗0118/facebook/{node_id}.edges"
    if not os.path.exists(edge_file):
        raise FileNotFoundError(f"找不到邊界檔案: {edge_file}")
    G = nx.read_edgelist(edge_file, nodetype=int)
    
    circle_file = f"實驗0118/facebook/{node_id}.circles"
    ground_truth_communities = []
    if os.path.exists(circle_file):
        with open(circle_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                nodes = [int(n) for n in parts[1:]]
                ground_truth_communities.append(set(nodes))
    return G, ground_truth_communities

def evaluate_accuracy(detected_communities, ground_truth, all_nodes):
    def get_labels(communities, nodes):
        label_dict = {node: -1 for node in nodes}
        for i, comm in enumerate(communities):
            for node in comm:
                if node in label_dict:
                    label_dict[node] = i
        return [label_dict[n] for n in nodes]
    node_list = list(all_nodes)
    true_labels = get_labels(ground_truth, node_list)
    pred_labels = get_labels(detected_communities, node_list)
    return normalized_mutual_info_score(true_labels, pred_labels)

def extract_bond_communities(G, res_dict):
    bond_edges = [e for e, t in res_dict.items() if t == 'Bond']
    tmp_G = nx.Graph()
    tmp_G.add_nodes_from(G.nodes())
    tmp_G.add_edges_from(bond_edges)
    return [list(c) for c in nx.connected_components(tmp_G)]

# --- 3. 執行批量實驗 ---

def run_batch_experiment(alpha_value):
    """
    接收 alpha 值，並對指定的社交圈跑完一遍實驗
    """
    ego_nodes = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]
    results = []

    print(f"\n>>> 正在測試 alpha = {alpha_value} <<<")
    print(f"{'NodeID':<8} | {'HETA NMI':<10} | {'LDHETA NMI':<10} | {'Improvement':<10}")
    print("-" * 50)

    for node_id in ego_nodes:
        try:
            G, ground_truth = load_facebook_data(node_id)
            all_nodes = set(G.nodes())

            # 傳入目前的 alpha 值
            h_res, ld_res = run_analysis_logic(G, alpha=alpha_value)

            h_comm = extract_bond_communities(G, h_res)
            ld_comm = extract_bond_communities(G, ld_res)
            
            h_nmi = evaluate_accuracy(h_comm, ground_truth, all_nodes)
            ld_nmi = evaluate_accuracy(ld_comm, ground_truth, all_nodes)
            
            improvement = ((ld_nmi - h_nmi) / h_nmi * 100) if h_nmi > 0 else 0
            
            results.append({
                'Alpha': alpha_value,
                'Node': node_id,
                'HETA_NMI': round(h_nmi, 4),
                'LDHETA_NMI': round(ld_nmi, 4),
                'Improvement_%': round(improvement, 2)
            })
            
            print(f"{node_id:<8} | {h_nmi:<10.4f} | {ld_nmi:<10.4f} | {improvement:>9.2f}%")

        except Exception as e:
            print(f"節點 {node_id} 運算跳過: {e}")

    df_res = pd.DataFrame(results)
    avg_imp = df_res['Improvement_%'].mean()
    print("-" * 50)
    print(f"Alpha {alpha_value} 平均改進率: {avg_imp:.2f}%")
    return df_res

# --- 4. Main 區塊 ---

if __name__ == "__main__":
    # 使用陣列依序測試不同的 alpha 值
    alpha_list = [-2.0, -1.5, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 1.5, 2.0]
    all_reports = []

    for a in alpha_list:
        report = run_batch_experiment(a)
        all_reports.append(report)

    # 合併所有測試結果並儲存
    final_df = pd.concat(all_reports)
    final_df.to_csv("facebook_alpha_test_results.csv", index=False)
    print("\n所有 alpha 測試完成，結果已儲存至 facebook_alpha_test_results.csv")