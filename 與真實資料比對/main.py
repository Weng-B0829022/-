import networkx as nx
import numpy as np
import os
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

# ==========================================
# 1. LDHETA & HETA 核心公式
# ==========================================

def calculate_R_ij(G, u, v):
    """計算親密度指標 R_ij (k=1)"""
    set_u = set(G.neighbors(u)) - {v}
    set_v = set(G.neighbors(v)) - {u}
    if not set_u or not set_v: return 0
    return len(set_u & set_v) / min(len(set_u), len(set_v))

def calculate_LD_ij(G, u, v):
    """計算局部密度 D_ij"""
    local_nodes = set(G.neighbors(u)) | set(G.neighbors(v))
    n_size = len(local_nodes)
    if n_size <= 1: return 0
    local_subgraph = G.subgraph(local_nodes)
    actual_e = local_subgraph.number_of_edges()
    max_e = (n_size * (n_size - 1)) / 2
    return actual_e / max_e

def run_experiment(node_id, data_path, alpha=0.5):
    edge_file = f"{data_path}/{node_id}.edges"
    circle_file = f"{data_path}/{node_id}.circles"
    
    if not os.path.exists(edge_file):
        return None

    # A. 載入網路
    G = nx.read_edgelist(edge_file, nodetype=int)
    edges = [tuple(sorted(e)) for e in G.edges()]
    
    # B. 執行 HETA & LDHETA 分類
    r_vals = {e: calculate_R_ij(G, e[0], e[1]) for e in edges}
    vals = list(r_vals.values())
    t_e_base = np.mean(vals) + 0.5 * np.std(vals)
    
    d_vals = {e: calculate_LD_ij(G, e[0], e[1]) for e in edges}
    avg_d = np.mean(list(d_vals.values()))

    heta_bonds = [e for e in edges if r_vals[e] >= t_e_base]
    ld_bonds = []
    for e in edges:
        g_ij = d_vals[e] / avg_d if avg_d > 0 else 1
        if r_vals[e] >= t_e_base * (1 + alpha * (g_ij - 1)):
            ld_bonds.append(e)

    # C. 載入 Ground Truth (Circles)
    nodes = sorted(list(G.nodes()))
    y_true = {n: -1 for n in nodes}
    if os.path.exists(circle_file):
        with open(circle_file, 'r') as f:
            for i, line in enumerate(f):
                members = list(map(int, line.strip().split('\t')[1:]))
                for m in members:
                    if m in y_true: y_true[m] = i
    
    true_labels = [y_true[n] for n in nodes]

    # D. 計算 NMI
    def get_nmi(bond_list):
        tmp_G = nx.Graph()
        tmp_G.add_nodes_from(G.nodes())
        tmp_G.add_edges_from(bond_list)
        communities = list(nx.connected_components(tmp_G))
        pred_map = {n: -1 for n in nodes}
        for i, c in enumerate(communities):
            for n in c: pred_map[n] = i
        pred_labels = [pred_map[n] for n in nodes]
        return normalized_mutual_info_score(true_labels, pred_labels)

    nmi_heta = get_nmi(heta_bonds)
    nmi_ldheta = get_nmi(ld_bonds)
    
    return {
        'NodeID': node_id,
        'Nodes': len(G.nodes()),
        'Edges': len(G.edges()),
        'HETA_NMI': round(nmi_heta, 4),
        'LDHETA_NMI': round(nmi_ldheta, 4),
        'Improvement': f"{((nmi_ldheta - nmi_heta) / nmi_heta * 100):.2f}%" if nmi_heta > 0 else "N/A"
    }

# ==========================================
# 2. 批量執行與結果輸出
# ==========================================

def main():
    data_folder = "實驗0118/facebook"
    ego_nodes = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]
    results = []

    print(f"正在執行 10 個 Facebook 網路的 NMI 實驗 (Alpha=0.5)...\n")
    
    for node in ego_nodes:
        res = run_experiment(node, data_folder, alpha=0.5)
        if res:
            results.append(res)
            # 即時輸出單筆結果
            print(f"Node {res['NodeID']:<5}: HETA NMI = {res['HETA_NMI']:.4f}, LDHETA NMI = {res['LDHETA_NMI']:.4f} ({res['Improvement']})")

    # 輸出最終彙整表格
    df = pd.DataFrame(results)
    print("\n" + "="*65)
    print(df.to_string(index=False))
    print("="*65)

if __name__ == "__main__":
    main()