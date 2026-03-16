import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score

# --- [核心函數：共同鄰居與密度計算] ---

def get_neighbors_at_k(G, node, k, exclude_node):
    lengths = nx.single_source_shortest_path_length(G, node, cutoff=k)
    nodes_at_k = {n for n, dist in lengths.items() if dist == k}
    nodes_at_k.discard(exclude_node)
    nodes_at_k.discard(node)
    return nodes_at_k

def calculate_R_k(G, u, v, k):
    """計算共同鄰居比例 R_ij^k (Source 2)"""
    if k == 1:
        N_u = set(G.neighbors(u)) - {v}
        N_v = set(G.neighbors(v)) - {u}
        intersect = N_u & N_v
        denom = min(len(N_u), len(N_v))
        return len(intersect) / denom if denom > 0 else 0
    else:
        Vi_k, Vj_k = get_neighbors_at_k(G, u, k, v), get_neighbors_at_k(G, v, k, u)
        Vi_km1, Vj_km1 = get_neighbors_at_k(G, u, k-1, v), get_neighbors_at_k(G, v, k-1, u)
        num_set = (Vi_k & Vj_k) | (Vi_km1 & Vj_k) | (Vi_k & Vj_km1)
        denom = min(len(Vi_k), len(Vj_k)) + min(len(Vi_km1), len(Vj_k)) + min(len(Vi_k), len(Vj_km1))
        return len(num_set) / denom if denom > 0 else 0

def calculate_D_k(G, u, v, k):
    """計算局部子圖密度 D_ij^k (Source 3)"""
    V_i_k, V_j_k = get_neighbors_at_k(G, u, k, v), get_neighbors_at_k(G, v, k, u)
    S = V_i_k | V_j_k | {u, v}
    sub = G.subgraph(S)
    actual_edges = sub.number_of_edges()
    n = len(S)
    max_edges = n * (n - 1) / 2
    return actual_edges / max_edges if max_edges > 0 else 0

def get_external_thresholds(G, k_max, n_random=10):
    """模擬隨機網路產生的統計門檻 (Source 112)"""
    degrees = [d for n, d in G.degree()]
    all_R = {k: [] for k in range(1, k_max + 1)}
    for _ in range(n_random):
        RG = nx.configuration_model(degrees); RG = nx.Graph(RG); RG.remove_edges_from(nx.selfloop_edges(RG))
        for k in range(1, k_max + 1):
            for u, v in RG.edges(): all_R[k].append(calculate_R_k(RG, u, v, k))
    return {k: (np.mean(all_R[k]) + 2 * np.std(all_R[k])) if all_R[k] else 1.0 for k in range(1, k_max + 1)}

# --- [演算法主邏輯：識別與劃分] ---

def run_edge_analysis(G, is_ldheta=False):
    """執行 HETA/LDHETA 識別"""
    try: avg_path = nx.average_shortest_path_length(G) if nx.is_connected(G) else 4.0
    except: avg_path = 4.0
    k_max = max(1, int(avg_path // 2))
    T_E_all = get_external_thresholds(G, k_max)
    link_types, pass_status = {tuple(sorted(e)): None for e in G.edges()}, {tuple(sorted(e)): True for e in G.edges()}
    
    # Silk Links
    for u, v in G.edges():
        if G.degree(u) == 1 or G.degree(v) == 1: link_types[tuple(sorted((u, v)))], pass_status[tuple(sorted((u, v)))] = "Silk", False

    for k in range(1, k_max + 1):
        active = [e for e, p in pass_status.items() if p]
        if not active: break
        R_vals = {e: calculate_R_k(G, e[0], e[1], k) for e in active}
        D_vals = {e: calculate_D_k(G, e[0], e[1], k) for e in active} if is_ldheta else {e: 1.0 for e in active}
        avg_D = np.mean(list(D_vals.values())) if is_ldheta else 1.0
        
        candidates = []
        for e in active:
            g_ij = D_vals[e] / avg_D if is_ldheta and avg_D > 0 else 1.0
            if R_vals[e] >= T_E_all[k] * g_ij: link_types[e], pass_status[e] = "Bond", False
            else: candidates.append(e)
        
        if candidates:
            bridge_R = [R_vals[e] for e in candidates]
            T_I_base = np.mean(bridge_R) - np.std(bridge_R)
            for e in candidates:
                g_ij = D_vals[e] / avg_D if is_ldheta and avg_D > 0 else 1.0
                if R_vals[e] > T_I_base * g_ij: link_types[e], pass_status[e] = "Bridge", False
                    
    for e, p in pass_status.items():
        if p: link_types[e] = "Bridge"
    return link_types

def get_partition_labels(G, link_types):
    """Algorithm 2: 移除非鍵結連結以取得社群"""
    C = G.copy()
    C.remove_edges_from([e for e, t in link_types.items() if t != "Bond"])
    communities = list(nx.connected_components(C))
    labels = np.zeros(G.number_of_nodes())
    for idx, comm in enumerate(communities):
        for node in comm: labels[node] = idx
    return labels

# --- [模擬實驗與繪圖] ---

n_nodes, k_neighbors = 60, 4
probs = [0.01, 0.05, 0.1, 0.2, 0.4]
ground_truth = np.array([i // 10 for i in range(n_nodes)]) # 預期分組

results = []
for p in probs:
    G = nx.Graph()
    lattice = [tuple(sorted((i, (i + j) % n_nodes))) for j in range(1, k_neighbors // 2 + 1) for i in range(n_nodes)]
    G.add_edges_from(lattice); lattice_set = set(lattice)
    shortcuts = [tuple(sorted((u, v))) for u in range(n_nodes) for v in range(u+1, n_nodes) if not G.has_edge(u, v) and np.random.random() < p]
    G.add_edges_from(shortcuts); shortcut_set = set(shortcuts)
    
    types_h = run_edge_analysis(G, is_ldheta=False); types_l = run_edge_analysis(G, is_ldheta=True)
    
    # Accuracy
    acc_h = sum(1 for e, t in types_h.items() if (e in lattice_set and t == "Bond") or (e in shortcut_set and t == "Bridge")) / len(G.edges())
    acc_l = sum(1 for e, t in types_l.items() if (e in lattice_set and t == "Bond") or (e in shortcut_set and t == "Bridge")) / len(G.edges())
    
    # NMI
    nmi_h = normalized_mutual_info_score(ground_truth, get_partition_labels(G, types_h))
    nmi_l = normalized_mutual_info_score(ground_truth, get_partition_labels(G, types_l))
    
    results.append({"p": p, "HETA_Acc": acc_h, "LDHETA_Acc": acc_l, "HETA_NMI": nmi_h, "LDHETA_NMI": nmi_l})

df = pd.DataFrame(results)

# 產生圖片邏輯
plt.figure(figsize=(10, 5))
plt.plot(df['p'], df['HETA_Acc'], 'o-', label='HETA Acc'); plt.plot(df['p'], df['LDHETA_Acc'], 's--', label='LDHETA Acc')
plt.title('Accuracy Comparison'); plt.legend(); plt.savefig('accuracy_comparison_plot.png')

plt.figure(figsize=(10, 5))
plt.plot(df['p'], df['HETA_NMI'], 'o-', label='HETA NMI'); plt.plot(df['p'], df['LDHETA_NMI'], 's--', label='LDHETA NMI')
plt.title('NMI Comparison'); plt.legend(); plt.savefig('nmi_comparison_plot.png')

print(df)