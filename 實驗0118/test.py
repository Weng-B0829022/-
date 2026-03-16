import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_ldheta_global_experiment():
    G = nx.Graph()
    # 建立異質密度網路
    for i in range(1, 6): # 密集區
        for j in range(i + 1, 6): G.add_edge(i, j)
    G.add_edges_from([(6,7), (7,8), (6,8), (8,9), (9,10), (8,10)]) # 稀疏區
    G.add_edge(5, 6) # 這是一條明顯的 Global Bridge

    edges = list(G.edges())
    
    # 計算 R_ij 並自動取得 HETA 門檻 T
    r_values = []
    for u, v in edges:
        u_n, v_n = set(G.neighbors(u)), set(G.neighbors(v))
        common = len(u_n.intersection(v_n))
        min_deg = min(len(u_n), len(v_n))
        r_values.append(common / min_deg if min_deg > 0 else 0)
    
    T_heta = np.mean(r_values)
    
    # 計算 D_ij
    d_map = {}
    for u, v in edges:
        scope = set(G.neighbors(u)).union(set(G.neighbors(v))).union({u, v})
        sub = G.subgraph(scope)
        d_map[(u, v)] = sub.number_of_edges() / (sub.number_of_nodes() * (sub.number_of_nodes() - 1) / 2)
    avg_d = np.mean(list(d_map.values()))

    results = []
    for i, (u, v) in enumerate(edges):
        r_ij, d_ij = r_values[i], d_map[(u, v)]
        g_ij = d_ij / avg_d
        adaptive_t = T_heta * g_ij
        
        # 新增 Global 判定邏輯
        if r_ij == 0:
            final_type = 'Global Bridge'
        elif r_ij >= adaptive_t:
            final_type = 'Bond'
        else:
            final_type = 'Local Bridge'
            
        results.append({'u': u, 'v': v, 'R_ij': r_ij, 'Type': final_type})

    # 繪圖
    df = pd.DataFrame(results)
    pos = nx.spring_layout(G, seed=42)
    
    # 顏色：紅(Bond), 藍(Local Bridge), 綠(Global Bridge)
    color_map = {'Bond': 'red', 'Local Bridge': 'blue', 'Global Bridge': 'green'}
    colors = [color_map[t] for t in df['Type']]

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightgray', edge_color=colors, width=3)
    plt.title("LDHETA with Global Bridge Detection\nRed: Bond, Blue: Local Bridge, Green: Global Bridge")
    plt.show()

run_ldheta_global_experiment()