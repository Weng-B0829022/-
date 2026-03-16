import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import itertools

def calculate_rij(G):
    """計算每一條邊的共同鄰居比例 R_ij"""
    rij_dict = {}
    for u, v in G.edges():
        neighbors_u = set(G.neighbors(u))
        neighbors_v = set(G.neighbors(v))
        common = neighbors_u.intersection(neighbors_v)
        denom = min(len(neighbors_u), len(neighbors_v))
        rij_dict[(u, v)] = len(common) / denom if denom > 0 else 0
    return rij_dict

# 1. 建立網路基礎架構
G = nx.Graph()
nodes_h1 = list(range(0, 10))
nodes_h2 = list(range(10, 20))
nodes_ld = list(range(20, 32))

# 建立社群內部連線
G.add_edges_from(nx.complete_graph(nodes_h1).edges())
G.add_edges_from(nx.complete_graph(nodes_h2).edges())

# 密集區內部連線減少 (HD區減少 20%+20%)
random.seed(42)
for _ in range(2):
    current_edges = list(G.edges())
    # 只針對 h1 和 h2 內部的邊進行修剪
    internal_edges = [e for e in current_edges if (e[0] in nodes_h1 and e[1] in nodes_h1) or (e[0] in nodes_h2 and e[1] in nodes_h2)]
    G.remove_edges_from(random.sample(internal_edges, int(len(internal_edges) * 0.2)))

# 跨社群連線 (30 條)
np.random.seed(99)
inter_count = 30
added = 0
while added < inter_count:
    u, v = np.random.choice(nodes_h1), np.random.choice(nodes_h2)
    if not G.has_edge(u, v):
        G.add_edge(u, v)
        added += 1

# --- 低密度區處理 ---
nx.add_cycle(G, nodes_ld)
G.add_edge(9, 20) # 連接橋樑

# 【新增修改】：讓稀疏區稍微密集一點點 (額外多 5 條邊)
potential_ld_edges = list(itertools.combinations(nodes_ld, 2))
existing_ld_edges = set(G.edges())
# 過濾掉已存在的邊
possible_to_add = [e for e in potential_ld_edges if e not in existing_ld_edges and (e[1], e[0]) not in existing_ld_edges]

random.seed(101) 
extra_edges = random.sample(possible_to_add, 5)
G.add_edges_from(extra_edges)

# 2. 佈局計算與右側子群調整
pos = nx.spring_layout(G, k=0.5, iterations=150, seed=42)

# 右側子群減少 10% (延續原始要求)
target_nodes = nodes_h2 if np.mean([pos[n][0] for n in nodes_h2]) > np.mean([pos[n][0] for n in nodes_h1]) else nodes_h1
target_edges = [(u, v) for u, v in G.edges() if u in target_nodes and v in target_nodes]
random.seed(44)
G.remove_edges_from(random.sample(target_edges, int(len(target_edges) * 0.1)))

# 3. 計算全域指標 R_ij
rij_values = calculate_rij(G)
threshold = np.mean(list(rij_values.values()))

# 4. 佈局擴張 (讓 HD 區節點散開一點)
coords = np.array([pos[n] for n in range(0, 20)])
centroid = coords.mean(axis=0)
for n in range(0, 20):
    pos[n] = centroid + 2.2 * (pos[n] - centroid)

# 5. 繪圖
plt.figure(figsize=(14, 10))

# 繪製節點 (一致黑色)
nx.draw_networkx_nodes(G, pos, node_color='black', node_size=400)

# 繪製邊 (強紅弱綠)
for (u, v) in G.edges():
    r = rij_values.get((u, v), rij_values.get((v, u), 0))
    is_bond = r >= threshold
    color = 'red' if is_bond else 'green'
    width = 2.5 
    alpha = 0.8 if is_bond else 0.4
    
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                           edge_color=color, 
                           width=width, 
                           alpha=alpha)

plt.title("Network with Enhanced Sparse Region (+5 Edges)", fontsize=15)
plt.axis('off')
plt.show()