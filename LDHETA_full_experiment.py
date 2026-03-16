# ==========================================================
# LD-HETA 實驗完整程式
# Author: 翁旻醇
# Description:
#   本程式完整實作 HETA 與 LD-HETA 的比較實驗，
#   包含：
#       1. 密度異質網路生成
#       2. HETA / LD-HETA 演算法實作（Silk 先判）
#       3. 六張結果圖自動生成：
#           (1) HETA vs LD-HETA 可視化
#           (2) 各社群長條圖
#           (3) 全體分布長條圖
#           (4) 密度 vs CNR 散點圖
#           (5) CNR 分布變化圖
#           (6) 橋邊可視化圖
# ==========================================================

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
from collections import Counter

# ----------------------------------------------------------
# 1. 生成密度異質網路 (SBM)
# ----------------------------------------------------------
def generate_heterogeneous_network():
    """
    建立三社群、密度不同的隨機網路
    高密：0.35、中密：0.30、低密：0.05
    """
    sizes = [60, 60, 120]
    p = [
        [0.35, 0.01, 0.002],
        [0.01, 0.30, 0.002],
        [0.002, 0.002, 0.05],
    ]
    G = nx.stochastic_block_model(sizes, p, seed=42)
    return G


# ----------------------------------------------------------
# 2. k-hop 鄰居集合
# ----------------------------------------------------------
def k_hop_neighbors(G, node, k):
    """
    回傳節點 node 的 k-hop 鄰居集合
    """
    nodes = {node}
    for _ in range(k):
        nbrs = set()
        for n in nodes:
            nbrs.update(G.neighbors(n))
        nodes.update(nbrs)
    nodes.discard(node)
    return nodes


# ----------------------------------------------------------
# 3. CNR(k) 計算
# ----------------------------------------------------------
def cnr_k(G, u, v, k):
    """
    計算邊 (u,v) 在 k-hop 下的共同鄰居比例 CNR
    """
    Nu = k_hop_neighbors(G, u, k)
    Nv = k_hop_neighbors(G, v, k)
    inter = Nu & Nv
    denom = min(len(Nu), len(Nv))
    return len(inter) / denom if denom > 0 else 0.0


# ----------------------------------------------------------
# 4. 節點局部密度 (ego-network density)
# ----------------------------------------------------------
def ego_density(G, node):
    """
    取 node 的 ego-network 密度
    """
    ego = nx.ego_graph(G, node, radius=1)
    n, m = ego.number_of_nodes(), ego.number_of_edges()
    if n <= 1:
        return 0.0
    return (2 * m) / (n * (n - 1))


# ----------------------------------------------------------
# 5. 使用 KMeans 自動求 β₁, β₂
# ----------------------------------------------------------
def find_beta_thresholds(cnr_values):
    """
    將所有邊的 CNR(k=1) 三分群，取群中心平均求出 β₁, β₂
    """
    X = np.array(cnr_values).reshape(-1, 1)
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    centers = sorted(km.cluster_centers_.flatten())
    beta1 = (centers[1] + centers[2]) / 2
    beta2 = (centers[0] + centers[1]) / 2
    return beta1, beta2


# ----------------------------------------------------------
# 6. HETA 分類 (Silk 先判斷)
# ----------------------------------------------------------
def heta_classify(G, k_max=3):
    """
    HETA：以全域門檻 β₁, β₂ 判定邊型
    Silk：先判（若任一端度數 ≤1）
    """
    cnr_dict = {(u, v): [] for u, v in G.edges()}
    for k in range(1, k_max + 1):
        for u, v in G.edges():
            cnr_dict[(u, v)].append(cnr_k(G, u, v, k))

    cnr_all = [vals[0] for vals in cnr_dict.values()]
    beta1, beta2 = find_beta_thresholds(cnr_all)

    edge_class = {}
    for (u, v), cnrs in cnr_dict.items():
        r1 = cnrs[0]
        deg_u, deg_v = G.degree[u], G.degree[v]

        # Silk: 若一端為末端節點
        if deg_u <= 1 or deg_v <= 1:
            edge_class[(u, v)] = "silk"
            continue

        # 依門檻分類
        if r1 >= beta1:
            edge_class[(u, v)] = "bond"
        elif r1 >= beta2:
            edge_class[(u, v)] = "local"
        else:
            edge_class[(u, v)] = "global"

    return edge_class, beta1, beta2


# ----------------------------------------------------------
# 7. LD-HETA (密度補償版)
# ----------------------------------------------------------
def ldheta_classify(G, beta1, beta2, k_max=3, base=0.15):
    """
    LD-HETA：在 HETA 架構下加入局部密度補償
    強邊在高密區被壓低、低密區被放大
    """
    local_d = {n: ego_density(G, n) for n in G.nodes()}
    edge_class = {}

    for u, v in G.edges():
        deg_u, deg_v = G.degree[u], G.degree[v]
        cnrs = []
        for k in range(1, k_max + 1):
            r = cnr_k(G, u, v, k)
            scale = max(local_d[u], local_d[v])
            adj_r = r / (base + scale)
            cnrs.append(adj_r)

        r1 = cnrs[0]

        # Silk：度數小的外圍邊
        if deg_u <= 1 or deg_v <= 1:
            edge_class[(u, v)] = "silk"
            continue

        # 依補償後 CNR 分類
        if r1 >= beta1:
            edge_class[(u, v)] = "bond"
        elif r1 >= beta2:
            edge_class[(u, v)] = "local"
        else:
            edge_class[(u, v)] = "global"

    return edge_class


# ----------------------------------------------------------
# 8. 抽樣邊以便視覺化
# ----------------------------------------------------------
def reduce_edges(G, keep_ratio=0.5):
    """
    隨機保留部分邊，降低密度避免圖太擁擠
    """
    edges = list(G.edges())
    random.seed(42)
    keep_n = int(len(edges) * keep_ratio)
    keep_edges = random.sample(edges, keep_n)
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(keep_edges)
    if "partition" in G.graph:
        H.graph["partition"] = G.graph["partition"]
    return H


# ----------------------------------------------------------
# 9. 計算每社群邊型統計
# ----------------------------------------------------------
def count_edge_types_by_community(G, edge_class):
    partitions = G.graph["partition"]
    community_counts = []

    for i, nodes in enumerate(partitions):
        sub_edges = [(u, v) for (u, v) in G.edges() if u in nodes and v in nodes]
        type_count = {"bond": 0, "local": 0, "global": 0, "silk": 0}
        for e in sub_edges:
            etype = edge_class.get(e) or edge_class.get((e[1], e[0]))
            if etype:
                type_count[etype] += 1
        community_counts.append(type_count)
        print(f"\n社群 {i+1}（節點數 {len(nodes)}）:")
        for k, v in type_count.items():
            print(f"  {k:<6} = {v}")
    return community_counts


# ----------------------------------------------------------
# 10. 圖1：HETA vs LD-HETA 可視化
# ----------------------------------------------------------
def draw_compare(G, heta_edges, ldheta_edges):
    offset_positions = {}
    current_offset = 0
    for idx, nodes in enumerate(G.graph["partition"]):
        subG = G.subgraph(nodes)
        sub_pos = nx.spring_layout(subG, seed=idx, k=0.5, iterations=100)
        for n, p in sub_pos.items():
            sub_pos[n] = (p[0] + current_offset, p[1])
        offset_positions.update(sub_pos)
        current_offset += 3.0
    pos = offset_positions

    color_map = {"bond": "blue", "local": "red", "global": "green", "silk": "purple"}

    plt.figure(figsize=(12, 6))
    # 左：HETA
    plt.subplot(1, 2, 1)
    for (u, v), etype in heta_edges.items():
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color_map[etype], width=1, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, node_color="lightgray", node_size=30)
    plt.title("HETA (Formal Definition)")
    plt.axis("off")

    # 右：LD-HETA
    plt.subplot(1, 2, 2)
    for (u, v), etype in ldheta_edges.items():
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color_map[etype], width=1, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, node_color="lightgray", node_size=30)
    plt.title("LD-HETA (Density-Aware Formal)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("Fig1_HETA_vs_LDHETA.png", dpi=300)
    plt.show()


# ----------------------------------------------------------
# 11. 圖2：各社群邊型長條圖
# ----------------------------------------------------------
def plot_edge_type_bars(heta_counts, ldheta_counts, G):
    labels = ["Bond", "Local", "Global", "Silk"]
    x = np.arange(len(labels))
    width = 0.35
    partitions = G.graph["partition"]

    for idx in range(len(partitions)):
        heta_vals = [heta_counts[idx][t.lower()] for t in labels]
        ldheta_vals = [ldheta_counts[idx][t.lower()] for t in labels]

        plt.figure(figsize=(6, 4))
        plt.bar(x - width/2, heta_vals, width, label="HETA", color="#4A90E2", alpha=0.8)
        plt.bar(x + width/2, ldheta_vals, width, label="LD-HETA", color="#E94E77", alpha=0.8)
        plt.xticks(x, labels)
        plt.xlabel("Edge Type")
        plt.ylabel("Edge Count")
        plt.title(f"Community {idx+1} Edge Type Comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"Fig2_Community_{idx+1}_Bar.png", dpi=300)
        plt.close()
        print(f"✅ 已輸出 Fig2_Community_{idx+1}_Bar.png")


# ----------------------------------------------------------
# 12. 圖3：全體分類比例長條圖
# ----------------------------------------------------------
def plot_overall_distribution(heta_edges, ldheta_edges):
    labels = ["bond", "local", "global", "silk"]
    heta_counts = Counter(heta_edges.values())
    ldheta_counts = Counter(ldheta_edges.values())
    heta_vals = [heta_counts.get(t, 0) for t in labels]
    ldheta_vals = [ldheta_counts.get(t, 0) for t in labels]

    plt.figure(figsize=(6, 4))
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width/2, heta_vals, width, label="HETA", color="#4A90E2")
    plt.bar(x + width/2, ldheta_vals, width, label="LD-HETA", color="#E94E77")
    plt.xticks(x, [t.capitalize() for t in labels])
    plt.ylabel("Edge Count")
    plt.title("Overall Edge-Type Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Fig3_Overall_Distribution.png", dpi=300)
    plt.show()


# ----------------------------------------------------------
# 13. 圖4：邊密度 vs CNR 散點圖
# ----------------------------------------------------------
def plot_density_vs_cnr(G):
    cnrs, densities = [], []
    for u, v in G.edges():
        cnrs.append(cnr_k(G, u, v, 1))
        densities.append(max(ego_density(G, u), ego_density(G, v)))

    plt.figure(figsize=(6, 4))
    plt.scatter(densities, cnrs, alpha=0.4, s=10, c="gray")
    plt.xlabel("Local Density of Edge")
    plt.ylabel("CNR (k=1)")
    plt.title("Relationship between Local Density and Edge Strength")
    plt.tight_layout()
    plt.savefig("Fig4_Density_vs_CNR.png", dpi=300)
    plt.show()


# ----------------------------------------------------------
# 14. 圖5：CNR 分布變化
# ----------------------------------------------------------
def plot_cnr_distribution(G):
    cnr_raw = [cnr_k(G, u, v, 1) for u, v in G.edges()]
    densities = {n: ego_density(G, n) for n in G.nodes()}
    cnr_adj = [cnr / (0.15 + max(densities[u], densities[v])) for u, v in G.edges()]

    plt.figure(figsize=(6, 4))
    plt.hist(cnr_raw, bins=30, alpha=0.5, label="HETA CNR")
    plt.hist(cnr_adj, bins=30, alpha=0.5, label="LD-HETA CNR'")
    plt.xlabel("CNR Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Distribution of CNR Before and After Density Correction")
    plt.tight_layout()
    plt.savefig("Fig5_CNR_Distribution.png", dpi=300)
    plt.show()


# ----------------------------------------------------------
# 15. 圖6：橋邊可視化 (Global / Silk)
# ----------------------------------------------------------
def draw_bridges(G, heta_edges, ldheta_edges):
    pos = nx.spring_layout(G, seed=42, k=0.3, iterations=150)
    plt.figure(figsize=(12, 6))

    # HETA 桥
    plt.subplot(1, 2, 1)
    bridge_edges = [e for e, t in heta_edges.items() if t in ["global", "silk"]]
    nx.draw(G, pos, node_color="gray", node_size=20, edgelist=bridge_edges, edge_color="red")
    plt.title("HETA - Bridge/Silk Edges")

    # LD-HETA 桥
    plt.subplot(1, 2, 2)
    bridge_edges = [e for e, t in ldheta_edges.items() if t in ["global", "silk"]]
    nx.draw(G, pos, node_color="gray", node_size=20, edgelist=bridge_edges, edge_color="red")
    plt.title("LD-HETA - Bridge/Silk Edges")

    plt.tight_layout()
    plt.savefig("Fig6_Bridge_Comparison.png", dpi=300)
    plt.show()


# ----------------------------------------------------------
# 16. 主程式執行區
# ----------------------------------------------------------
if __name__ == "__main__":
    G = generate_heterogeneous_network()
    G_reduced = reduce_edges(G, keep_ratio=0.5)
    print(f"原始邊數: {G.number_of_edges()} → 保留後: {G_reduced.number_of_edges()}")

    # 執行 HETA
    heta_edges, b1, b2 = heta_classify(G_reduced, k_max=3)
    print(f"HETA β₁={b1:.3f}, β₂={b2:.3f}")

    # 執行 LD-HETA (使用相同門檻)
    ldheta_edges = ldheta_classify(G_reduced, b1, b2, k_max=3, base=0.15)

    # 統計結果
    print("\n=== HETA 各社群邊型數量 ===")
    heta_counts = count_edge_types_by_community(G_reduced, heta_edges)

    print("\n=== LD-HETA 各社群邊型數量 ===")
    ldheta_counts = count_edge_types_by_community(G_reduced, ldheta_edges)

    # 繪圖
    draw_compare(G_reduced, heta_edges, ldheta_edges)
    plot_edge_type_bars(heta_counts, ldheta_counts, G_reduced)
    plot_overall_distribution(heta_edges, ldheta_edges)
    plot_density_vs_cnr(G_reduced)
    plot_cnr_distribution(G_reduced)
    draw_bridges(G_reduced, heta_edges, ldheta_edges)

    print("\n✅ 所有圖已輸出完成！")
