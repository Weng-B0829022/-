import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random

# ==== 1. 生成密度異質玩具網路 ====
def generate_heterogeneous_network():
    sizes = [60, 60, 120]
    p = [
        [0.35, 0.01, 0.002],
        [0.01, 0.30, 0.002],
        [0.002, 0.002, 0.05],   # 第三團仍為低密區
    ]
    G = nx.stochastic_block_model(sizes, p, seed=42)
    return G


# ==== 2. k-hop 鄰居 ====
def k_hop_neighbors(G, node, k):
    nodes = {node}
    for _ in range(k):
        nbrs = set()
        for n in nodes:
            nbrs.update(G.neighbors(n))
        nodes.update(nbrs)
    nodes.discard(node)
    return nodes


# ==== 3. CNR(k) ====
def cnr_k(G, u, v, k):
    Nu = k_hop_neighbors(G, u, k)
    Nv = k_hop_neighbors(G, v, k)
    inter = Nu & Nv
    denom = min(len(Nu), len(Nv))
    return len(inter) / denom if denom > 0 else 0.0


# ==== 4. 節點局部密度 ====
def ego_density(G, node):
    ego = nx.ego_graph(G, node, radius=1)
    n, m = ego.number_of_nodes(), ego.number_of_edges()
    if n <= 1:
        return 0.0
    return (2 * m) / (n * (n - 1))


# ==== 5. 自動求 β₁, β₂ ====
def find_beta_thresholds(cnr_values):
    X = np.array(cnr_values).reshape(-1, 1)
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    centers = sorted(km.cluster_centers_.flatten())
    beta1 = (centers[1] + centers[2]) / 2
    beta2 = (centers[0] + centers[1]) / 2
    return beta1, beta2


# ==== 6. 正式 HETA（修正版 Silk 先判斷） ====
def heta_classify(G, k_max=3):
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

        # 1️⃣ Silk：至少一端度數小於等於 1（外圍節點）
        if deg_u <= 1 or deg_v <= 1:
            edge_class[(u, v)] = "silk"
            continue

        # 2️⃣ 依 CNR 門檻分類
        if r1 >= beta1:
            edge_class[(u, v)] = "bond"
        elif r1 >= beta2:
            edge_class[(u, v)] = "local"   # local bridge
        else:
            edge_class[(u, v)] = "global"

    return edge_class, beta1, beta2


# ==== 7. LD-HETA（密度補償版，Silk 同樣先判斷） ====
def ldheta_classify(G, beta1, beta2, k_max=3, base=0.15):
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

        # 1️⃣ Silk：至少一端度數小於等於 1
        if deg_u <= 1 or deg_v <= 1:
            edge_class[(u, v)] = "silk"
            continue

        # 2️⃣ 根據補償後 CNR 分類
        if r1 >= beta1:
            edge_class[(u, v)] = "bond"
        elif r1 >= beta2:
            edge_class[(u, v)] = "local"
        else:
            edge_class[(u, v)] = "global"

    return edge_class


# ==== 8. 抽樣邊（減少密度） ====
def reduce_edges(G, keep_ratio=0.5):
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


# ==== 9. 統計每社群邊型 ====
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


# ==== 10. 畫 HETA vs LDHETA 圖 ====
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

    color_map = {
        "bond": "blue",
        "local": "red",
        "global": "green",
        "silk": "purple",
    }

    plt.figure(figsize=(12, 6))

    # 左：HETA
    plt.subplot(1, 2, 1)
    for (u, v), etype in heta_edges.items():
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               edge_color=color_map[etype], width=1, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, node_color="lightgray", node_size=30)
    plt.title("HETA (Formal Definition)")
    plt.axis("off")

    # 右：LD-HETA
    plt.subplot(1, 2, 2)
    for (u, v), etype in ldheta_edges.items():
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               edge_color=color_map[etype], width=1, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, node_color="lightgray", node_size=30)
    plt.title("LD-HETA (Density-Aware Formal)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("HETA_vs_LDHETA_correct.png", dpi=300)
    plt.show()


# ==== 11. 畫長條圖 ====
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
        plt.savefig(f"Community_{idx+1}_EdgeType_Bar_correct.png", dpi=300)
        plt.close()
        print(f"✅ 已輸出 Community_{idx+1}_EdgeType_Bar_correct.png")


# ==== 12. 主程式 ====
if __name__ == "__main__":
    G = generate_heterogeneous_network()
    G_reduced = reduce_edges(G, keep_ratio=0.5)
    print(f"原始邊數: {G.number_of_edges()} → 保留後: {G_reduced.number_of_edges()}")

    # HETA
    heta_edges, b1, b2 = heta_classify(G_reduced, k_max=3)
    print(f"HETA β₁={b1:.3f}, β₂={b2:.3f}")

    # LD-HETA（用相同 β）
    ldheta_edges = ldheta_classify(G_reduced, b1, b2, k_max=3, base=0.15)

    # 統計
    print("\n=== HETA 各社群邊型數量 ===")
    heta_counts = count_edge_types_by_community(G_reduced, heta_edges)

    print("\n=== LD-HETA 各社群邊型數量 ===")
    ldheta_counts = count_edge_types_by_community(G_reduced, ldheta_edges)

    # 視覺化
    draw_compare(G_reduced, heta_edges, ldheta_edges)
    plot_edge_type_bars(heta_counts, ldheta_counts, G_reduced)
