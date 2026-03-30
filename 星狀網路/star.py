import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ════════════════════════════════════════════════════════
# 1. 共同鄰居計算
# ════════════════════════════════════════════════════════

def get_kth_layer_neighbors(G, node, exclude, k):
    """取得節點在第 k 層的鄰居集合"""
    if k == 1:
        return set(G.neighbors(node)) - {exclude}

    visited = {node, exclude}
    frontier = {node}

    for _ in range(1, k):
        next_frontier = set()
        for n in frontier:
            for nb in G.neighbors(n):
                if nb not in visited:
                    next_frontier.add(nb)
                    visited.add(nb)
        frontier = next_frontier

    kth = set()
    for n in frontier:
        for nb in G.neighbors(n):
            if nb not in visited:
                kth.add(nb)
                visited.add(nb)   # ✅ Bug fix：原版缺少此行，導致重複計算
    return kth


def compute_r_uv_k(G, u, v, k):
    """計算共同鄰居比例 R_uv^k（對應 project 版的 compute_common_neighbor_ratio）"""
    if k == 1:
        v1_u = set(G.neighbors(u)) - {v}
        v1_v = set(G.neighbors(v)) - {u}
        if not v1_u or not v1_v:
            return 0.0
        return len(v1_u & v1_v) / min(len(v1_u), len(v1_v))

    vk_u  = get_kth_layer_neighbors(G, u, v, k)
    vk_v  = get_kth_layer_neighbors(G, v, u, k)
    vk1_u = get_kth_layer_neighbors(G, u, v, k - 1)
    vk1_v = get_kth_layer_neighbors(G, v, u, k - 1)

    # ✅ 與 project 版一致：先檢查 union 是否為空
    union = (vk_u & vk_v) | (vk1_u & vk_v) | (vk_u & vk1_v)
    if not union:
        return 0.0

    num = len(vk_u & vk_v) + len(vk1_u & vk_v) + len(vk_u & vk1_v)
    den = (min(len(vk_u), len(vk_v)) +
           min(len(vk1_u), len(vk_v)) +
           min(len(vk_u), len(vk1_v)))
    return num / den if den > 0 else 0.0


# ════════════════════════════════════════════════════════
# 2. kmax 計算
# ════════════════════════════════════════════════════════

def compute_kmax(G):
    """kmax = floor(平均最短路徑 / 2)，最小為 1"""
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    try:
        avg_spl = nx.average_shortest_path_length(G)
    except Exception:
        return 1
    return max(1, int(avg_spl / 2))


# ════════════════════════════════════════════════════════
# 3. 隨機化網路生成
# ════════════════════════════════════════════════════════

def switching_randomization(G, Q=100):
    """隨機化網路生成（對應 project 版的 switching_randomize）"""
    Gr = G.copy()
    edges = list(Gr.edges())
    m = len(edges)

    for _ in range(Q * m):
        if m < 2:
            break
        i, j = random.sample(range(m), 2)
        (a, b), (c, d) = edges[i], edges[j]
        if len({a, b, c, d}) == 4 and not (Gr.has_edge(a, d) or Gr.has_edge(c, b)):
            Gr.remove_edge(a, b)
            Gr.remove_edge(c, d)
            Gr.add_edge(a, d)
            Gr.add_edge(c, b)
            edges[i], edges[j] = (a, d), (c, b)
    return Gr


# ════════════════════════════════════════════════════════
# 4. 閾值計算
# ════════════════════════════════════════════════════════

def compute_external_threshold(G, k, n_random=100):
    """
    計算第 k 層外部閾值 T_E^k。
    對應論文公式 (5)：T_E^k = Mean + 2 * SD（基於隨機化網路）
    """
    rand_ratios = []
    for _ in range(n_random):
        Gr = switching_randomization(G)
        for (u, v) in Gr.edges():
            rand_ratios.append(compute_r_uv_k(Gr, u, v, k))
    if not rand_ratios:
        return 0.8
    return float(np.mean(rand_ratios) + 2 * np.std(rand_ratios))


def compute_internal_threshold(candidate_ratios):
    """
    計算內部閾值 T_I^k。
    對應論文公式 (6)：T_I^k = Mean - SD（基於 candidate 邊）
    """
    if not candidate_ratios:
        return 0.0
    arr = np.array(candidate_ratios)
    print(f"T_I^k = {np.mean(arr) - np.std(arr)}")
    return float(np.mean(arr) - np.std(arr))


# ════════════════════════════════════════════════════════
# 5. 主演算法：heta_analysis
# ════════════════════════════════════════════════════════

def heta_analysis(G, n_random=100):
    """
    HETA 識別演算法（對應 project 版的 heta()）

    回傳 dict：{ (u,v): 'BOND' | 'LOCAL' | 'GLOBAL' | 'SILK' }
    """
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    kmax  = compute_kmax(G)
    edges = list(G.edges())

    print(f"kmax = {kmax}")   # ✅ Bug fix：移到迴圈外，只印一次

    link_types = {e: 'UNDEFINED' for e in edges}
    pass_flag  = {e: True        for e in edges}

    # Step 3.1: Silk Links
    for u, v in edges:
        if G.degree(u) == 1 or G.degree(v) == 1:
            link_types[(u, v)] = 'SILK'
            pass_flag[(u, v)]  = False

    # Step 2: 預先計算各層外部閾值
    ext_thresholds = {}
    for k in range(1, kmax + 1):
        ext_thresholds[k] = compute_external_threshold(G, k, n_random)

    # Step 3.2: 逐層識別 Bond / Local
    for k in range(1, kmax + 1):
        T_E = ext_thresholds[k]
        candidate_edges  = []
        candidate_ratios = []

        for e in edges:
            if not pass_flag[e]:
                continue
            r = compute_r_uv_k(G, e[0], e[1], k)
            if r >= T_E:
                link_types[e] = 'BOND'
                pass_flag[e]  = False
            else:
                candidate_edges.append(e)
                candidate_ratios.append(r)

        if candidate_edges:
            T_I = compute_internal_threshold(candidate_ratios)
            for e, r in zip(candidate_edges, candidate_ratios):
                if r > T_I:
                    link_types[e] = 'LOCAL'
                    pass_flag[e]  = False
                # r < T_I：保留 pass_flag=True，繼續下一層

    # Step 3.3: Global Bridge（所有層都未分類的邊）
    for e in edges:
        if pass_flag[e]:
            link_types[e] = 'GLOBAL'

    return link_types


# ════════════════════════════════════════════════════════
# 6. 統計摘要（對應 project 版的 summarize）
# ════════════════════════════════════════════════════════

def summarize(link_types):
    """統計四種類型的數量與比例"""
    counts = {'BOND': 0, 'LOCAL': 0, 'GLOBAL': 0, 'SILK': 0}
    for t in link_types.values():
        if t in counts:
            counts[t] += 1
    total = sum(counts.values())
    for k, v in counts.items():
        pct = round(v / total * 100, 1) if total else 0
        print(f"  {k:>6} : {v:>3} 條  ({pct}%)")


# ════════════════════════════════════════════════════════
# 7. 繪圖
# ════════════════════════════════════════════════════════

def plot_heta_graph(G, link_types):
    """繪製分類後的圖形"""
    pos = nx.spring_layout(G, seed=42)

    color_map = {
        'BOND':      'blue',
        'LOCAL':     'red',
        'GLOBAL':    'green',
        'SILK':      'orange',
        'UNDEFINED': 'black',
    }

    edge_colors = [color_map.get(link_types.get(edge, 'UNDEFINED'), 'black')
                   for edge in G.edges()]

    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=False, node_color='black',
            edge_color=edge_colors, width=2, node_size=30,
            edgecolors='black')

    legend_elements = [
        Line2D([0], [0], color='blue',   lw=2, label='Bond link'),
        Line2D([0], [0], color='red',    lw=2, label='Local bridge'),
        Line2D([0], [0], color='green',  lw=2, label='Global bridge'),
        Line2D([0], [0], color='orange', lw=2, label='Silk link'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title("HETA Link Type Identification")
    plt.tight_layout()
    plt.show()
import networkx as nx

def build_star_hybrid_network(
    n_centers: int = 3,
    tree_depth: int = 2,
    branching: int = 3,
    n_local_edges: int = 4,
    n_shared_neighbors: int = 0,
    n_bond_cliques: int = 1,
    bond_clique_size: int = 3,
):
    """
    參數
    ----
    n_centers          : 中心節點數量（完全圖）
    tree_depth         : 樹狀層數
    branching          : 每層分枝數
    n_local_edges      : 隨機橫向邊數量（local bridge）
    n_shared_neighbors : 每對中心節點共享的公共鄰居數
                         （拉高 R 值，確保中心邊被判為 bond link）
    n_bond_cliques     : 額外植入的小完全圖數量（高共同鄰居 → bond link）
    bond_clique_size   : 每個小完全圖的節點數（建議 4~6）
    """
    import random

    G = nx.Graph()

    # ── 核心層：完全圖 ───────────────────────────────────────
    centers = list(range(n_centers))
    for i in centers:
        for j in centers:
            if i < j:
                G.add_edge(i, j)

    node_id = n_centers

    # ── 共享公共鄰居（確保中心邊 R 值高 → bond link）────────
    for _ in range(n_shared_neighbors):
        sn = node_id; node_id += 1
        for c in centers:
            G.add_edge(c, sn)

    # ── 樹狀層 ───────────────────────────────────────────────
    depth_nodes = {c: {0: [c]} for c in centers}

    for c in centers:
        for d in range(1, tree_depth + 1):
            depth_nodes[c][d] = []
            for parent in depth_nodes[c][d - 1]:
                for _ in range(branching):
                    child = node_id; node_id += 1
                    G.add_edge(parent, child)
                    depth_nodes[c][d].append(child)

    # ── 橫向邊：同中心、不同父節點的末層節點對 ──────────────
    added = 0
    attempts = 0
    max_attempts = n_local_edges * 20

    candidate_pool = []
    for c in centers:
        leaves = depth_nodes[c][tree_depth]
        parent_of = {}
        for node in leaves:
            for p in G.neighbors(node):
                if p in depth_nodes[c].get(tree_depth - 1, [c]):
                    parent_of[node] = p
                    break
        groups = {}
        for node, p in parent_of.items():
            groups.setdefault(p, []).append(node)
        group_list = [v for v in groups.values() if len(v) >= 1]
        candidate_pool.append(group_list)

    while added < n_local_edges and attempts < max_attempts:
        attempts += 1
        c = random.randint(0, n_centers - 1)
        groups = candidate_pool[c]
        if len(groups) < 2:
            continue
        gi, gj = random.sample(range(len(groups)), 2)
        na = random.choice(groups[gi])
        nb = random.choice(groups[gj])
        if na != nb and not G.has_edge(na, nb):
            G.add_edge(na, nb)
            added += 1

    # ── 隨機 bond cliques：植入小完全圖 ─────────────────────
    # 每個 clique 內部所有邊的 R^1 = 1.0（完全共同鄰居）
    # 再接一條錨邊連回主網路，確保連通性
    for _ in range(n_bond_cliques):
        clique_nodes = list(range(node_id, node_id + bond_clique_size))
        node_id += bond_clique_size

        # clique 內部完全連接
        for i in clique_nodes:
            for j in clique_nodes:
                if i < j:
                    G.add_edge(i, j)

        # 錨邊：隨機選一個中心節點連入，避免孤立子圖
        anchor = random.choice(centers)
        G.add_edge(random.choice(clique_nodes), anchor)

    return G

if __name__ == '__main__':
    random.seed(42)
    #存在問題的toymodel
    G_test = build_star_hybrid_network(3,3,3,10,0,0,0)

    #G_test = build_star_hybrid_network()

    # 條件 A：最小反例，候選池只剩 1 條 r=0 的邊
    #G_test = build_toy_model()


    print(f"kmax = {compute_kmax(G_test)}")
    results = heta_analysis(G_test, n_random=20)
    print("\n── 分類結果 ──")
    summarize(results)
    plot_heta_graph(G_test, results)