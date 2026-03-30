import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ════════════════════════════════════════════════════════
# 1. 共同鄰居計算
# ════════════════════════════════════════════════════════

def get_kth_layer_neighbors(G, node, exclude, k):
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
                visited.add(nb)
    return kth


def compute_r_uv_k(G, u, v, k):
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
    rand_ratios = []
    for _ in range(n_random):
        Gr = switching_randomization(G)
        for (u, v) in Gr.edges():
            rand_ratios.append(compute_r_uv_k(Gr, u, v, k))
    if not rand_ratios:
        return 0.8
    return float(np.mean(rand_ratios) + 2 * np.std(rand_ratios))


def compute_internal_threshold(candidate_ratios):
    if not candidate_ratios:
        return 0.0
    arr = np.array(candidate_ratios)
    return float(np.mean(arr) - np.std(arr))


# ════════════════════════════════════════════════════════
# 5. R-B 混合機制（Layer 3b）
# ════════════════════════════════════════════════════════

def is_ti_degenerate(candidate_ratios):
    """
    偵測 T_I 是否退化。
    退化條件：所有候選邊 R 值相同（SD = 0），
    最常見是全為 0 → T_I = 0，導致嚴格大於條件全部不成立。
    """
    arr = np.array(candidate_ratios)
    return float(np.std(arr)) == 0.0


def compute_betweenness_threshold(G, candidate_edges):
    """
    計算 R-B 混合機制的補充閾值 T_B。

    步驟：
      1. 計算全圖 edge betweenness（NetworkX 已正規化）
      2. 只取 candidate_edges 的 B 值
      3. T_B = Median(B) - MAD(B)

    回傳：(b_values_dict, T_B)
    """
    # NetworkX edge_betweenness_centrality 回傳 {(u,v): float}
    # normalized=False 讓數值更好解讀（路徑數量）
    all_eb = nx.edge_betweenness_centrality(G, normalized=False)

    # 統一 key 方向（NetworkX 有時 (u,v) 有時 (v,u)）
    eb = {}
    for (u, v), val in all_eb.items():
        eb[(u, v)] = val
        eb[(v, u)] = val

    b_vals = np.array([eb.get(e, eb.get((e[1], e[0]), 0.0)) for e in candidate_edges])

    median_b = float(np.median(b_vals))
    mad_b    = float(np.median(np.abs(b_vals - median_b)))
    T_B      = median_b - mad_b

    return b_vals, T_B


# ════════════════════════════════════════════════════════
# 6. 主演算法：heta_rb（含 R-B 混合機制）
# ════════════════════════════════════════════════════════

def heta_rb(G, n_random=100):
    """
    HETA + Layer 3b R-B 混合機制。

    相較原始 heta_analysis：
      - Step 3.2 每層計算 T_I 後，若偵測到退化，
        啟動 betweenness 補充分類。
      - B > T_B  → GLOBAL（提前確認為重要橋梁）
      - B <= T_B → 繼續往下層（可能是葉枝小邊）

    回傳 dict：{ (u,v): 'BOND'|'LOCAL'|'GLOBAL'|'SILK' }
    附帶 rb_log：記錄每次 R-B 介入的層數與統計資訊
    """
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    kmax  = compute_kmax(G)
    edges = list(G.edges())

    print(f"  kmax = {kmax}")

    link_types = {e: 'UNDEFINED' for e in edges}
    pass_flag  = {e: True        for e in edges}
    rb_log     = []   # 記錄每次 R-B 介入資訊

    # ── Step 3.1: Silk Links ────────────────────────────
    for u, v in edges:
        if G.degree(u) == 1 or G.degree(v) == 1:
            link_types[(u, v)] = 'SILK'
            pass_flag[(u, v)]  = False

    # ── Step 2: 預先計算各層外部閾值 ────────────────────
    ext_thresholds = {}
    for k in range(1, kmax + 1):
        ext_thresholds[k] = compute_external_threshold(G, k, n_random)

    # ── Step 3.2: 逐層識別 Bond / Local（含 R-B 補充）──
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

        if not candidate_edges:
            continue

        T_I = compute_internal_threshold(candidate_ratios)

        # ── Layer 3b 介入點：偵測 T_I 退化 ──────────────
        if is_ti_degenerate(candidate_ratios):
            b_vals, T_B = compute_betweenness_threshold(G, candidate_edges)

            rb_activated = []
            for e, b in zip(candidate_edges, b_vals):
                if b > T_B:
                    link_types[e] = 'GLOBAL'
                    pass_flag[e]  = False
                    rb_activated.append((e, b))
                # else: 繼續往下層，pass_flag 保持 True

            rb_log.append({
                'layer':          k,
                'n_candidates':   len(candidate_edges),
                'T_I':            T_I,
                'T_B':            T_B,
                'median_B':       float(np.median(b_vals)),
                'mad_B':          float(np.median(np.abs(b_vals - np.median(b_vals)))),
                'rb_classified':  len(rb_activated),
            })

            print(f"  [R-B 介入] layer={k}  候選邊={len(candidate_edges)}  "
                  f"T_I={T_I:.4f}(退化)  T_B={T_B:.2f}  "
                  f"→ {len(rb_activated)} 條提前判為 GLOBAL")

        else:
            # T_I 正常：走原始邏輯
            for e, r in zip(candidate_edges, candidate_ratios):
                if r > T_I:
                    link_types[e] = 'LOCAL'
                    pass_flag[e]  = False

    # ── Step 3.3: Global Bridge（所有層都未分類的邊）────
    for e in edges:
        if pass_flag[e]:
            link_types[e] = 'GLOBAL'

    return link_types, rb_log


# ════════════════════════════════════════════════════════
# 7. 統計摘要
# ════════════════════════════════════════════════════════

def summarize(link_types, label=""):
    counts = {'BOND': 0, 'LOCAL': 0, 'GLOBAL': 0, 'SILK': 0}
    for t in link_types.values():
        if t in counts:
            counts[t] += 1
    total = sum(counts.values())
    if label:
        print(f"\n── {label} ──")
    for k, v in counts.items():
        pct = round(v / total * 100, 1) if total else 0
        print(f"  {k:>6} : {v:>3} 條  ({pct}%)")
    return counts


# ════════════════════════════════════════════════════════
# 8. 繪圖
# ════════════════════════════════════════════════════════

def plot_heta_graph(G, link_types, title="HETA Link Type Identification"):
    pos = nx.spring_layout(G, seed=42)
    color_map = {
        'BOND':   'royalblue',
        'LOCAL':  'tomato',
        'GLOBAL': 'seagreen',
        'SILK':   'orange',
        'UNDEFINED': 'black',
    }
    edge_colors = [color_map.get(link_types.get(e, 'UNDEFINED'), 'black')
                   for e in G.edges()]
    plt.figure(figsize=(11, 7))
    nx.draw(G, pos, with_labels=False, node_color='#222222',
            edge_color=edge_colors, width=2, node_size=35,
            edgecolors='#222222')
    legend_elements = [
        Line2D([0], [0], color='royalblue', lw=2, label='Bond link'),
        Line2D([0], [0], color='tomato',    lw=2, label='Local bridge'),
        Line2D([0], [0], color='seagreen',  lw=2, label='Global bridge'),
        Line2D([0], [0], color='orange',    lw=2, label='Silk link'),
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    plt.title(title, fontsize=13)
    plt.tight_layout()
    return plt


# ════════════════════════════════════════════════════════
# 9. 測試網路建構（來自 star.py）
# ════════════════════════════════════════════════════════

def build_star_hybrid_network(
    n_centers=3, tree_depth=2, branching=3,
    n_local_edges=4, n_shared_neighbors=0,
    n_bond_cliques=1, bond_clique_size=3,
):
    G = nx.Graph()
    centers = list(range(n_centers))
    for i in centers:
        for j in centers:
            if i < j:
                G.add_edge(i, j)
    node_id = n_centers
    for _ in range(n_shared_neighbors):
        sn = node_id; node_id += 1
        for c in centers:
            G.add_edge(c, sn)
    depth_nodes = {c: {0: [c]} for c in centers}
    for c in centers:
        for d in range(1, tree_depth + 1):
            depth_nodes[c][d] = []
            for parent in depth_nodes[c][d - 1]:
                for _ in range(branching):
                    child = node_id; node_id += 1
                    G.add_edge(parent, child)
                    depth_nodes[c][d].append(child)
    added = 0; attempts = 0; max_attempts = n_local_edges * 20
    candidate_pool = []
    for c in centers:
        leaves = depth_nodes[c][tree_depth]
        parent_of = {}
        for node in leaves:
            for p in G.neighbors(node):
                if p in depth_nodes[c].get(tree_depth - 1, [c]):
                    parent_of[node] = p; break
        groups = {}
        for node, p in parent_of.items():
            groups.setdefault(p, []).append(node)
        candidate_pool.append([v for v in groups.values() if len(v) >= 1])
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
            G.add_edge(na, nb); added += 1
    for _ in range(n_bond_cliques):
        clique_nodes = list(range(node_id, node_id + bond_clique_size))
        node_id += bond_clique_size
        for i in clique_nodes:
            for j in clique_nodes:
                if i < j:
                    G.add_edge(i, j)
        G.add_edge(random.choice(clique_nodes), random.choice(centers))
    return G


# ════════════════════════════════════════════════════════
# 10. 主程式
# ════════════════════════════════════════════════════════

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    print("=" * 55)
    print("  測試網路：build_star_hybrid_network(3, 3, 3, 10, 0, 0, 0)")
    print("  （純樹狀，無 shared neighbors，無 cliques）")
    print("=" * 55)

    G_test = build_star_hybrid_network(3, 3, 3, 10, 0, 0, 0)
    print(f"  節點數：{G_test.number_of_nodes()}，邊數：{G_test.number_of_edges()}")

    # ── 執行 R-B 混合版 HETA ────────────────────────────
    print("\n[執行 heta_rb]")
    results, rb_log = heta_rb(G_test, n_random=20)

    # ── 統計摘要 ─────────────────────────────────────────
    summarize(results, label="R-B 混合版分類結果")

    # ── R-B 介入摘要 ─────────────────────────────────────
    if rb_log:
        print("\n── R-B 介入日誌 ──")
        for entry in rb_log:
            print(f"  layer {entry['layer']}：候選邊 {entry['n_candidates']} 條 | "
                  f"T_I={entry['T_I']:.4f}(退化) | "
                  f"Median(B)={entry['median_B']:.1f} | "
                  f"MAD(B)={entry['mad_B']:.1f} | "
                  f"T_B={entry['T_B']:.1f} | "
                  f"→ {entry['rb_classified']} 條判為 GLOBAL")
    else:
        print("\n  （本次未觸發 R-B 介入）")

    # ── 繪圖並儲存 ───────────────────────────────────────
    fig = plot_heta_graph(
        G_test, results,
        title="HETA + R-B 混合機制｜Star Hybrid Network (3,3,3,10,0,0,0)"
    )
    fig.savefig("/mnt/user-data/outputs/heta_rb_result.png", dpi=150, bbox_inches='tight')
    print("\n  圖片已儲存。")