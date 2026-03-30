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
# 5. R-B 二分法機制（Layer 3b，alpha 線性插值版）
#
#  問題根源：原 R-B 只做「GLOBAL / 繼續」二分法，
#            沒有路徑產生 LOCAL，導致全局只有 GLOBAL。
#
#  修正設計：用單一閾值 T 做二分法，T 由 alpha 線性插值控制
#
#    T = (min(B) - ε) + alpha × (max(B) - min(B) + 2ε)
#
#    B > T  → GLOBAL
#    B ≤ T  → LOCAL（當前 k 層局部橋梁）
#
#  alpha 語意（保證端點行為）：
#    alpha = 0.0  → T < min(B) → 所有候選邊 > T → 全部 GLOBAL
#    alpha = 1.0  → T > max(B) → 所有候選邊 ≤ T → 全部 LOCAL
#    alpha ∈ (0,1) → T 在 [min, max] 之間線性移動，GLOBAL/LOCAL 連續過渡
#
#  設計優點：
#    · 端點行為保證：任何網路上 alpha=0 必為全 GLOBAL，alpha=1 必為全 LOCAL
#    · 不依賴 Mean/Std/Median/MAD，不受離散 B 值分布退化影響
#    · 語意直觀：alpha 就是「LOCAL 比例的旋鈕」
# ════════════════════════════════════════════════════════

def is_ti_degenerate(candidate_ratios):
    """
    偵測 T_I 是否退化。
    退化條件：所有候選邊 R 值的 SD = 0（全相同，最常見是全為 0）。
    """
    arr = np.array(candidate_ratios)
    return float(np.std(arr)) == 0.0


def compute_betweenness_threshold(G, candidate_edges, alpha=0.5):
    """
    計算 R-B 二分法的單一閾值 T。

    T = (min(B) - ε) + alpha × (max(B) - min(B) + 2ε)

    alpha=0 → T < min(B) → 全部 GLOBAL
    alpha=1 → T > max(B) → 全部 LOCAL
    alpha∈(0,1) → 線性插值，連續過渡

    回傳：(b_values_array, T, stats_dict)
    """
    all_eb = nx.edge_betweenness_centrality(G, normalized=False)

    # 統一 key 方向
    eb = {}
    for (u, v), val in all_eb.items():
        eb[(u, v)] = val
        eb[(v, u)] = val

    b_vals = np.array([eb.get(e, eb.get((e[1], e[0]), 0.0)) for e in candidate_edges])

    min_b    = float(np.min(b_vals))
    max_b    = float(np.max(b_vals))
    mean_b   = float(np.mean(b_vals))
    median_b = float(np.median(b_vals))
    eps = 1.0  # 保證端點嚴格包含/排除所有邊

    T = (min_b - eps) + alpha * (max_b - min_b + 2 * eps)

    stats = {
        'min_B': min_b, 'max_B': max_b,
        'mean_B': mean_b, 'median_B': median_b,
        'T': T,
    }
    return b_vals, T, stats


# ════════════════════════════════════════════════════════
# 6. 主演算法：heta_rb（含 R-B 三分法機制）
# ════════════════════════════════════════════════════════

def heta_rb(G, n_random=100, alpha=0.5):
    """
    HETA + Layer 3b R-B 線性插值二分法機制。

    參數
    ----
    alpha : float，範圍 [0.0, 1.0]
        控制 GLOBAL/LOCAL 比例的旋鈕：
            T = (min(B) - ε) + alpha × (max(B) - min(B) + 2ε)
        alpha=0.0 → T < min(B) → 所有候選邊判 GLOBAL（全 GLOBAL）
        alpha=1.0 → T > max(B) → 所有候選邊判 LOCAL（全 LOCAL）
        alpha∈(0,1) → T 線性移動，GLOBAL/LOCAL 連續過渡
        此保證在任何網路、任何層都成立。

    回傳
    ----
    link_types : dict { (u,v): 'BOND'|'LOCAL'|'GLOBAL'|'SILK' }
    rb_log     : list of dict，每次 R-B 介入的統計資訊
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha 必須在 [0, 1] 範圍內，收到 alpha={alpha}")

    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    kmax  = compute_kmax(G)
    edges = list(G.edges())

    print(f"  kmax = {kmax}")

    link_types = {e: 'UNDEFINED' for e in edges}
    pass_flag  = {e: True        for e in edges}
    rb_log     = []

    # ── Step 3.1: Silk Links ────────────────────────────
    for u, v in edges:
        if G.degree(u) == 1 or G.degree(v) == 1:
            link_types[(u, v)] = 'SILK'
            pass_flag[(u, v)]  = False

    # ── Step 2: 預先計算各層外部閾值 ────────────────────
    ext_thresholds = {}
    for k in range(1, kmax + 1):
        ext_thresholds[k] = compute_external_threshold(G, k, n_random)

    # ── Step 3.2: 逐層識別 Bond / Local（含 R-B 二分法）─
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

        # ── Layer 3b 介入點：T_I 退化時啟動 R-B 二分法 ──
        if is_ti_degenerate(candidate_ratios):
            b_vals, T, stats = compute_betweenness_threshold(
                G, candidate_edges, alpha=alpha
            )

            n_global = n_local = 0
            for e, b in zip(candidate_edges, b_vals):
                if b > T:
                    link_types[e] = 'GLOBAL'
                    pass_flag[e]  = False
                    n_global += 1
                else:
                    link_types[e] = 'LOCAL'
                    pass_flag[e]  = False
                    n_local += 1

            rb_log.append({
                'layer':        k,
                'n_candidates': len(candidate_edges),
                'T_I':          T_I,
                'alpha':        alpha,
                **stats,
                'rb_global':    n_global,
                'rb_local':     n_local,
            })

            print(f"  [R-B 二分法] layer={k}  候選邊={len(candidate_edges)}  "
                  f"T_I={T_I:.4f}(退化)  alpha={alpha}  "
                  f"min(B)={stats['min_B']:.0f}  max(B)={stats['max_B']:.0f}  T={T:.1f}  "
                  f"→ GLOBAL={n_global}  LOCAL={n_local}")

        else:
            # T_I 正常：走原始 HETA 邏輯
            for e, r in zip(candidate_edges, candidate_ratios):
                if r > T_I:
                    link_types[e] = 'LOCAL'
                    pass_flag[e]  = False

    # ── Step 3.3: 剩餘未分類 = Global Bridge ────────────
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

    print("=" * 60)
    print("  測試網路：build_star_hybrid_network(3, 4, 3, 10, 0, 0, 0)")
    print("  （純樹狀，4層，無 shared neighbors，無 cliques）")
    print("=" * 60)

    G_test = build_star_hybrid_network(3, 4, 3, 10, 0, 0, 0)
    print(f"  節點數：{G_test.number_of_nodes()}，邊數：{G_test.number_of_edges()}")

    # ── alpha 掃描：0.0=全GLOBAL, 1.0=全LOCAL，中間連續過渡
    alphas      = [0.0, 0.3, 0.7, 1.0]
    all_results = {}
    all_counts  = {}

    for a in alphas:
        print(f"\n{'─'*55}")
        print(f"  [alpha = {a}]")
        res, log = heta_rb(G_test, n_random=20, alpha=a)
        all_results[a] = res
        c = summarize(res, label=f"alpha={a} 分類結果")
        all_counts[a]  = c

        if log:
            print("  ── R-B 二分法日誌 ──")
            for entry in log:
                print(f"    layer {entry['layer']}：候選邊 {entry['n_candidates']} | "
                      f"min(B)={entry['min_B']:.0f}  max(B)={entry['max_B']:.0f}  T={entry['T']:.1f}  "
                      f"→ GLOBAL={entry['rb_global']}  LOCAL={entry['rb_local']}")

    # ── 2x2 對比圖 ───────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    color_map = {
        'BOND':  'royalblue', 'LOCAL': 'tomato',
        'GLOBAL':'seagreen',  'SILK':  'orange', 'UNDEFINED':'black',
    }
    pos = nx.spring_layout(G_test, seed=42)

    for ax, a in zip(axes.flat, alphas):
        res = all_results[a]
        ec  = [color_map.get(res.get(e, 'UNDEFINED'), 'black') for e in G_test.edges()]
        nx.draw(G_test, pos, ax=ax, with_labels=False,
                node_color='#222222', edge_color=ec,
                width=1.8, node_size=25, edgecolors='#222222')
        c = all_counts[a]
        ax.set_title(
            f"alpha = {a}\n"
            f"BOND={c['BOND']}  LOCAL={c['LOCAL']}  "
            f"GLOBAL={c['GLOBAL']}  SILK={c['SILK']}",
            fontsize=10
        )

    legend_elements = [
        Line2D([0],[0], color='royalblue', lw=2, label='Bond link'),
        Line2D([0],[0], color='tomato',    lw=2, label='Local bridge'),
        Line2D([0],[0], color='seagreen',  lw=2, label='Global bridge'),
        Line2D([0],[0], color='orange',    lw=2, label='Silk link'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=4, fontsize=11, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(
        "HETA + R-B Linear Interpolation: alpha sensitivity\n"
        "Star Hybrid Network (3,4,3)  |  "
        "alpha=0: all GLOBAL   alpha=1: all LOCAL",
        fontsize=12
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig("./heta_rb_alpha.png", dpi=150, bbox_inches='tight')
    print("\n\n  對比圖已儲存：heta_rb_alpha.png")