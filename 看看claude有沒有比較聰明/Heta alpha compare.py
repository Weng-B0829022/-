"""
HETA Alpha Threshold Comparison
=================================
對多個 alpha 值執行 HETA，並以表格輸出結果。

T_E^k = Mean + alpha × SD
  alpha > 0 → 嚴格門檻（較少 bond）
  alpha = 0 → 只用平均值當門檻
  alpha < 0 → 寬鬆門檻（較多 bond）

使用方式：
  python3 heta_alpha_compare.py

調整下方 ALPHA_LIST 即可測試不同組合，例如：
  ALPHA_LIST = [2, 1, 0, -1]
"""

import networkx as nx
import numpy as np
import random

# ── 設定 ────────────────────────────────────────────────
CIRCLES_FILE = '0_circles.txt'
EDGES_FILE   = '0_edges.txt'
N_RANDOM     = 100
SEED         = 42

ALPHA_LIST = [2, 1, 0, -1]   # ← 在這裡修改想測試的 alpha 值

SELECTED_CIRCLES = ['circle4', 'circle6', 'circle11', 'circle15', 'circle16', 'circle19']


# ════════════════════════════════════════════════════════
# HETA 核心函式
# ════════════════════════════════════════════════════════

def get_kth_layer_neighbors(G, node, exclude, k):
    if k == 1:
        return set(G.neighbors(node)) - {exclude}
    visited  = {node, exclude}
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


def compute_common_neighbor_ratio(G, u, v, k):
    if k == 1:
        V1_u = set(G.neighbors(u)) - {v}
        V1_v = set(G.neighbors(v)) - {u}
        if not V1_u or not V1_v:
            return 0.0
        return len(V1_u & V1_v) / min(len(V1_u), len(V1_v))
    Vk_u  = get_kth_layer_neighbors(G, u, v, k)
    Vk_v  = get_kth_layer_neighbors(G, v, u, k)
    Vk1_u = get_kth_layer_neighbors(G, u, v, k - 1)
    Vk1_v = get_kth_layer_neighbors(G, v, u, k - 1)
    union = (Vk_u & Vk_v) | (Vk1_u & Vk_v) | (Vk_u & Vk1_v)
    if not union:
        return 0.0
    num = len(Vk_u & Vk_v) + len(Vk1_u & Vk_v) + len(Vk_u & Vk1_v)
    den = (min(len(Vk_u), len(Vk_v)) +
           min(len(Vk1_u), len(Vk_v)) +
           min(len(Vk_u), len(Vk1_v)))
    return num / den if den > 0 else 0.0


def compute_kmax(G):
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    try:
        avg_spl = nx.average_shortest_path_length(G)
    except Exception:
        return 1
    return max(1, int(avg_spl / 2))


def switching_randomize(G, Q=100):
    Gr    = G.copy()
    edges = list(Gr.edges())
    m     = len(edges)
    for _ in range(Q * m):
        if len(edges) < 2:
            break
        i, j = random.sample(range(len(edges)), 2)
        a, b = edges[i]
        c, d = edges[j]
        if a == d or c == b:
            continue
        if Gr.has_edge(a, d) or Gr.has_edge(c, b):
            continue
        Gr.remove_edge(a, b)
        Gr.remove_edge(c, d)
        Gr.add_edge(a, d)
        Gr.add_edge(c, b)
        edges[i] = (a, d)
        edges[j] = (c, b)
    return Gr


def compute_external_threshold(G, k, n_random=100, alpha=2.0):
    """
    T_E^k = Mean + alpha × SD
    alpha=2 為原版，alpha=1 為降低門檻版。
    """
    rand_ratios = []
    for _ in range(n_random):
        Gr = switching_randomize(G)
        for (u, v) in Gr.edges():
            rand_ratios.append(compute_common_neighbor_ratio(Gr, u, v, k))
    if not rand_ratios:
        return 0.0
    return float(np.mean(rand_ratios) + alpha * np.std(rand_ratios))


def compute_internal_threshold(candidate_ratios):
    if not candidate_ratios:
        return 0.0
    arr = np.array(candidate_ratios)
    return float(np.mean(arr) - np.std(arr))


def heta(G, n_random=100, alpha=2.0):
    """
    執行 HETA，回傳每條邊的類型。

    參數
    ----
    G        : NetworkX 無向圖
    n_random : 隨機化網路數量
    alpha    : T_E^k = Mean + alpha × SD（原版 alpha=2，降低門檻用 alpha=1）

    回傳
    ----
    dict : { (u, v): type_string }
      type_string ∈ {'silk', 'bond', 'local_bridge_k{k}', 'global_bridge'}
    """
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    kmax  = compute_kmax(G)
    edges = list(G.edges())

    link_types = {e: None for e in edges}
    pass_flag  = {e: True  for e in edges}

    # silk links
    for (u, v) in edges:
        if G.degree(u) == 1 or G.degree(v) == 1:
            link_types[(u, v)] = 'silk'
            pass_flag[(u, v)]  = False

    # 計算外部閾值
    ext_thresholds = {
        k: compute_external_threshold(G, k, n_random, alpha)
        for k in range(1, kmax + 1)
    }
    print(f"  [alpha={alpha}] kmax={kmax}, T_E^1={ext_thresholds.get(1, 0):.4f}")

    # 逐層分類
    for k in range(1, kmax + 1):
        T_E    = ext_thresholds[k]
        cands  = []
        ratios = []
        for (u, v) in edges:
            if not pass_flag[(u, v)]:
                continue
            r = compute_common_neighbor_ratio(G, u, v, k)
            if r >= T_E:
                link_types[(u, v)] = 'bond'
                pass_flag[(u, v)]  = False
            else:
                cands.append((u, v))
                ratios.append(r)
        if cands:
            T_I = compute_internal_threshold(ratios)
            for (u, v), r in zip(cands, ratios):
                if r >= T_I:
                    link_types[(u, v)] = f'local_bridge_k{k}'
                    pass_flag[(u, v)]  = False

    # 剩餘 → global bridge
    for (u, v) in edges:
        if pass_flag[(u, v)]:
            link_types[(u, v)] = 'global_bridge'

    return link_types


# ════════════════════════════════════════════════════════
# 輔助函式
# ════════════════════════════════════════════════════════

def gt_color(node_group, u, v):
    """Ground truth：同群=紅，跨群=綠"""
    return 'red' if node_group[u] == node_group[v] else 'green'


def heta_color(link_types, u, v):
    """HETA 分類：bond=紅，其餘=綠"""
    t = link_types.get((u, v)) or link_types.get((v, u)) or 'other'
    return 'red' if t == 'bond' else 'green'


def compute_stats(G, node_group, link_types):
    """
    計算單一 alpha 結果的統計數字。
    回傳 dict：red, green, brown, purple, diff, acc
    """
    total  = G.number_of_edges()
    red    = 0
    brown  = 0   # GT=紅，HETA=綠（漏判 bond）
    purple = 0   # GT=綠，HETA=紅（誤判 bond）

    for u, v in G.edges():
        gc = gt_color(node_group, u, v)
        hc = heta_color(link_types, u, v)
        if hc == 'red':
            red += 1
        if gc == 'red' and hc == 'green':
            brown += 1
        elif gc == 'green' and hc == 'red':
            purple += 1

    diff = brown + purple
    return {
        'red':    red,
        'green':  total - red,
        'brown':  brown,
        'purple': purple,
        'diff':   diff,
        'acc':    (total - diff) / total * 100,
    }


def print_table(total, gt_r, results):
    """
    印出單一表格，results 為 list of (alpha, stats_dict)。
    """
    W = 72
    col = f"{'方法':<26} {'紅色':>6} {'綠色':>6} {'咖啡色':>7} {'紫色':>6} {'差異':>6} {'準確率':>8}"
    print("=" * W)
    print(col)
    print("-" * W)
    print(f"{'Ground Truth':<26} {gt_r:>6} {total-gt_r:>6} {'—':>7} {'—':>6} {'—':>6} {'100.0%':>8}")
    for alpha, s in results:
        label = f"HETA  alpha={alpha}  (Mean{alpha:+g}SD)"
        print(f"{label:<26} {s['red']:>6} {s['green']:>6} "
              f"{s['brown']:>7} {s['purple']:>6} {s['diff']:>6} {s['acc']:>7.1f}%")
    print("=" * W)
    print("  咖啡色 = GT紅、HETA綠（漏判 bond）")
    print("  紫色   = GT綠、HETA紅（誤判 bond）")
    print()


# ════════════════════════════════════════════════════════
# 主程式
# ════════════════════════════════════════════════════════

if __name__ == '__main__':
    random.seed(SEED)
    np.random.seed(SEED)

    # ── 讀資料 ────────────────────────────────────────
    circles_raw = {}
    with open(CIRCLES_FILE) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                circles_raw[parts[0]] = list(map(int, parts[1:]))

    all_edges = []
    with open(EDGES_FILE) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                all_edges.append((int(parts[0]), int(parts[1])))

    # 建立節點群體對應
    node_group = {}
    for cid in SELECTED_CIRCLES:
        for n in circles_raw.get(cid, []):
            if n not in node_group:
                node_group[n] = cid

    # 建圖
    all_nodes = set(node_group.keys())
    G = nx.Graph()
    G.add_nodes_from(all_nodes)
    for a, b in all_edges:
        if a in all_nodes and b in all_nodes:
            G.add_edge(a, b)
    G.remove_nodes_from([n for n in G.nodes if G.degree(n) == 0])

    total = G.number_of_edges()
    gt_r  = sum(1 for u, v in G.edges() if gt_color(node_group, u, v) == 'red')
    print(f"圖：{G.number_of_nodes()} 節點, {total} 邊")
    print(f"Ground Truth — 紅色: {gt_r}, 綠色: {total - gt_r}\n")
    print(f"執行 HETA，alpha 清單：{ALPHA_LIST}\n")

    # ── 對每個 alpha 執行 HETA 並收集結果 ────────────
    results = []
    for alpha in ALPHA_LIST:
        print(f"[alpha={alpha}] 執行中...")
        types = heta(G, n_random=N_RANDOM, alpha=float(alpha))
        s     = compute_stats(G, node_group, types)
        results.append((alpha, s))
        print(f"  → red={s['red']}, green={s['green']}, "
              f"brown={s['brown']}, purple={s['purple']}, acc={s['acc']:.1f}%\n")

    # ── 輸出表格（每個 alpha 一張，加上總覽）────────
    print("\n" + "━" * 72)
    print("  各 alpha 獨立表格")
    print("━" * 72 + "\n")

    for alpha, s in results:
        label = f"HETA  alpha={alpha}  (Mean {'+' if alpha >= 0 else ''}{alpha}×SD)"
        print(f"  {label}")
        W = 60
        col = f"  {'類別':<20} {'數量':>8} {'佔總邊數':>10}"
        print("  " + "=" * W)
        print(col)
        print("  " + "-" * W)
        print(f"  {'Ground Truth 紅色':<20} {gt_r:>8} {gt_r/total*100:>9.1f}%")
        print(f"  {'Ground Truth 綠色':<20} {total-gt_r:>8} {(total-gt_r)/total*100:>9.1f}%")
        print("  " + "-" * W)
        print(f"  {'HETA 紅色 (bond)':<20} {s['red']:>8} {s['red']/total*100:>9.1f}%")
        print(f"  {'HETA 綠色 (others)':<20} {s['green']:>8} {s['green']/total*100:>9.1f}%")
        print("  " + "-" * W)
        print(f"  {'咖啡色 (漏判bond)':<20} {s['brown']:>8} {s['brown']/total*100:>9.1f}%")
        print(f"  {'紫色 (誤判bond)':<20} {s['purple']:>8} {s['purple']/total*100:>9.1f}%")
        print(f"  {'差異合計':<20} {s['diff']:>8} {s['diff']/total*100:>9.1f}%")
        print(f"  {'準確率':<20} {'':>8} {s['acc']:>9.1f}%")
        print("  " + "=" * W + "\n")

    # ── 總覽比較表 ────────────────────────────────────
    print("━" * 72)
    print("  總覽比較表")
    print("━" * 72 + "\n")
    print_table(total, gt_r, results)