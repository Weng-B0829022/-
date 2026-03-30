"""
HETA - Hierarchical Edge Type Analysis
=======================================
根據論文：Huang et al. (2019), Physica A 536, 121027
"Beyond bond links in complex networks: Local bridges, global bridges and silk links"

四種邊類型：
  - silk link      : 端點之一的 degree = 1
  - bond link      : 共同鄰居比例超過外部閾值 T_E^k
  - local bridge   : 共同鄰居比例介於 T_I^k 與 T_E^k 之間
  - global bridge  : 所有層都無法分類的邊
"""

import networkx as nx
import numpy as np
import random


# ════════════════════════════════════════════════════════
# 1. 共同鄰居計算
# ════════════════════════════════════════════════════════

def get_kth_layer_neighbors(G, node, exclude, k):
    """
    回傳節點 node 的第 k 層鄰居集合（排除 exclude 節點及前幾層已訪問的節點）。

    參數
    ----
    G       : NetworkX 無向圖
    node    : 起始節點
    exclude : 邊的另一端點（排除在外）
    k       : 層數（從 1 開始）

    回傳
    ----
    set : 第 k 層鄰居節點集合
    """
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


def compute_common_neighbor_ratio(G, u, v, k):
    """
    計算邊 (u, v) 的第 k 層共同鄰居比例 R^k_{u,v}。

    - k=1 使用論文公式 (1)
    - k>1 使用論文公式 (2)，同時考慮 k 層與 k-1 層的交集

    回傳值範圍 [0, 1]；0 表示無共同鄰居。
    """
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


# ════════════════════════════════════════════════════════
# 2. kmax 計算
# ════════════════════════════════════════════════════════

def compute_kmax(G):
    """
    kmax = floor( 平均最短路徑長度 / 2 )，最小值為 1。

    對應論文公式 (4)。
    若網路不連通，取最大連通子圖計算。
    """
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    try:
        avg_spl = nx.average_shortest_path_length(G)
    except Exception:
        return 1
    return max(1, int(avg_spl / 2))


# ════════════════════════════════════════════════════════
# 3. 隨機化網路生成（switching algorithm）
# ════════════════════════════════════════════════════════

def switching_randomize(G, Q=100):
    """
    使用 switching randomization 生成一個隨機化網路。
    保留原網路的節點數、邊數與節點度分布。

    參數
    ----
    G : 原始網路（NetworkX 無向圖）
    Q : 每條邊執行的 switching 次數（論文建議 Q=100）

    回傳
    ----
    NetworkX 無向圖（隨機化後）
    """
    Gr = G.copy()
    edges = list(Gr.edges())
    m = len(edges)

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


# ════════════════════════════════════════════════════════
# 4. 閾值計算
# ════════════════════════════════════════════════════════

def compute_external_threshold(G, k, n_random=1000):
    """
    計算第 k 層的外部閾值 T_E^k。

    對應論文公式 (5)：
        T_E^k = Mean_E^k(RG) + 2 * SD_E^k(RG)

    方法：生成 n_random 個隨機化網路，收集所有邊的 R^k 值，
    取平均加兩倍標準差作為上閾值。

    超過 T_E^k 的邊被判定為 bond link。
    """
    rand_ratios = []
    for _ in range(n_random):
        Gr = switching_randomize(G)
        for (u, v) in Gr.edges():
            rand_ratios.append(compute_common_neighbor_ratio(Gr, u, v, k))
    if not rand_ratios:
        return 0.0
    return float(np.mean(rand_ratios) + 2 * np.std(rand_ratios))


def compute_internal_threshold(candidate_ratios):
    """
    計算第 k 層的內部閾值 T_I^k。

    對應論文公式 (6)：
        T_I^k = Mean_I^k(candidates) - SD_I^k(candidates)

    超過 T_I^k 的候選邊被判定為 kth-layer local bridge。
    """
    if not candidate_ratios:
        return 0.0
    arr = np.array(candidate_ratios)
    return float(np.mean(arr) - np.std(arr))


# ════════════════════════════════════════════════════════
# 5. 主演算法：HETA
# ════════════════════════════════════════════════════════

def heta(G, n_random=1000):
    """
    對網路 G 執行 HETA，回傳每條邊的類型。

    參數
    ----
    G        : NetworkX 無向圖（不含自迴圈與平行邊）
    n_random : 生成隨機化網路的數量（論文使用 1000）

    回傳
    ----
    dict : { (u, v): type_string }

    type_string 可能值：
        'silk'              - silk link
        'bond'              - bond link
        'local_bridge_k{k}' - kth-layer local bridge
        'global_bridge'     - global bridge
    """
    # 確保使用最大連通子圖
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    kmax  = compute_kmax(G)
    edges = list(G.edges())

    link_types = {e: None  for e in edges}
    pass_flag  = {e: True  for e in edges}   # True = 尚未分類

    # ── Step 3.1：識別 silk links ──────────────────────────
    for (u, v) in edges:
        if G.degree(u) == 1 or G.degree(v) == 1:
            link_types[(u, v)] = 'silk'
            pass_flag[(u, v)]  = False

    # ── Step 2：計算各層外部閾值 ───────────────────────────
    ext_thresholds = {}
    for k in range(1, kmax + 1):
        ext_thresholds[k] = compute_external_threshold(G, k, n_random)

    # ── Step 3.2：逐層識別 bond links 與 local bridges ─────
    for k in range(1, kmax + 1):
        T_E = ext_thresholds[k]
        candidate_edges  = []
        candidate_ratios = []

        for (u, v) in edges:
            if not pass_flag[(u, v)]:
                continue
            r = compute_common_neighbor_ratio(G, u, v, k)
            if r >= T_E:
                # bond link：共同鄰居比例超過外部閾值
                link_types[(u, v)] = 'bond'
                pass_flag[(u, v)]  = False
            else:
                candidate_edges.append((u, v))
                candidate_ratios.append(r)

        if candidate_edges:
            T_I = compute_internal_threshold(candidate_ratios)
            for (u, v), r in zip(candidate_edges, candidate_ratios):
                if r > T_I:
                    # kth-layer local bridge
                    link_types[(u, v)] = f'local_bridge_k{k}'
                    pass_flag[(u, v)]  = False
                # 否則繼續往下一層

    # ── Step 3.3：剩餘未分類 = global bridges ──────────────
    for (u, v) in edges:
        if pass_flag[(u, v)]:
            link_types[(u, v)] = 'global_bridge'

    return link_types


# ════════════════════════════════════════════════════════
# 6. 輔助：統計各類型比例
# ════════════════════════════════════════════════════════

def summarize(link_types):
    """
    統計四種類型的數量與比例。

    回傳
    ----
    dict : {
        'bond':   {'count': int, 'pct': float},
        'local':  {'count': int, 'pct': float},
        'global': {'count': int, 'pct': float},
        'silk':   {'count': int, 'pct': float},
    }
    """
    counts = {'bond': 0, 'local': 0, 'global': 0, 'silk': 0}
    for t in link_types.values():
        if   t == 'bond':                counts['bond']   += 1
        elif t and t.startswith('local'): counts['local']  += 1
        elif t == 'global_bridge':        counts['global'] += 1
        elif t == 'silk':                 counts['silk']   += 1
    total = sum(counts.values())
    return {
        k: {'count': v, 'pct': round(v / total * 100, 1) if total else 0}
        for k, v in counts.items()
    }


# ════════════════════════════════════════════════════════
# 7. 快速測試（直接執行此檔案時觸發）
# ════════════════════════════════════════════════════════

if __name__ == '__main__':
    import random as _random
    _random.seed(42)

    # 用 Karate Club 網路做測試（論文 Table 1 有此網路）
    G = nx.karate_club_graph()
    print(f"網路：Karate Club")
    print(f"節點數：{G.number_of_nodes()}，邊數：{G.number_of_edges()}")
    print(f"kmax：{compute_kmax(G)}")
    print("執行 HETA（n_random=100，僅供快速測試）...")

    types = heta(G, n_random=100)
    stats = summarize(types)

    print("\n── 分類結果 ──")
    for k, v in stats.items():
        print(f"  {k:>12} : {v['count']:>3} 條  ({v['pct']}%)")