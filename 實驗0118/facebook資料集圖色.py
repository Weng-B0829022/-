import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

# --- 1. 區分強弱連結的演算法函式 ---

def classify_links(G, threshold=0.1):
    """
    基於 Neighborhood Overlap (Jaccard 相似度) 區分邊為強連結或弱連結。
    threshold: 相似度門檻，高於此值為強連結。
    """
    strong_edges = []
    weak_edges = []
    
    for u, v in G.edges():
        # 獲取 u 和 v 的鄰居集合
        neighbors_u = set(G.neighbors(u))
        neighbors_v = set(G.neighbors(v))
        
        # 計算交集與聯集
        intersection = len(neighbors_u & neighbors_v)
        union = len(neighbors_u | neighbors_v) - 2 # 扣除 u, v 本身
        
        # 計算重疊度 (Overlap)
        overlap = intersection / union if union > 0 else 0
        
        if overlap >= threshold:
            strong_edges.append((u, v))
        else:
            weak_edges.append((u, v))
            
    return strong_edges, weak_edges

# --- 2. 繪製區分強弱連結的圖片 ---

def visualize_strong_weak_ties(node_id, data_path="實驗0118/facebook", overlap_threshold=0.1):
    edge_file = f"{data_path}/{node_id}.edges"
    if not os.path.exists(edge_file):
        print(f"錯誤：找不到檔案 {edge_file}")
        return

    # 讀取圖形
    G = nx.read_edgelist(edge_file, nodetype=int)
    
    # 使用剛才定義的函式區分邊
    strong_edges, weak_edges = classify_links(G, threshold=overlap_threshold)
    
    # 偵測社群 (為了讓節點上色，看起來更清楚)
    communities = nx.community.louvain_communities(G)
    node_to_community = {node: i for i, comm in enumerate(communities) for node in comm}
    community_ids = [node_to_community[node] for node in G.nodes()]

    # 設定布局
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.15, seed=42)
    
    # 1. 繪製弱連結 (Weak Ties / Bridges) - 使用灰色、虛線或較細的線
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=weak_edges, 
        width=1.0, 
        alpha=0.4, 
        edge_color='gray', 
        style='dashed',
        label='Weak Ties (Bridges)'
    )
    
    # 2. 繪製強連結 (Strong Ties) - 使用黑色或較粗的線
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=strong_edges, 
        width=2.5, 
        alpha=0.8, 
        edge_color='black', 
        label='Strong Ties (Intra-community)'
    )
    
    # 3. 繪製節點
    cmap = cm.get_cmap('tab20', max(community_ids) + 1)
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=60, 
        node_color=community_ids, 
        cmap=cmap,
        edgecolors='white',
        linewidths=0.5
    )

    plt.title(f"Facebook Ego Network {node_id}: Strong vs Weak Ties", fontsize=15)
    plt.legend(scatterpoints=1)
    plt.axis('off')
    
    print(f"節點總數: {G.number_of_nodes()}")
    print(f"強連結數量: {len(strong_edges)}")
    print(f"弱連結數量: {len(weak_edges)}")
    plt.show()

def visualize_facebook_communities(node_id, data_path="實驗0118/facebook"):
    # 1. 檢查檔案路徑並讀取 Facebook 邊界資料
    edge_file = f"{data_path}/{node_id}.edges"
    if not os.path.exists(edge_file):
        print(f"錯誤：找不到檔案 {edge_file}，請確認路徑是否正確。")
        return

    # 讀取圖形
    G = nx.read_edgelist(edge_file, nodetype=int)
    print(f"成功讀取節點 {node_id}：共有 {G.number_of_nodes()} 個節點, {G.number_of_edges()} 條邊。")

    # 2. 使用 Louvain 算法偵測社群 (找出節點分組)
    # 註：此算法會回傳一個字典 {節點: 社群ID}
    communities = nx.community.louvain_communities(G)
    
    # 建立一個顏色映射字典
    node_to_community = {}
    for i, community in enumerate(communities):
        for node in community:
            node_to_community[node] = i

    # 3. 設定視覺化參數
    plt.figure(figsize=(12, 10))
    
    # 使用 spring_layout 讓社群自然聚集
    pos = nx.spring_layout(G, k=0.15, seed=42)
    
    # 根據社群 ID 獲取顏色
    community_ids = [node_to_community[node] for node in G.nodes()]
    cmap = cm.get_cmap('viridis', max(community_ids) + 1)

    # 4. 繪製圖形
    # 繪製邊 (設定透明度避免干擾視覺)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    
    # 繪製節點，顏色由社群決定
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_size=50, 
        node_color=community_ids, 
        cmap=cmap,
        edgecolors='white',
        linewidths=0.5
    )

    plt.title(f"Facebook Ego Network: Node {node_id} (Community Visualization)", fontsize=15)
    plt.axis('off') # 隱藏座標軸
    
    print(f"偵測到 {len(communities)} 個社群。正在顯示圖像...")
    plt.show()

# 執行
if __name__ == "__main__":
    #判斷社群
    visualize_facebook_communities(0)

    # 調整 overlap_threshold 可以改變強弱判定標準
    #visualize_strong_weak_ties(0, overlap_threshold=0.15)