import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# 1. 資料準備
X, _ = make_moons(n_samples=200, noise=0.5, random_state=47)
X = StandardScaler().fit_transform(X)

def plot_dbscan_steps(X, eps=0.3, min_samples=5):
    n_points = X.shape[0]
    labels = -np.ones(n_points)  # -1 表示未分類
    neighbors_model = NearestNeighbors(radius=eps).fit(X)
    neighborhoods = neighbors_model.radius_neighbors(X, return_distance=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    titles = ["Step 1: Initial (Unclassified)", "Step 2: First Cluster Expanding", 
              "Step 3: Mid-Process (Multi-Clusters)", "Step 4: Final Convergence"]
    
    # 記錄要在哪幾個步數截圖
    snapshot_steps = [0, 20, 100, n_points]
    current_snapshot = 0
    cluster_id = 0
    
    processed_count = 0
    for i in range(n_points):
        if labels[i] != -1: continue
        
        # 檢查是否為核心點
        if len(neighborhoods[i]) >= min_samples:
            labels[i] = cluster_id
            stack = list(neighborhoods[i])
            
            while stack:
                neighbor_idx = stack.pop()
                if labels[neighbor_idx] == -1:
                    labels[neighbor_idx] = cluster_id
                    if len(neighborhoods[neighbor_idx]) >= min_samples:
                        stack.extend(neighborhoods[neighbor_idx])
                processed_count += 1
                
                # 檢查是否達到截圖點 (根據處理進度)
                if current_snapshot < 3 and processed_count >= snapshot_steps[current_snapshot + 1]:
                    ax = axes[current_snapshot]
                    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, edgecolors='k', alpha=0.7)
                    ax.set_title(titles[current_snapshot])
                    current_snapshot += 1
            cluster_id += 1
            
    # 最後一張圖：完整結果
    axes[3].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, edgecolors='k', alpha=0.7)
    axes[3].set_title(titles[3])
    
    plt.tight_layout()
    plt.show()

plot_dbscan_steps(X)