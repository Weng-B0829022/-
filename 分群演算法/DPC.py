import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_blobs
from scipy.spatial.distance import pdist, squareform

# ==========================================
# 1. 產生模擬資料：複雜的「笑臉形」
# ==========================================
n_samples = 600
# 產生嘴巴 (半圓弧線)
t = np.linspace(-np.pi/1.2, -np.pi/6, int(n_samples/2))
mouth_x = 3.5 * np.cos(t)
mouth_y = 3.5 * np.sin(t) - 1.5
mouth_data = np.column_stack([mouth_x, mouth_y]) + np.random.normal(scale=0.15, size=(len(mouth_x), 2))

# 產生眼睛 (兩個高密度圓點)
eyes_data, _ = make_blobs(n_samples=int(n_samples/3), centers=[[-1.5, 2], [1.5, 2]], cluster_std=0.4, random_state=42)

# 產生一些背景雜訊
noise_data = np.random.uniform(low=-5, high=5, size=(int(n_samples/6), 2))

# 合併資料
data = np.vstack([mouth_data, eyes_data, noise_data])
n_points = len(data)

# 計算兩兩之間的距離矩陣
dist_matrix = squareform(pdist(data))

# ==========================================
# 2. DPC 核心演算法
# ==========================================
# 截斷距離 dc (參數)，這對幾何形狀影響極大
# 這裡我們設得比較小 (1% 分位數)，試圖抓到精細結構
dc = np.percentile(dist_matrix, 1.0) 

# 計算局部密度 Rho (ρ) - 使用 Gaussian Kernel
rho = np.sum(np.exp(-(dist_matrix / dc)**2), axis=1)

# 計算相對距離 Delta (δ) & 記錄最近的高密度鄰居 (n_id)
delta = np.zeros_like(rho)
n_id = np.zeros(n_points, dtype=int)
sorted_idx = np.argsort(-rho) # 依照密度由大到小排序

for i in range(1, n_points):
    idx = sorted_idx[i]
    # 在所有密度比點 idx 高的點中，找出距離最近的
    higher_density_idx = sorted_idx[:i]
    distances = dist_matrix[idx, higher_density_idx]
    delta[idx] = np.min(distances)
    n_id[idx] = higher_density_idx[np.argmin(distances)]

# 密度最大的點，其 Delta 設定為所有距離中的最大值
delta[sorted_idx[0]] = np.max(delta)

# ==========================================
# 3. 視覺化展示
# ==========================================
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# 圖一：原始資料
ax[0].scatter(data[:, 0], data[:, 1], c='gray', alpha=0.5, s=15)
ax[0].set_title('Original Data (Smile Face with Noise)')
ax[0].axis('equal')

# 圖二：決策圖 (Decision Graph)
gamma = rho * delta # 綜合指標
ax[1].scatter(rho, delta, c='black', alpha=0.6, s=20)
ax[1].set_xlabel(r'Local Density ($\rho$)')
ax[1].set_ylabel(r'Min. Distance to Higher Density ($\delta$)')
ax[1].set_title('DPC Decision Graph')

# 自動挑選中心 (簡化：選取 Gamma 最大的前 3 個)
n_clusters = 3
center_ids = np.argsort(-gamma)[:n_clusters]
ax[1].scatter(rho[center_ids], delta[center_ids], c='red', s=100, marker='X', label='Identified Peaks')
ax[1].legend()

# 圖三：DPC 聚類結果
labels = -np.ones(n_points, dtype=int)
for i, c_id in enumerate(center_ids):
    labels[c_id] = i

# 瀑布分配 (從高密度往低密度傳遞標籤)
for i in range(n_points):
    idx = sorted_idx[i]
    if labels[idx] == -1:
        labels[idx] = labels[n_id[idx]] # 跟隨最近的高密度鄰居的標籤

# 繪製分配結果，並標記中心
cmap = plt.cm.get_cmap('Set1', n_clusters)
ax[2].scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap, s=25, edgecolors='k', alpha=0.7)
ax[2].scatter(data[center_ids, 0], data[center_ids, 1], c='red', marker='X', s=200, label='Centers')
ax[2].set_title(f'DPC Clustering (n={n_clusters})')
ax[2].axis('equal')

plt.tight_layout()
plt.show()