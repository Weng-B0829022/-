import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_circles

# 1. 產生測試數據：兩個嵌套的圓環 (Non-spherical data)
# factor 控制內外圈距離，noise 增加一點隨機擾動
X, _ = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)

# 2. 執行 K-means 聚類
# 我們設定 k=2，預期它能分出內外圈，但結果會讓你失望
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# 3. 繪圖可視化
plt.figure(figsize=(8, 6))

# 畫出數據點，顏色代表 K-means 分配的標籤
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='Paired', alpha=0.6, edgecolors='w')

# 畫出聚類中心 (大的 紅色 X)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')

plt.title("K-means Clustering on Nested Circles")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()