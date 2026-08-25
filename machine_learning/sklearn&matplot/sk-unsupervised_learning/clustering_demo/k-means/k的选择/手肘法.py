from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 绘制不同 K 对应的 Inertia，找“手肘”点：
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_) # 所有样本点到其所属簇中心的距离平方和

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('手肘法选择 K')
plt.show()