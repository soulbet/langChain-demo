
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=200, centers=3)
gmm = GaussianMixture(n_components=3)
gmm.fit(X)

# 聚类类别
labels = gmm.predict(X)
print(labels)
# 归属概率
proba = gmm.predict_proba(X)
print(proba)