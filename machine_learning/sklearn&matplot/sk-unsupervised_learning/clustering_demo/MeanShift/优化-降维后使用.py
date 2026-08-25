from sklearn.decomposition import PCA

# 先降维到 10-20 维
pca = PCA(n_components=15)
X_pca = pca.fit_transform(X)

# 再运行 Mean Shift
ms = MeanShift(bandwidth=estimate_bandwidth(X_pca))
y_pred = ms.fit_predict(X_pca)