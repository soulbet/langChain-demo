
from sklearn.cluster import AgglomerativeClustering

# Ward 法（默认）
model_ward = AgglomerativeClustering(linkage='ward', n_clusters=3)

# 单链接
model_single = AgglomerativeClustering(linkage='single', metric='euclidean', n_clusters=3)

# 全链接
model_complete = AgglomerativeClustering(linkage='complete', metric='euclidean', n_clusters=3)

# 平均链接
model_average = AgglomerativeClustering(linkage='average', metric='euclidean', n_clusters=3)

# 中心链接
from sklearn.neighbors import DistanceMetric
# 注：sklearn 不直接支持 centroid，需使用 scipy.cluster.hierarchy.linkage(method='centroid')