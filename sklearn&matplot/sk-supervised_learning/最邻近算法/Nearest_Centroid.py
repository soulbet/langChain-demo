
"""

每个类别均以其质心表示，测试样本被分类为具有最接近质心的类别。
"""

from sklearn.neighbors import NearestCentroid
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
clf = NearestCentroid()
clf.fit(X, y)

print(clf.predict([[-0.8, -1]]))