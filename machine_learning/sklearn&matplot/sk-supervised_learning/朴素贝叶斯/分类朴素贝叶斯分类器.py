"""
它假设由索引描述的每个特征都有自己的分类分布。它假设由索引描述的每个特征都有自己的绝对分布。
分类朴素贝叶斯分类器适用于具有分类分布的离散特征的分类。每个特征的类别均来自分类分布。

"""

import numpy as np
rng = np.random.RandomState(1)
X = rng.randint(5, size=(6, 100))
y = np.array([1, 2, 3, 4, 5, 6])
from sklearn.naive_bayes import CategoricalNB
clf = CategoricalNB()
clf.fit(X, y)
print(clf.predict(X[2:3]))