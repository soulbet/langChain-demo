"""
CNB是标准多项式朴素贝叶斯(MNB)算法的一种自适应算法，特别适用于不平衡的数据集
算 「不属于其他所有类的概率」
用 除了 y 以外所有类的总和（补集） 计算
"""

import numpy as np
rng = np.random.RandomState(1)
X = rng.randint(5, size=(6, 100))
y = np.array([1, 2, 3, 4, 5, 6])
from sklearn.naive_bayes import ComplementNB
clf = ComplementNB()
clf.fit(X, y)

print(clf.predict(X[2:3]))