"""
1. 适用场景
离散计数型数据
文本分类（词频、单词个数）
统计次数、频次类特征
2. 核心思想
假设特征服从多项式分布，特征是出现次数 / 数量，不是 0-1，Xi必须非负整数
引入拉普拉斯平滑（避免概率为 0）
P(xi|y)=(Ny,i+α)/Ny+αV
常用：α=1拉普拉斯平滑

多项式分布含义
一次试验，分成V 个类别，统计每个类别出现多少次，就是多项式分布。
P(X1=n1,X2=n2...)=(n!/n1!*n2!)*P1^n1*P2^n2
n!/n1!*n2! 总共有n!种排列，n1!*n2!去除重复排列，比如n1=3,X1类别出现3次，
"""

import numpy as np
rng = np.random.RandomState(1)
X = rng.randint(5, size=(6, 100))
y = np.array([1, 2, 3, 4, 5, 6])
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X, y)
print(clf.predict(X[2:3]))