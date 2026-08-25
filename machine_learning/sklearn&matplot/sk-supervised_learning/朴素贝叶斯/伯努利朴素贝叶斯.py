"""
处理的是0/1 特征，只关注 “有没有出现”，不关心次数
P(x_i | y) = P(i | y) * x_i + (1 - P(i | y)) * (1 - x_i)
x_i：二值特征（0 或 1，比如 “词是否出现”）
P(i | y)：在类别 y 下，特征 i 出现的概率
当 x_i = 1 时，式子简化为 P(x_i | y) = P(i | y)
当 x_i = 0 时，式子简化为 P(x_i | y) = 1 - P(i | y)
"""
import numpy as np
rng = np.random.RandomState(1)
X = rng.randint(5, size=(6, 100))
Y = np.array([1, 2, 3, 4, 4, 5])
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(X, Y)
print(clf.predict(X[2:3]))