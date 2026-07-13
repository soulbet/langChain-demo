
"""
ElasticNet是一个训练时同时用ℓ1和ℓ2范数进行正则化的线性回归模型。

"""


from sklearn.linear_model import ElasticNetCV
from sklearn.datasets import make_regression
X, y = make_regression(n_features=2, random_state=0)
regr = ElasticNetCV(cv=5, random_state=0)
regr.fit(X, y)
ElasticNetCV(cv=5, random_state=0)
print(regr.alpha_)

print(regr.intercept_)

print(regr.predict([[0, 0]]))