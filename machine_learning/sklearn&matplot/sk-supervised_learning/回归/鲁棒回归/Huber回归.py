
"""
Robustness regression
对异常值具有鲁棒性的线性回归模型。
鲁棒拟合(robust fitting)的一个重要概念是崩溃点(breakdown point):即拟合模型（仍准确预测）所能承受的离群值的最大比例。
"""
import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.datasets import make_regression
rng = np.random.RandomState(0)
X, y, coef = make_regression(
    n_samples=200, n_features=2, noise=4.0, coef=True, random_state=0)
X[:4] = rng.uniform(10, 20, (4, 2))
y[:4] = rng.uniform(10, 20, 4)
huber = HuberRegressor().fit(X, y)
huber.score(X, y)
huber.predict(X[:1,])

linear = LinearRegression().fit(X, y)
print("True coefficients:", coef)

print("Huber coefficients:", huber.coef_)

print("Linear Regression coefficients:", linear.coef_)
