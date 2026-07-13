from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
"""
Perceptron
输入层 (Input Layer)：接收外部数据，通常用向量 x = [x₁, x₂, ..., xₙ] 表示。
权重 (Weights)：每个输入都有一个对应的权重 w = [w₁, w₂, ..., wₙ]，表示该输入的重要性。
偏置 (Bias)：一个常数项 b，可以看作是“激活的门槛”。
激活函数 (Activation Function)：一个简单的函数，决定神经元是否“ firing ”（激活）。最经典的感知机使用阶跃函数（Step Function）。
"""


X, y = load_digits(return_X_y=True)
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)

clf.score(X, y)