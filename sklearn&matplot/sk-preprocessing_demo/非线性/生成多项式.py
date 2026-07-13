import numpy as np
from sklearn.preprocessing import PolynomialFeatures

"""
变换后的特征：
 [[ 2.  3.  4.  6.  9.]
 [ 4.  5. 16. 20. 25.]]
对应的特征名字： ['x0' 'x1' 'x0^2' 'x0 x1' 'x1^2']

比如X0X1 是两个特征交互作用   模型就能理解“长且窄”和“短且宽”是不同的形状。
'x0^2' 'x1^2' 会导致特征之间量纲被无限拉大，所以需要进行标准化
最佳搭档 = 线性回归 / 逻辑回归 + PolynomialFeatures(degree=2) + StandardScaler + Lasso/Ridge正则化
"""

# 假设有两套房子的长和宽
X = np.array([[2, 3],   # 房子A：长2，宽3
              [4, 5]])  # 房子B：长4，宽5

# 设置 degree=2（二次），include_bias=False（不要那个全1的截距列，因为模型会自动加）

poly = PolynomialFeatures(degree=2, include_bias=False)

X_poly = poly.fit_transform(X)
print("变换后的特征：\n", X_poly)
print("对应的特征名字：", poly.get_feature_names_out())
