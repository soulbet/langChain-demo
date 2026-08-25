from sklearn import preprocessing
import numpy as np

"""
逻辑回归、SVM、KNN（只要是基于距离或梯度的模型，首选它）
Xij = (Xij - μi) / σ
"""

X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
# StandardScaler 通过去除平均值和缩放到单位方差来标准化特征。
# 下划线 _ 结尾的属性，都代表它是通过 fit() 方法从数据中“学习”到的状态
scaler = preprocessing.StandardScaler().fit(X_train)
# 不会改变数据，只是通过fit()计算之后保存下来
print(f"平均数：{scaler.mean_}")
print(f"标准差：{scaler.scale_}")

# 真正的标准化，fit()计算均值和标准差，然后将X_train数据减去均值，除以标准差，变成标准正态分布
print(f"平均数：{scaler.transform(X_train)}")
