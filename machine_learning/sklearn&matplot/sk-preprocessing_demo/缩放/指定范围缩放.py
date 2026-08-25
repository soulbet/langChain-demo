import numpy as np
from sklearn import preprocessing


"""
针对某个范围的特征缩放，将数据缩放到[0-1]，对异常值敏感，
应用场景：图像像素处理、特定要求的神经网络
数学公式：
X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

X_scaled = X_std * (max - min) + min
"""
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
# feature_range 指定缩放范围,默认(0,1)
# feature_range(-1,1),公式：X_std * (new_max - new_min) + new_min
min_max_scaler=preprocessing.MinMaxScaler(feature_range=(0,1))
print(f"{min_max_scaler.scale_}")
print(f"{min_max_scaler.min_}")
print(f"{min_max_scaler.max_}")

X_train_minmax = min_max_scaler.fit_transform(X_train)
print(X_train_minmax)

# 应用到测试集
X_test = np.array([[-3., -1.,  4.]])
X_test_minmax = min_max_scaler.transform(X_test)
