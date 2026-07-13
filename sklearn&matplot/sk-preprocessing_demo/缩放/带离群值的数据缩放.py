import numpy as np
from sklearn.preprocessing import RobustScaler

"""
公式：
(x-中位数)/四分位距
四分位距 (IQR)： 第75百分位数（Q3）与第25百分位数（Q1）的差值
"""
data = np.array([20, 22, 25, 24, 23, 1000]).reshape(2, 3)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(data)
print(X_scaled)

# 处理离群值
import numpy as np
from sklearn.preprocessing import StandardScaler

# 假设 X 是一列数据
# 方法：把小于 1% 分位数的设为 1% 分位数，大于 99% 分位数的设为 99% 分位数
q01 = np.quantile(data, 0.01)
q99 = np.quantile(data, 0.99)
X_capped = np.clip(data, q01, q99)  # 把超出范围的数据截断

# 截尾之后，数据安全了，再放心使用 StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_capped)


