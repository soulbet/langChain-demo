
"""
把单输出回归模型封装，一次预测多个连续目标值，一个样本输出多个回归结果。

"""

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# 构造数据：单输入特征，双输出目标
X = np.array([[1,2], [3,4], [5,6], [7,8], [9,10]])
# 同时预测两个连续值
y = np.array([[10,20], [30,40], [50,60], [70,80], [90,100]])

# 基础单输出回归模型
base_reg = GradientBoostingRegressor(random_state=42)
# 封装为多输出回归
multi_reg = MultiOutputRegressor(base_reg)

# 训练+预测
multi_reg.fit(X, y)
y_pred = multi_reg.predict(X)

print("真实值：\n", y)
print("预测值：\n", np.round(y_pred,1))