
"""
公式：
目标 = 1/2n（y-y预）^2+α∑w^2   实际使用时，α使用的是α/2 ，为了在求导时，将平方的导出产生的2化掉
/n 是为了均方误差，防止随着总样本变法，误差无限变大

所有权重的平方和   （W1^2 + W2^2 W3^2 W4^2）

1、解决多重共线问题
特征之间有重复信息，
2、模型稳定性
不会因为异常值的存在，导致模型较大波动
3、防止过拟合
通过压制权重的幅度，它强制模型去关注所有特征中比较靠谱的那些，而不是死记硬背训练集里的噪声。

只能把权重压小，但永远不会把权重变成真正的 0
"""
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
y_train = rng.randn(n_samples)
X_train = rng.randn(n_samples, n_features)
X_test=rng.randn(20,5)
# 黄金法则：Ridge 必须和 StandardScaler 绑死在一起！
# 因为如果不缩放，alpha=0.5 对大数值特征的惩罚力度，和对小数值特征的惩罚力度是不公平的。
# model = make_pipeline(
#     StandardScaler(),
#     Ridge(alpha=0.5)
# )
# RidgeCV 会自动生成一堆 alpha 去试（默认会试 100 个值）
model = make_pipeline(StandardScaler, RidgeCV(cv=5, alphas=np.logspace(-6, 6, 5)))
print(X_test)
# 然后正常 fit 和 predict
model.fit(X_train, y_train)
Ridge_name = model.named_steps["ridge"]
print(f"系数：{Ridge_name.coef_}")
print(f"截距：{Ridge_name.intercept_}")

print(model.predict(X_test))
