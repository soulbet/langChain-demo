from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification

"""
将回归计算的连续数值，返回-1，1分类
1、比逻辑回归快
2、支持类别不平衡的类别设置
3、抗多重共线性，需搭配标准化
4、不能输出概率
"""

# 造一些假数据 (1000个样本, 20个特征, 4个类别)
X, y = make_classification(n_samples=1000, n_features=20, n_classes=4, random_state=0)

# 黄金法则：Ridge系必须缩放！
model = make_pipeline(
    StandardScaler(),
    RidgeClassifier(alpha=0.5)  # 注意这里换成了分类器
)

model.fit(X, y)

# 预测类别
y_pred = model.predict(X[:5])
print("预测结果:", y_pred)

# 想看内部 Ridge 回归的权重？依然用 named_steps！
ridge_inside = model.named_steps["RidgeClassifier"]
print("权重形状:", ridge_inside.coef_.shape) # 会输出 (4, 20)，因为有4个类别，内部训练了4个回归器
