
"""
逻辑回归示例：使用鸢尾花数据集进行分类任务

本示例演示了如何使用sklearn的逻辑回归模型进行：
1. 数据加载
2. 模型训练
3. 预测分类
4. 概率预测
5. 模型评估
"""
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# 加载鸢尾花数据集，返回特征矩阵X和标签向量y
X, y = load_iris(return_X_y=True)

# 创建逻辑回归分类器并训练模型（设置随机种子确保可重复性）
clf = LogisticRegression(random_state=0).fit(X, y)

# 对前两个样本进行类别预测
clf.predict(X[:2, :])

# 对前两个样本进行概率预测，返回每个类别的预测概率
clf.predict_proba(X[:2, :])

# 计算模型在训练集上的准确率评分
clf.score(X, y)