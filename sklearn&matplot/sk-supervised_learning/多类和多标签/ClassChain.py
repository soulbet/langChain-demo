"""
是一种将多个二分类器组合成一个能够利用目标之间相关性的单一多标签模型的方法
训练阶段后序分类器输入 = 原始特征 + 前面所有标签的真实标签值
预测阶段后序分类器输入 = 原始特征 + 前面所有标签的模型预测值
"""

import numpy as np
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score

# 1. 生成多标签数据（3个标签）
X, y = make_multilabel_classification(n_samples=1000, n_features=10,
                                         n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. 基分类器 + 分类器链
base_clf = LogisticRegression()
# order=None → 按标签0,1,2顺序；也可写 order=[2,0,1]
chain = ClassifierChain(base_clf, order=None, random_state=42)

# 3. 训练+预测
chain.fit(X_train, y_train)
y_pred = chain.predict(X_test)

# 4. 评估（多标签常用 Jaccard）
print("Jaccard 相似度：", jaccard_score(y_test, y_pred, average="samples"))