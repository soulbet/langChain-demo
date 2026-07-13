"""
原始样本只能分两类，OneVsRest可以将两类做多类标签z

"""
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

# 1. 构造多标签样本
X = np.array([[1,2], [3,4], [5,6], [7,8]])
# 每个样本附带多个标签
y_raw = [['体育','娱乐'], ['科技'], ['体育','财经'], ['娱乐','财经']]

# 2. 多标签转0-1矩阵
mlb = MultiLabelBinarizer()
y_bin = mlb.fit_transform(y_raw)
print("标签映射顺序：", mlb.classes_)
print("二值化标签：\n", y_bin)

# 3. OvR包装二分类模型，适配多标签
base_clf = LogisticRegression(max_iter=1000)
model = OneVsRestClassifier(base_clf)

# 4. 训练预测
model.fit(X, y_bin)
y_pred = model.predict(X)

# 5. 转回原始标签文本
pred_labels = mlb.inverse_transform(y_pred)
print("\n预测标签结果：")
for res in pred_labels:
    print(res)