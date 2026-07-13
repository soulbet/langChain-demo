"""
最近邻算法（K-Nearest Neighbors, KNN）
KNN 是一种基于实例的监督学习算法，通过查找待预测样本在训练集中最相似的 K 个样本（最近邻），
然后根据这些邻居的标签进行投票（分类）或平均（回归）
1、k太小   对噪音敏感  决策边界复杂
2、K太大	决策边界平滑，欠拟合	可能包含异类
3、K=1	最近邻分类（1-NN）	完全过拟合
4、K≈sqrt(N)	经验规则	常用起点
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 数据准备
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特征缩放（必须！KNN 对尺度敏感）
scaler = StandardScaler()
# 学习训练数据的均值/标准差，并同时将训练数据标准化
X_train = scaler.fit_transform(X_train)
# 用训练数据学到的参数，只转换测试数据（不再重新学习）
X_test = scaler.transform(X_test)

# 模型训练
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
knn.fit(X_train, y_train)  # 实际上只是存储数据

# 预测与评估
y_pred = knn.predict(X_test)
print(f"准确率: {knn.score(X_test, y_test):.3f}")