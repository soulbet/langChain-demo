"""
多分类组合成二分类

工作原理（3 步看懂）
假设 4 个类：猫、狗、鸟、鱼
第一步：设计编码表（每行 = 类别，每列 = 1 个二分类器）
比如码长 = 3（3 个二分类器）：
猫：0 0 1
狗：0 1 0
鸟：1 0 0
鱼：1 1 1
第二步：按列训练二分类器
第 1 列：猫/狗=0，鸟/鱼=1 → 训练分类器 f1
第 2 列：猫/鸟=0，狗/鱼=1 → 训练分类器 f2
第 3 列：猫/鱼=1，狗/鸟=0 → 训练分类器 f3
第三步：预测 + 解码
新样本 → f1/f2/f3 输出 0 1 1计算与 4 个类别编码的汉明距离（不同位的数量）：
猫 (001)：距离 1
狗 (010)：距离 1
鸟 (100)：距离 2
鱼 (111)：距离 1
→ 选距离最小的（猫 / 狗 / 鱼，通常取第一个或用概率加权）

"""


from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 数据
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. 基础二分类器 + Output-Codes包装
base_clf = LinearSVC(random_state=0)
# code_size=1.5 → 分类器数量≈1.5*类别数（3类→4~5个）
ecoc = OutputCodeClassifier(base_clf, code_size=1.5, random_state=0)

# 3. 训练预测
ecoc.fit(X_train, y_train)
y_pred = ecoc.predict(X_test)

# 4. 结果
print("准确率：", accuracy_score(y_test, y_pred))
print("编码矩阵形状：", ecoc.code_.shape)  # (类别数, 分类器数)