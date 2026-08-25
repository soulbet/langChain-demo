from sklearn import preprocessing

"""
OrdinalEncoder：赋予大小顺序（序数编码）
把类别变成整数，并且强行规定它们之间有大小/顺序关系。
"""

enc = preprocessing.OrdinalEncoder()
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)
print(enc.transform([['female', 'from US', 'uses Safari']]))


"""
OneHotEncoder：制造正交关系（独热编码）
核心思想：承认类别之间毫无大小关系，为每一个类别创造一个全新的“是/否（0或1）”列。
"""

from sklearn.preprocessing import OneHotEncoder
X = [["红"], ["绿"], ["蓝"]]
# 默认输出稀疏矩阵(sparse=True)节省内存，为了好看我们设为 False
enc = OneHotEncoder(sparse_output=False)
print(enc.fit_transform(X))
# 输出：
# [[1. 0. 0.]  红色
#  [0. 1. 0.]  绿色
#  [0. 0. 1.]] 蓝色


"""
LabelEncoder 是专门用来处理“标签（目标变量 Y）”的，而绝对不应该用来处理“特征（X）”。
只接受一维数据 按字典序排序打标签
"""

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = ["猫", "狗", "狗", "猫", "鸟"]
y_transformed = le.fit_transform(y)
print(y_transformed)
# 一键还原成文字！
y_original = le.inverse_transform(y_transformed)
print(y_original)
# 输出: [0 1 1 0 2]
