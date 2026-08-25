import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from scipy import sparse

"""
如果原始数据里的最小值不是 0（比如某列特征的值在 1 到 5 之间），那么公式中的减法会让原本为 0 的元素变成负数（0−1=−1）。
稀疏矩阵中一旦出现了大量的非零值（哪怕是 -1），它就变成了“密集矩阵”，内存直接爆炸！
即使 最小值刚好是 0，由于除法操作（除以极差），底层计算时也往往会强制转换为密集矩阵来运算，极其危险。
"""
X_train_sparse= np.diag(np.arange(1, 6))
print(X_train_sparse)
# 假设 X_train_sparse 是一个稀疏矩阵
# X/max(|X|) 数据除以该列的最大绝对值。
scaler = MaxAbsScaler()
# 它能安全地处理稀疏矩阵，且不破坏稀疏结构
X_train_scaled = scaler.fit_transform(X_train_sparse)


# 保持正太分布
# 核心参数：with_mean=False
scaler = StandardScaler(with_mean=False)
X_train_sparse = scaler.fit_transform(X_train_sparse)

from sklearn.feature_extraction.text import TfidfVectorizer

# 处理的是文本数据，并且正在做词频统计
# sublinear_tf=True 是平滑词频（常用技巧）
# norm=None 关闭它默认的L2归一化
# use_idf=False 不计算逆文档频率（这就变成了纯粹的词频统计）
# 然后用 binary=False, dtype=np.float64 保持原样
# 其实最直接的是直接开启 norm='l1' 或保持默认的 'l2'

# 最标准的做法：什么都不用管，它默认就给你缩放好了
text_data = ""
vec = TfidfVectorizer(norm='l2')
X_scaled = vec.fit_transform(text_data)

