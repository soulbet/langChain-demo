import numpy as np
from sklearn.preprocessing import QuantileTransformer

"""
效果： 变换后的数据，完美、绝对、100% 服从正态分布（如果你指定了正态分布的话）。
它也能把数据完美压缩到 0~1 之间（类似 MinMaxScaler，但抗异常值能力极强）。
缺点： 它破坏了特征之间的线性关系（如果特征A和特征B原来是线性相关，变换后可能不是了），
而且可解释性变差（因为你不知道具体的数学公式了，变成了一种基于排名的映射）。
"""

X = np.array([[1, 2],])
qt = QuantileTransformer(output_distribution='normal', random_state=42)
X_transformed = qt.fit_transform(X)
