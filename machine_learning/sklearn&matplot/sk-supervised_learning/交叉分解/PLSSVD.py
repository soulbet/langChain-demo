
"""
PLSSVD	降维简化版：基于SVD的简单降维	对交叉协方差矩阵X'Y做SVD分解	仅用于数据降维和可视化，不做预测
工作原理：它直接对交叉协方差矩阵X'Y进行奇异值分解（SVD），得到X和Y的权重。

适用场景：

仅用于降维：当你只想得到X和Y的低维表示，而不关心如何用X去预测Y时，它是一个轻量级选择。

其他算法的预处理步骤：作为更复杂模型中的一步。
"""
import numpy as np
from sklearn.cross_decomposition import PLSSVD
X = np.array([[0., 0., 1.],
[1.,0.,0.],
[2.,2.,2.],
[2.,5.,4.]])
Y = np.array([[0.1, -0.2],
[0.9, 1.1],
[6.2, 5.9],
[11.9, 12.3]])
plsca = PLSSVD(n_components=2)
plsca.fit(X, Y)
PLSSVD()
X_c, Y_c = plsca.transform(X, Y)
