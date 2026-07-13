"""
高斯过程：把所有可能的曲线都考虑进来，给每条曲线一个概率，本身也是贝叶斯方法输出后验分布
高斯过程可以看作无限维的高斯分布。它由两个函数完全定义：
1、均值函数 m(x)：描述函数的"基准值"
2、协方差函数（核函数）k(x,x′)：描述不同输入点之间的"相关性"f(x)∼GP(m(x),k(x,x ′))
2.1、均值函数通常取 m(x)=0（先验假设函数围绕 0 波动）
2.2、核函数决定函数的平滑性、周期性等性质

优缺点：
1、小样本（n < 几千）、高维
传统方法容易过拟合或欠拟合；神经网络需要大数据；GP 在数据少时依然稳健，且给出置信区间。
2、需要不确定性估计（医疗、金融、工程安全）
天然输出方差，可计算置信区间。
3、贝叶斯优化 / 超参数搜索（AutoML 常用）
问题：调超参数，每次评估代价极高（如训练一个神经网络要几小时），希望在尽量少的尝试中找到最优参数。
为什么用 GP：GP 给出预测 + 不确定性，可以用采集函数（如 EI、UCB）决定下一步试哪里——在"可能最优"和"不确定区域"之间平衡。
4、时间序列 + 不确定度
预测未来值，同时知道置信区间。
5、主动学习（选最不确定的样本标注）
不适合
1、大数据（n > 1 万）：太慢（O (n³)）
2、强高维稀疏（如文本）：不如线性模型
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# 生成数据
X = np.linspace(0, 10, 20).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# 定义核函数：RBF + 白噪声
kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

# 创建高斯过程回归模型
gp = GaussianProcessRegressor(kernel=kernel, alpha=0, n_restarts_optimizer=10)

# 训练
gp.fit(X, y)

# 预测（带不确定性）
X_pred = np.linspace(0, 12, 100).reshape(-1, 1)
y_mean, y_std = gp.predict(X_pred, return_std=True)

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='red', label='训练数据')
plt.plot(X_pred, y_mean, 'b-', label='预测均值')
plt.fill_between(X_pred.ravel(),
                  y_mean - 1.96*y_std,
                  y_mean + 1.96*y_std,
                  alpha=0.2, label='95% 置信区间')
plt.legend()
plt.title('高斯过程回归')
plt.show()

print(f"优化后的核函数: {gp.kernel_}")