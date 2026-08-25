"""
RANSAC (RANdom SAmple Consensus)能够使用完整数据集中的样本点组成的随机子集拟合模型。
RANSAC是一种不确定的算法，它以一定概率产生一个合理的结果，而这还取决于迭代次数(见参数 max_trials)它通常用于线性和非线性回归问题，在计算机视觉领域尤其流行。
该算法将完整的输入样本数据分割成一组可能样本点集合， 该集合收到噪声和离群值的影响，这些离群值是由于对数据的错误测量或无效假设引起的。然后，仅从集合中估计得到模型。

步骤：
1、从原始数据中随机选择min_samples个样本，并检查数据集是否有效(请参阅is_data_valid)
2、将模型拟合到随机子集上(base_estimator.fit)，并检查估计的模型是否有效(请参阅is_model_valid)
3、通过计算模型的残差(base_estimator.predict(X) - y), 如果样本绝对残差小于residual_threshold就会被认为是局内点。按照这种方式将数据分为局内点和离群点。
4、当内部的局内样本数达到最大时，模型达到最优就保存下来。如果当前的估计模型有相同的局内点，只有当它有更好的分数时，它才被认为是最好的模型。

"""

import numpy as np
from matplotlib import pyplot as plt

from sklearn import linear_model, datasets


n_samples = 1000
n_outliers = 50


# 生成包含噪声和离群值的回归数据集
X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1,
                                      n_informative=1, noise=10,
                                      coef=True, random_state=0)

# 人为添加离群点到数据集的前50个样本
np.random.seed(0)
X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)


# 训练普通线性回归模型
lr = linear_model.LinearRegression()
lr.fit(X, y)

# 训练RANSAC回归模型并获取局内点和离群点的掩码
ransac = linear_model.RANSACRegressor()
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# 生成用于绘制回归线的预测数据
line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_y = lr.predict(line_X)
line_y_ransac = ransac.predict(line_X)


# ... existing code ...

# 可视化对比结果：区分局内点、离群点以及两种回归方法的拟合直线
lw = 2
plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',
            label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.',
            label='Outliers')
plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')
plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,
         label='RANSAC regressor')
plt.legend(loc='lower right')
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()