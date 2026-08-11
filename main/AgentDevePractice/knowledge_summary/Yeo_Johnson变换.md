## 1. 核心定义
Yeo-Johnson变换是一种用于数据预处理的非线性变换方法，它可以将数据转换为接近正态分布的形式，从而提高后续分析的准确性。

## 2. 数学原理/公式
Yeo-Johnson变换通过将数据映射到一个新的空间，使得数据在该空间中更接近正态分布。变换公式如下：
$$
y = \begin{cases} 
\left(\frac{x + \lambda}{1 + \lambda}\right)^{\frac{1}{\lambda}} & \text{if } \lambda \neq 0 \\
\log(x + 1) & \text{if } \lambda = 0 
\end{cases}
$$
其中，$\lambda$ 是一个变换参数，通过优化使得变换后的数据更接近正态分布。

## 3. 关键要点
- Yeo-Johnson变换可以处理正数、负数和零。
- 变换后的数据通常更接近正态分布。
- 可以自动计算最优的 $\lambda$ 参数。

## 4. 代码示例 (Python)
```python
from sklearn.preprocessing import PowerTransformer
import numpy as np

X = np.array([[1, -2, 2],
              [-3, 1, 0],
              [0, 1, -1]])
pt = PowerTransformer(method='yeo-johnson', standardize=True)
X_transformed = pt.fit_transform(X)
print("自动计算的最优lambda:", pt.lambdas_)
```

## 5. 面试常见问题
1. 什么是 Yeo-Johnson变换？
2. Yeo-Johnson变换如何处理正数、负数和零？
3. 如何计算 Yeo-Johnson变换的最优 $\lambda$ 参数？

## 6. 优缺点
- 优点: 可以处理正数、负数和零，变换后的数据更接近正态分布。
- 缺点: 计算复杂度相对较高，可能需要较长时间。

## 7. 常用的使用场景
- 数据预处理，特别是在需要进行正态性检验或使用假设检验时。
- 机器学习模型训练，特别是在输入数据需要满足正态分布假设时。