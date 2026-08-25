from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression

"""
假设：
平面里两个特征向量 x1,x2，目标向量 y，一开始残差 r=y
y=(4,3)
x1=(1,0)
投影y = (y * x1 / x1 * x1 ) * x1
新残差 = y - 投影y

OMP使用内积选中最大相关性的特征，然后所有向量与目标向量做一次最小二乘投影，保证这个特征不会被选中，
用新的残差再去选内积最大的特征，最后用所有被选中的 K 个特征，做最小二乘拟合，得到最终稀疏系数。

最小二乘投影：
把已选特征能解释的信息全部榨干
残差只保留已选特征解释不了的正交分量
从根源避免重复选、冗余拟合

缺点：无法回溯
场景：
1、极度稀疏的数据
2、特征数远大于样本数
"""


# 造假数据：100个样本，1000个特征（典型的高维稀疏场景）
X, y = make_regression(n_samples=100, n_features=1000, n_informative=10, noise=10, random_state=0)

# 黄金法则：依然要做标准化
model = make_pipeline(
    StandardScaler(),
    # 核心参数：n_nonzero_coefs 指定只保留 15 个非零特征！
    OrthogonalMatchingPursuit(n_nonzero_coefs=15)
)

model.fit(X, y)

omp = model.named_steps["orthogonalmatchingpursuit"]

print(f"选出的非零特征数量: {sum(omp.coef_ != 0)}")  # 输出必定是 15
