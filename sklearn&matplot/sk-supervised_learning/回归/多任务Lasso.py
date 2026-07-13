
from sklearn.linear_model import MultiTaskLasso, MultiTaskLassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

"""
只挑选对所有目标共同有用的特征，做不到“共同有用”的直接物理消灭
适用场景：多标签预测、多指标同时回归、且这些标签/指标之间逻辑强相关。
"""

# 1. 数据形状要求！
n_samples = 100    # 100个样本（房子）
n_features = 20    # 20个特征（面积、地段等）
n_tasks = 3        # 3个任务（售价、租金、物业费）

X = np.random.randn(n_samples, n_features)

# 注意看 y 的形状！必须是二维的 (100, 3)
# 普通Lasso的y是一维 (100,)，多任务必须是二维！
y = np.random.randn(n_samples, n_tasks)

# 2. 依然是黄金法则：必须缩放！
# 强烈建议用 MultiTaskLassoCV 自动找 alpha
model = make_pipeline(
    StandardScaler(),
    MultiTaskLassoCV(cv=3)
)

model.fit(X, y)

# 3. 查看结果
mt_lasso = model.named_steps["multitasklassocv"]
print(f"最佳 alpha: {mt_lasso.alpha_}")

# 权重形状变成了 (3, 20) -> 3个任务，每个任务20个特征权重
print(f"权重矩阵形状: {mt_lasso.coef_.shape}")

# 神奇的时刻：看看是不是整列都被砍成了0
# axis=0 代表按列（特征）看
zero_features_count = np.sum(np.all(mt_lasso.coef_ == 0, axis=0))
print(f"被彻底砍掉（三个任务全为0）的特征数量: {zero_features_count}")
