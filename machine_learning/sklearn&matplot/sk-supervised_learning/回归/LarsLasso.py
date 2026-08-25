from matplotlib import pyplot as plt
from sklearn.linear_model import LassoLarsCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression

"""
步骤：
1、先选和残差内积最大的特征x1
2、不直接拟合完、不做正交投影，沿着x1方向慢慢往前滑
3、一直滑到：残差和x1、x2的内积变得相等
4、不再沿x1走，改走x1,x 2的等角线（角平分线） 继续滑
5、再往前滑，直到残差和x1 x2 x3三者内积相等
6、再换三者等角方向继续走…… 循环

缺点：
1、噪音敏感
2、数据量远大于特征数
3、对数据缩放季度苛刻
"""


# 造数据：注意！样本量(1000) 必须大于 特征量(20)，否则 LARS 会报错
X, y = make_regression(n_samples=1000, n_features=20, noise=5, random_state=0)

# 黄金法则：LARS 对缩放的要求到了丧心病狂的程度，必须做！
model = make_pipeline(
    StandardScaler(),
    # cv=5 代表用交叉验证选最佳 alpha
    LassoLarsCV(cv=5)
)

model.fit(X, y)

# 提取模型
lars = model.named_steps["lassolarscv"]

print(f"选出的最佳 alpha: {lars.alpha_}")
print(f"非零特征数量: {sum(lars.coef_ != 0)}")

# --- LARS 独有的神仙功能：查看正则化路径 ---
# lars.alphas_ 记录了它滑行过程中经历的所有“拐点” alpha
# lars.coef_path_ 记录了在这些拐点时，每个特征的权重是多少
print(f"经历的拐点数量: {len(lars.alphas_)}")

# 取出路径数据 (特征数 × 拐点数)
coefs_path = lars.coef_path_
print(coefs_path)
# 画图
plt.figure(figsize=(10, 6))
for i in range(coefs_path.shape[0]):
    plt.plot(lars.alphas_, coefs_path[i, :], label=f'Feature {i}')

plt.xlabel('Alpha (惩罚力度，从右往左减小)')
plt.ylabel('权重系数')
plt.title('LARS-Lasso 正则化路径图')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 增加图例
plt.gca().invert_xaxis() # 翻转X轴，让 alpha 从大到小显示（符合直觉）
plt.show()