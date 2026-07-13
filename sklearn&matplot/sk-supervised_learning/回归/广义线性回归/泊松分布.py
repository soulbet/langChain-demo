import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_poisson_deviance

"""
泊松分布的连接函数：
ln(λ)=η
"""

# ==========================================
# 1. 构造模拟数据
# ==========================================
np.random.seed(42)
n_samples = 1000

# 自变量
time_on_site = np.random.randint(10, 300, size=n_samples)  # 停留时间 10-300分钟
num_visits = np.random.randint(1, 10, size=n_samples)      # 访问次数 1-10次
is_premium = np.random.choice([0, 1], size=n_samples)      # 是否高级会员

# 真实的系数 (我们假设知道真实的底层数学关系)
b0, b1, b2, b3 = -2.5, 0.01, 0.3, 0.8

# 根据泊松回归公式计算期望值 lambda: λ = e^(b0 + b1*x1 + b2*x2 + b3*x3)
lambda_true = np.exp(b0 + b1*time_on_site + b2*num_visits + b3*is_premium)

# 根据期望值 lambda 生成服从泊松分布的购买数量 Y
purchase_count = np.random.poisson(lam=lambda_true)

# 整理成 DataFrame
df = pd.DataFrame({
    'time_on_site': time_on_site,
    'num_visits': num_visits,
    'is_premium': is_premium,
    'purchase_count': purchase_count
})

print("数据前5行：")
print(df.head())
print("-" * 50)

# ==========================================
# 2. 划分训练集和测试集
# ==========================================
X = df[['time_on_site', 'num_visits', 'is_premium']]
y = df['purchase_count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. 训练 PoissonRegressor 模型
# ==========================================
# alpha: L2正则化项的系数，相当于岭回归中的alpha，可以防止过拟合。默认是1。
# fit_intercept: 是否计算截距项(b0)。默认是True。
model = PoissonRegressor(alpha=0.1, fit_intercept=True)

# 拟合模型 (就这么简单！)
model.fit(X_train, y_train)

# ==========================================
# 4. 查看模型学到的系数
# ==========================================
print("模型学到的截距 (b0):", model.intercept_)
print("模型学到的系数 (b1, b2, b3):", model.coef_)

# 对比一下我们生成数据时用的真实值：b0=-2.5, b1=0.01, b2=0.3, b3=0.8
# 可以看到模型学到的系数非常接近真实值！
print("-" * 50)

# ==========================================
# 5. 进行预测
# ==========================================
# 预测的值就是期望值 lambda
y_pred = model.predict(X_test)

# 看看前5个真实值和预测值的对比
print("前5个真实购买数量:", y_test.values[:5])
print("前5个预测购买数量:", np.round(y_pred[:5], 2))
print("-" * 50)

# ==========================================
# 6. 模型评估 (泊松偏差)
# ==========================================
# 对于泊松回归，最合适的评估指标是泊松偏差 而不是 MSE
# 偏差越小，模型越好
poisson_deviance = mean_poisson_deviance(y_test, y_pred)
print("测试集上的泊松偏差:", poisson_deviance)
