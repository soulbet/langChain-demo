"""
公式：
1/n ∑(y-y预)^2 + α ∑|w|

1、特征选择，可以将系数惩罚成0，只保留有用的特征
2、多重共线处理差
3、数据量必须大于特征数

"""
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 告诉 Lasso：别瞎猜了，就试这 5 个数
my_alphas = [0.0001, 0.001, 0.01, 0.1, 1]

model_custom = make_pipeline(
    StandardScaler(),
    LassoCV(alphas=my_alphas, cv=5)
)
model_custom.fit(X, y)

# 它一定会从这 5 个里挑一个最好的
print("自定义队列选出的最佳值:", model_custom.named_steps["lassocv"].alpha_)
