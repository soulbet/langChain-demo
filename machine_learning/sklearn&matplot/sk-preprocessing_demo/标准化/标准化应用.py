from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
## 创建管道
# 1、把 X_train 传给 StandardScaler()，调用它的 .fit()，算出均值和标准差。
# 2、用刚刚算出的均值和标准差，对 X_train 调用 .transform()，完成标准化。
# 3、把标准化后的 X_train 传给 LogisticRegression()，训练逻辑回归模型
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)

# 这里直接拿出刚才在 X_train 上算好的均值和标准差，对 X_test 调用 .transform() 进行标准化。
# 用标准化后的 X_test 去让逻辑回归模型做预测，最后计算准确率。
score = pipe.score(X_test, y_test)
print(score)
lr_model = pipe.named_steps['logisticregression']

# 2. 获取系数 - 对应每个特征的权重
# 因为数据被 transform 成了均值为 0、标准差为 1 的状态，所有特征都在同一起跑线上。此时系数的绝对值越大，
# 说明该特征对模型预测结果的影响越大！ 你可以直接排序看哪个特征最重要。
print("系数:\n", lr_model.coef_)

# 3. 获取常数/截距
# 代表“当所有特征都等于平均值时的概率”
# 在所有特征都取平均值的条件下，样本被预测为正类的基准概率是多少。
print("常数:\n", lr_model.intercept_)