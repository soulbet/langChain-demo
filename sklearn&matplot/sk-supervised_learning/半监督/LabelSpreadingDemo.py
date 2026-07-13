from sklearn.datasets import load_iris
from sklearn.semi_supervised import LabelSpreading


"""
LabelPropagation、LabelSpreading 同属图半监督，核心差异在损失约束与标签更新规则
算法优先保证近邻样本标签相近，压制标签剧烈跳变，类别分界过渡自然，不会出现突兀的类别切换。
"""
X, y = load_iris(return_X_y=True)
# 构造半监督：部分标签置-1代表无标注
y_semi = y.copy()
y_semi[30:] = -1
ls = LabelSpreading(kernel='knn', n_neighbors=5)
ls.fit(X, y_semi)

print("标签扩散结果：")
print(ls.transduction_)