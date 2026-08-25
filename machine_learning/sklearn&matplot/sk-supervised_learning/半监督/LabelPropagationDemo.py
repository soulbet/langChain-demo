from sklearn.semi_supervised import LabelPropagation
from sklearn.datasets import load_iris
import numpy as np
"""
LabelPropagation（标签传播）
基于样本相似度构建图结构，标签顺着邻近相似样本自动扩散传递。
"""
X, y = load_iris(return_X_y=True)
# 构造半监督：部分标签置-1代表无标注
y_semi = y.copy()
y_semi[30:] = -1

model = LabelPropagation()
model.fit(X, y_semi)
y_pred = model.transduction_