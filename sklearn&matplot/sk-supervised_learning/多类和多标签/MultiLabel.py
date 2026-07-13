from sklearn.preprocessing import MultiLabelBinarizer

# 多标签数据（每个样本多个标签）
y = [
    ['猫', '狗'],
    ['鸟', '狗'],
    ['猫']
]

# 转换
mlb = MultiLabelBinarizer()
y_new = mlb.fit_transform(y)

print(mlb.classes_)  # 输出标签顺序：['猫', '狗', '鸟']
print(y_new)