from sklearn import preprocessing

"""
根据指定阈值，返回数据的布尔值0或1
"""

X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]

# 默认 threshold=0.0
binarizer = preprocessing.Binarizer().fit(X)  # fit does nothing

print(binarizer.transform(X))