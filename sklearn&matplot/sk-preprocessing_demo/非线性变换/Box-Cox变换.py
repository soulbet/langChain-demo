from sklearn import preprocessing


import numpy as np


"""
BoxCox只能应用到正值
"""
X = np.array([[1., -1., 2.],
             [2., 0., 0.],
             [0., 1., -1.]])
pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
X_lognormal = np.random.RandomState(616).lognormal(size=(3, 3))
pt.fit_transform(X_lognormal)