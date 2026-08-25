import numpy as np
from sklearn.preprocessing import Normalizer


X = np.array([[0., -3., 1.],
              [3., 1., 2.],
              [0., 1., -1.]])

normalizer = Normalizer(norm='l2') # l1, l2, max 可选
X_normalized = normalizer.fit_transform(X)
