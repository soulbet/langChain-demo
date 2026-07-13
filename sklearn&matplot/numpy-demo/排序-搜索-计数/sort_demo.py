import numpy as np
from numpy.ma.core import shape
from numpy.random import default_rng

data = default_rng(42).random((2, 3))
print(data)
# axis=None,会展平数组，axis=1 表示按行排序，axis=0 表示按列排序，行的方向(垂直方向)
print(np.sort(data, 0))
