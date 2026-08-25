import numpy as np
a = np.array([1, 2, 3, 4, 5, 6])
b = a[:2]
b += 1
print('a =', a, '; b =', b)

# 新数组
b_c = a[:2].copy()
print(f"复制：{b_c}")