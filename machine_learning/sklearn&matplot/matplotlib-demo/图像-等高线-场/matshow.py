

import matplotlib.pyplot as plt
import numpy as np

# 将二维矩阵或数组可视化为颜色编码图像。
a = np.diag(range(15))
print( a)

# 保护矩阵行列结构不被拉伸扭曲
plt.matshow(a)

plt.show()