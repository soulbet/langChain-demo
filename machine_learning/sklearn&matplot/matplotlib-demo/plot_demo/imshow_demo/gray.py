import matplotlib.pyplot as plt
import numpy as np

"""
灰度图
"""

# 生成一个二维随机数组
img = np.random.rand(10, 10)

# 绘制灰度图像
plt.imshow(img, cmap='gray')

# 显示图像
plt.show()