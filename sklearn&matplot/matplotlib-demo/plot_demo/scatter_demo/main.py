

import matplotlib.pyplot as plt
import numpy as np

"""
散点图
"""

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([1, 4, 9, 16, 7, 11, 23, 18])
sizes = np.array([20,50,100,200,500,1000,60,90])
colors = np.array(["red","green","black","orange","purple","beige","cyan","magenta"])
# s=sizes 设置点的大小
# c=colors 设置点的颜色
plt.scatter(x, y, s=sizes, c=colors)


## 设置第二组散点图
x = np.array([2,2,8,1,15,8,12,9,7,3,11,4,7,14,12])
y = np.array([100,105,84,105,90,99,90,95,94,100,79,112,91,80,85])
plt.scatter(x, y, color = '#88c999')

plt.show()