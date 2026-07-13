
import matplotlib.pyplot as plt
import numpy as np

"""
柱状图
"""

x = np.array(["Runoob-1", "Runoob-2", "Runoob-3", "C-RUNOOB"])
y = np.array([12, 22, 6, 18])

plt.bar(x,y, )

# 垂直方向
# plt.barh(x,y)
plt.show()