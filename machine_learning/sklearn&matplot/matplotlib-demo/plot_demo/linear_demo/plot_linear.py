import matplotlib.pyplot as plt
import numpy as np

x=np.array([1,2,3])
y=np.array([4,8,9])

"""
标记大小与颜色
我们可以自定义标记的大小与颜色，使用的参数分别是：
markersize，简写为 ms：定义标记的大小。
markerfacecolor，简写为 mfc：定义标记内部的颜色。
markeredgecolor，简写为 mec：定义标记边框的颜色。

fmt = '[marker][line][color]':

ls
"""

# plt.plot(x,y,'r',marker='o')

# 虚线
plt.plot(x,y,'o:r', ls=":")
plt.show()