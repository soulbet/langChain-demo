import matplotlib.pyplot as plt
import numpy as np

"""
设置 numRows ＝ 2，numCols ＝ 2，就是将图表绘制成 2x2 的图片区域, 对应的坐标为：

(1, 1), (1, 2)
(2, 1), (2, 2)
plotNum ＝ 1, 表示的坐标为(1, 1), 即第一行第一列的子图。
plotNum ＝ 2, 表示的坐标为(1, 2), 即第一行第二列的子图。
plotNum ＝ 3, 表示的坐标为(2, 1), 即第二行第一列的子图。
plotNum ＝ 4, 表示的坐标为(2, 2), 即第二行第二列的子图。
"""

#plot 1:
x = np.array([0, 6])
y = np.array([0, 100])

plt.subplot(2, 2, 1)
plt.plot(x,y)
plt.title("plot 1")

#plot 2:
x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])

plt.subplot(2, 2, 2)
plt.plot(x,y)
plt.title("plot 2")

#plot 3:
x = np.array([1, 2, 3, 4])
y = np.array([3, 5, 7, 9])

plt.subplot(2, 2, 3)
plt.plot(x,y)
plt.title("plot 3")

#plot 4:
x = np.array([1, 2, 3, 4])
y = np.array([4, 5, 6, 7])

plt.subplot(2, 2, 4)
plt.plot(x,y)
plt.title("plot 4")

plt.suptitle("RUNOOB subplot Test")
plt.show()