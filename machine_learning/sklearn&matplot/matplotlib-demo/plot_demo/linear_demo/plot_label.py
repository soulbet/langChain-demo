import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])
plt.plot(x, y)

font1 = {'color':'blue','size':20}
font2 = {'color':'darkred','size':15}

# 标题
plt.title("RUNOOB TEST TITLE")

# x,y标签,fontdict设置字体
# loc 显示标题位置
plt.xlabel("x - label", fontdict=font1, loc="left")
plt.ylabel("y - label")

plt.show()