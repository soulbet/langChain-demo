
import matplotlib.pyplot as plt
import numpy as np

# 造点测试数据
x = np.linspace(-3, 3, 100)
y = x ** 2

# 1. 默认绘图
plt.figure(figsize=(10, 8))

plt.subplot(221)
plt.plot(x, y)
plt.title("默认坐标轴")

# 2. 手动设置坐标范围 [xmin, xmax, ymin, ymax]
plt.subplot(222)
plt.plot(x, y)
plt.axis([-5, 5, 0, 10])
plt.title("plt.axis([xmin,xmax,ymin,ymax])")

# 3. 等比例坐标轴 x/y 单位长度一致
plt.subplot(223)
plt.plot(x, y)
plt.axis("equal")
plt.title("axis('equal') 等比例")

# 4. 紧凑边界 + 关闭坐标轴
plt.subplot(224)
plt.plot(x, y)
plt.axis("tight")   # 紧贴数据范围
plt.axis("off")      # 隐藏坐标轴
plt.title("tight + off 隐藏坐标")

plt.tight_layout()
plt.show()