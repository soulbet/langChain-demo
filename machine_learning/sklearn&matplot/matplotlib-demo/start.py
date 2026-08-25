# 导入matplotlib绘图库和numpy数值计算库
import matplotlib.pyplot as plt
import numpy as np

# 生成从0到2π的等间距数据点（共200个点）
x = np.linspace(0, 2 * np.pi, 200)
# 计算每个x点对应的正弦值
y = np.sin(x)

# 创建图形和坐标轴对象，也就是Figure和Axes
figure = plt.figure() # 创建一个没有axes空的Figure对象
fig, ax = plt.subplots()
# 绘制正弦曲线
ax.plot(x, y)
ax.set_title("正弦曲线") # 设置标题
ax.set_xlabel("x") # 设置x轴标签
ax.set_ylabel("y") # 设置y轴标签
# 显示图形
plt.show()
