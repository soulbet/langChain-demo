import matplotlib.pyplot as plt
import numpy as np

# 生成从0到2π的等间距数据点（共200个点）
x = np.linspace(0, 2 * np.pi, 200)
# 计算每个x点对应的正弦值
y = np.sin(x)
fig,ax = plt.subplots()
ax.plot(x, y)
plt.axis([0, 10, -5, 5]) # 设置坐标轴范围
plt.axis('tight') # 自动把坐标轴缩到刚好包住所有数据
plt.axis('equal') # x、y 轴单位长度一样，圆不会被拉成椭圆，几何画图必用
plt.axis('off')   # 隐藏坐标轴、刻度、边框
plt.axis('on')    # 恢复显示
plt.axis('square')   # 强制画布变成正方形
plt.axis('scaled')   # 等比例+适配画布
plt.axis('auto')     # 恢复默认自动刻度
plt.show()
