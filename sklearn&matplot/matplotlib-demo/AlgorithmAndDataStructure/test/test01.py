import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 假设这是你的数据，每一帧保存一次排序过程中的数组状态
# 这里用一个简单的列表模拟，你可以替换成自己排序函数的中间结果
data_by_frame = [
    [5, 3, 8, 1],  # 帧0：初始状态
    [3, 5, 8, 1],  # 帧1：第一次交换后
    [3, 5, 1, 8],  # 帧2：...
    [3, 1, 5, 8],
    [1, 3, 5, 8]   # 帧N：排序完成
]

fig, ax = plt.subplots()

def update(frame):
    ax.clear()  # 清空上一帧的内容
    # 根据当前帧数取出数据，绘制柱状图
    ax.bar(range(len(data_by_frame[frame])), data_by_frame[frame])
    ax.set_title(f'排序过程 - 第{frame+1}步')
    ax.set_ylim(0, max(max(data_by_frame)) + 1) # 固定Y轴范围

# 创建动画
ani = animation.FuncAnimation(
    fig=fig,
    func=update,
    frames=len(data_by_frame), # 总帧数
    interval=500               # 帧间隔时间（毫秒）
)

plt.show()