import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

A = np.array([[0, 0]])
B = np.array([[4, 3]])

print(f"欧氏距离:   {euclidean_distances(A, B)[0][0]:.2f}")   # 5.00
print(f"曼哈顿距离: {manhattan_distances(A, B)[0][0]}")       # 7

# 可视化曼哈顿路径
def manhattan_path(a, b):
    path = []
    x, y = a
    target_x, target_y = b
    # 先水平移动
    while x != target_x:
        x += 1 if x < target_x else -1
        path.append((x, y))
    # 再垂直移动
    while y != target_y:
        y += 1 if y < target_y else -1
        path.append((x, y))
    return path

print("曼哈顿路径:", manhattan_path((0,0), (4,3)))
# 输出: [(1,0), (2,0), (3,0), (4,0), (4,1), (4,2), (4,3)] 共7步