import torch
import torch.nn.functional as F

# X 是嵌入层输出的向量矩阵 [4, 6]
# X = torch.randn(4, 6)
X = torch.tensor([[ 1.7630, -0.0658,  0.4497,  0.3638,  1.5196, -0.5965],
        [-0.5063,  0.7680,  0.4680, -0.3725, -0.7083,  1.8805],
        [-1.0673, -0.4066,  0.1193, -0.6219,  1.2520, -0.1127],
        [-0.8689,  0.8315,  1.3764,  1.3732, -0.6091,  0.4928]])
print(f"X:{X}")
# --- 第1步：计算“相关性分数矩阵” ---
# 用矩阵乘法，一次性算出所有词对之间的点积
# X 乘以 X的转置 -> [4, 6] @ [6, 4] = [4, 4]
attn_scores = torch.matmul(X, X.T)

print("注意力分数矩阵:")
print(attn_scores)
# 输出解释：
# attn_scores[i][j] 表示 词i 和 词j 的相关性。
# 对角线是词和自己的分数，最高，因为向量和自己最相似。

# --- 第2步：用Softmax把分数变成权重 ---
# dim=-1 表示对每一行做softmax
attn_weights = F.softmax(attn_scores, dim=-1)

print("注意力权重矩阵:")
print(attn_weights)
# 现在每一行的数字都是正数，加起来等于1。
# 比如处理“吃”的这一行，它会给“苹果”一个很大的权重。

# --- 第3步：加权求和，得到输出 ---
# attn_weights @ X -> [4, 4] @ [4, 6] = [4, 6]
output = torch.matmul(attn_weights, X)

print("注意力的最终输出矩阵:")
print(output.shape) # torch.Size([4, 6])
# 这个新的“吃”的向量（output[2]），已经包含了“苹果”的信息。