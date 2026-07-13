"""
loc = label-based（标签）：用名字，包含结尾
iloc = integer-location（位置）：用数字，不含结尾
"""

import pandas as pd

df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': [100, 200, 300, 400, 500]
}, index=['a', 'b', 'c', 'd', 'e'])

print("原始数据：")
print(df)
# 选取行标签 'b' 到 'd'（包含 'd'），列标签 'B' 到 'C'（包含 'C'）
print(df.loc['b':'d', 'B':'C'])
# 选取第 1 行到第 3 行（不包含第 4 行），第 1 列到第 2 列（不包含第 3 列）
print(df.iloc[1:4, 1:3])