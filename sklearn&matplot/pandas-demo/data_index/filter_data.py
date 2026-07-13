
"""
1、单条件过滤
2、多条件过滤
3、字符串过滤
4、query，链式友好
5、isin()
6、between
7、filter
8、dropna () 和 fillna ()
9、删除指定列
10、duplicated()去重
11、nlargest () 和 nsmallest () 筛选极值

"""
import pandas as pd
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': [100, 200, 300, 400, 500]
}, index=['a', 'b', 'c', 'd', 'e'])
## 1、单条件过滤
# 筛选 A 列大于 10 的行
df_filtered = df[df['A'] > 10]

# 筛选 A 列等于某个值
df_filtered = df[df['A'] == 5]

# 筛选 A 列不等于某个值
df_filtered = df[df['A'] != 5]

## 2、多条件过滤
# 且（&）：同时满足
df_filtered = df[(df['A'] > 10) & (df['B'] < 50)]

# 或（|）：满足任一条件
df_filtered = df[(df['A'] > 10) | (df['B'] < 50)]

# 非（~）：取反
df_filtered = df[~(df['A'] > 10)]

## 3、字符串过滤
# 包含某字符
df_filtered = df[df['name'].str.contains('详情')]

# 以某字符开头
df_filtered = df[df['name'].str.startswith('测试')]

# 以某字符结尾
df_filtered = df[df['name'].str.endswith('.txt')]

# 匹配正则表达式
df_filtered = df[df['name'].str.contains(r'\d+')]

# 忽略大小写
df_filtered = df[df['name'].str.contains('abc', case=False)]

## 4、query，链式友好
# 基本用法
df_filtered = df.query('A > 10')

# 多条件
df_filtered = df.query('A > 10 and B < 50')

# 使用变量（@ 符号）
threshold = 10
df_filtered = df.query('A > @threshold')

# 包含字符串
df_filtered = df.query('name.str.contains("详情")')

## 5、isin()
# 筛选 A 列在列表中的行
df_filtered = df[df['A'].isin([1, 3, 5])]

# 筛选 A 列不在列表中的行
df_filtered = df[~df['A'].isin([1, 3, 5])]

# 多列同时判断（需要组合条件）
df_filtered = df[df['A'].isin([1, 2]) & df['B'].isin([10, 20])]

## 6、between
# 筛选 A 列在 10 到 50 之间（包含边界）
df_filtered = df[df['A'].between(10, 50)]

# 等价于
df_filtered = df[(df['A'] >= 10) & (df['A'] <= 50)]

## 7、filter
# 筛选列名包含某字符的列
df_filtered = df.filter(like='日期')

# 筛选列名匹配正则表达式
df_filtered = df.filter(regex=r'col_\d+')

# 筛选指定列名的列
df_filtered = df.filter(items=['A', 'C', 'E'])

## 8、dropna()和fillna()
# 删除包含 NaN 的行
df_filtered = df.dropna()

# 删除全部为 NaN 的行
df_filtered = df.dropna(how='all')

# 删除某列有 NaN 的行
df_filtered = df.dropna(subset=['A', 'B'])

# 填充 NaN 后再筛选
df_filtered = df.fillna(0)[df['A'] > 10]

## 9、删除指定列
# 删除指定行标签
df_filtered = df.drop(['row1', 'row2'])

# 删除指定列
df_filtered = df.drop(['A', 'B'], axis=1)

# 删除行位置（按索引）
df_filtered = df.drop(df.index[0:5])  # 删除前 5 行

## 10、duplicated()去重
# 保留第一次出现，删除重复行
df_filtered = df.drop_duplicates()

# 按指定列去重
df_filtered = df.drop_duplicates(subset=['A', 'B'])

# 保留最后一次出现
df_filtered = df.drop_duplicates(keep='last')

# 标记重复行
duplicate_mask = df.duplicated()
df_filtered = df[~duplicate_mask]

## 11、nlargest() 和 nsmallest() 筛选极值
# 筛选 A 列最大的 10 行
df_filtered = df.nlargest(10, 'A')

# 筛选 A 列最小的 5 行
df_filtered = df.nsmallest(5, 'A')