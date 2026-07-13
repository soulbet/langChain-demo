import pandas as pd
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': [100, 200, 300, 400, 500]
}, index=['a', 'b', 'c', 'd', 'e'])

for index, row in df.iterrows():
    print(index)
    print(row)

# 比iterrows()快得多，并且在大多数情况下比迭代DataFrame的值更可取
for row in df.itertuples():
    print(row) # Pandas(Index='a', A=1, B=10, C=100)
    print(row[0]) # 当前行的index
    print(row[1]) # # 当前行的A