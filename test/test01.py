dp=[0.0, -float('inf'), -float('inf')]
prob=-2.2246235515243336
back=[None, None, None]
if dp[0] + prob > dp[1]:
    print(111)
    dp[1] = dp[0] + prob
    back[1] = 0
print(back)