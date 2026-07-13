
import numpy as np

# 推荐用法：np.log1p
# 它的真实公式是 log(1 + X)
# 这样即使 X 是 0，算出来也是 log(1) = 0，完美避开报错。
X_transformed = np.log1p(X)

# 如果以后需要还原数据（反变换）：
X_original = np.expm1(X_transformed) # 算的是 e^X - 1
