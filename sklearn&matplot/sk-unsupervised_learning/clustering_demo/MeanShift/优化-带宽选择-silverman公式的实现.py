import numpy as np

def silverman_bandwidth(X):
    """Silverman's rule of thumb 带宽估计"""
    n, d = X.shape
    sigma = np.mean(np.std(X, axis=0))
    bandwidth = ((4 / (d + 2)) ** (1 / (d + 4))) * sigma * (n ** (-1 / (d + 4)))
    return bandwidth

# 使用
bw = silverman_bandwidth(X)
print(f"Silverman 带宽: {bw:.3f}")