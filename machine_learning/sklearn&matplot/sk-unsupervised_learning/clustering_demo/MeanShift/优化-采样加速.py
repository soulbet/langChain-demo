from sklearn.utils import resample

# 随机采样到 5000 以内
if len(X) > 5000:
    X = resample(X, n_samples=5000, random_state=42)