from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.datasets import make_classification

# 造数据
X, y = make_classification(n_samples=1000, random_state=42)

# 基础模型（SVM 概率很不准）
base_model = SVC()

# 校准！
# method='sigmoid' → Platt
# method='isotonic' → 保序回归
calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)

# 训练
calibrated_model.fit(X, y)

# 输出校准后的真实概率
print(calibrated_model.predict_proba(X[:5]))