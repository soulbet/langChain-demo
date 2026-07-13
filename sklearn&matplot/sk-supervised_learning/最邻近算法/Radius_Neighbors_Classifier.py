"""

分类器在给定半径内的临近点之间进行投票

"""
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
from sklearn.neighbors import RadiusNeighborsClassifier
neigh = RadiusNeighborsClassifier(radius=1.0)
neigh.fit(X, y)
print(neigh.predict_proba([[1.0]]))