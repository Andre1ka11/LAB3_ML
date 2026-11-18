import json
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
X, _ = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=0.8, random_state=42)
Xs = StandardScaler().fit_transform(X)
labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(Xs)
print(json.dumps({"silhouette": silhouette_score(Xs, labels)}))
