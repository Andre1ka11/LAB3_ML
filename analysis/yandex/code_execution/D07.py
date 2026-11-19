from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json

# Генерация данных
X, y = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=0.8, random_state=42)

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Кластеризация
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Расчет silhouette score
silhouette = silhouette_score(X_scaled, clusters)

# Вывод результата в формате JSON
result = {"silhouette_score": silhouette}
print(json.dumps(result))