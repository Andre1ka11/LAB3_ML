from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# создание синтетических данных
X, _ = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=0.8, random_state=42)

# стандартная нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# выполнение кластеризации
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# расчет показателя silhouette
silhouette = silhouette_score(X_scaled, clusters)

# вывод результата в формате json
result = {
    'silhouette': silhouette
}
print(result)
