from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# Загрузка датасета
data = load_wine()
X = data.data
y = data.target

# Стратифицированное разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Построение пайплайна
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2, svd_solver='full')),
    ('logreg', LogisticRegression(max_iter=2000, random_state=42))
])

# Обучение только на тренировочной выборке
pipeline.fit(X_train, y_train)

# Вычисление точности на тестовой выборке
test_accuracy = pipeline.score(X_test, y_test)

# Печать результата
print(json.dumps({"accuracy": test_accuracy}))
