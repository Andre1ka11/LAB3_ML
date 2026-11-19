from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Загрузка данных
data = load_wine()
X, y = data.data, data.target

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Создание пайплайна
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2, svd_solver='full')),
    ('logreg', LogisticRegression(max_iter=2000, random_state=42))
])

# Обучение модели
pipeline.fit(X_train, y_train)

# Предсказание и расчёт точности
y_pred = pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# Вывод результата в формате JSON
import json
print(json.dumps({"test_accuracy": test_accuracy}))