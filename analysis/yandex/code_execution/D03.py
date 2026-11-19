from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import json

# Загрузка данных
iris = load_iris()
X, y = iris.data, iris.target

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Создание пайплайна
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(weights='uniform', metric='minkowski', p=2))
])

# Настройка GridSearchCV
param_grid = {'knn__n_neighbors': [3, 5, 7, 9]}
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    n_jobs=1
)

# Обучение модели
grid_search.fit(X_train, y_train)

# Получение лучшего значения n_neighbors
best_k = grid_search.best_params_['knn__n_neighbors']

# Оценка модели на тестовом наборе
test_accuracy = grid_search.score(X_test, y_test)

# Вывод результата в формате JSON
result = {
    'best_k': best_k,
    'test_accuracy': test_accuracy
}
print(json.dumps(result, indent=4))