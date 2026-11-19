import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Загрузка данных
iris = load_iris()
y = iris.target
X = iris.data

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Создание pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier(weights='uniform', metric='minkowski', p=2))
])

# Параметры для GridSearchCV
param_grid = {'classifier__n_neighbors': [3, 5, 7, 9]}

# Настройка GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='accuracy',
    cv=StratifiedKFold(3, shuffle=True, random_state=42),
    n_jobs=1
)

# Фитование модели
grid_search.fit(X_train, y_train)

# Получение лучшего параметра и тестовой точности
best_k = grid_search.best_params_['classifier__n_neighbors']
test_accuracy = grid_search.score(X_test, y_test)

# Формирование JSON
json_result = {
    "best_k": best_k,
    "test_accuracy": test_accuracy
}

json_result