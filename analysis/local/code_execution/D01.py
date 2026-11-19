import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Загрузка данных
breast_cancer = load_breast_cancer()
x, y = breast_cancer.data, breast_cancer.target

# Разделение данных
x_train, x_test, y_train, y y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)

# Создание pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', gamma='scale'))
])

# Параметры для GridSearchCV
param_grid = {
    'svc__C': [0.1, 1, 10]
}

# Настройка GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='balanced_accuracy',
    cv=StratifiedKFold(3, shuffle=True, random_state=42),
    n_jobs=1
)

# Проведение поиска
grid_search.fit(x_train, y_train)

# Получение лучшего C и balanced_accuracy на тестовых данных
best_C = grid_search.best_params_['svc__C']
test_score = grid_search.score(x_test, y_test)

# Вывод результата
print(f'{{"best_C": {best_C}, "balanced_accuracy": {test_score}}}')