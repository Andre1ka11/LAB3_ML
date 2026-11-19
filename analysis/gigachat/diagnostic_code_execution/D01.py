import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Загрузка набора данных breast cancer
data = datasets.load_breast_cancer()
X = data.data
y = data.target

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Определение pipeline и параметров для GridSearchCV
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', gamma='scale'))
])
parameters = {'svc__C': [0.1, 1, 10]}

# Инициализация и выполнение GridSearchCV
grid_search = GridSearchCV(pipeline, parameters, scoring='balanced_accuracy', cv=StratifiedKFold(3, shuffle=True, random_state=42), n_jobs=1)
grid_search.fit(X_train, y_train)

# Оценка модели на тестовой выборке
balanced_acc = grid_search.best_estimator_.fit(X_train, y_train).score(X_test, y_test)

# Печать результата
print({"best_C": grid_search.best_params_['svc__C'], "balanced_accuracy": balanced_acc})
