from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# Загрузка данных
data = load_iris()
X = data.data
y = data.target

# Разделение данных на train и test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Определение p-метрики как евклидова (Minkowski с p=2)
param_grid = {'knn__n_neighbors': [3, 5, 7, 9]}
pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(weights='uniform', metric='minkowski', p=2))])

# Поиск лучшего k через GridSearchCV
grid_search = GridSearchCV(pipe, param_grid, scoring='accuracy', cv=StratifiedKFold(3, shuffle=True, random_state=42), n_jobs=1)
grid_search.fit(X_train, y_train)

# Получение лучшей точности и значения k
best_k = grid_search.best_params_['knn__n_neighbors']
test_accuracy = grid_search.score(X_test, y_test)

# Форматированный вывод результата
result = {
    'best_k': best_k,
    'test_accuracy': test_accuracy
}
print(result)
