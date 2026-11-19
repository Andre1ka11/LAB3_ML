import json
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Загрузка данных
X, y = load_diabetes(return_X_y=True)

# Разбиение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Построение Pipeline
model = Pipeline([('scaler', StandardScaler()), ('linear_regression', LinearRegression())])

# Обучение модели
model.fit(X_train, y_train)

# Прогнозирование на тестовой выборке
y_pred = model.predict(X_test)

# Вычисление метрик
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# Форматируем результат в JSON
result = {
    'rmse': rmse,
    'r2': r2
}
print(json.dumps(result))
