from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import json

# Загрузка данных
X, y = load_diabetes(return_X_y=True)

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)

# Создание и обучение модели
model = Pipeline([('scaler', StandardScaler()), ('regressor', LinearRegression())])
model.fit(X_train, y_train)

# Предсказания на тестовой выборке
y_pred = model.predict(X_test)

# Расчет метрик
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Вывод результатов в формате JSON
results = {'rmse': rmse, 'r2': r2}
print(json.dumps(results, indent=4))