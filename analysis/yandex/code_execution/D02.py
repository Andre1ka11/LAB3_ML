import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score

# Создание датасета
X, y = make_classification(n_samples=500, n_features=12, n_informative=6, n_redundant=2, weights=[0.85, 0.15], random_state=42)

# Введение пропусков
rng = np.random.default_rng(42)
X[rng.random(X.shape) < 0.1] = np.nan

# Разделение на train и test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Создание и обучение пайплайна
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42))
])

pipeline.fit(X_train, y_train)

# Предсказания и метрики
y_pred = pipeline.predict(X_test)
f1 = f1_score(y_test, y_pred, pos_label=1)
balanced_acc = balanced_accuracy_score(y_test, y_pred)

# Вывод результатов
print(json.dumps({"f1": f1, "balanced_accuracy": balanced_acc}))