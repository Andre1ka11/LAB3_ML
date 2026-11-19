from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

# создание синтетических данных
X, y = make_classification(n_samples=500, n_features=12, n_informative=6,
                           n_redundant=2, weights=[0.85, 0.15],
                           random_state=42)

# добавление случайных пропусков
rng = np.random.default_rng(42)
X[rng.random(X.shape) < 0.1] = np.nan

# разделение данных на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

# pipeline обучения
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(class_weight='balanced',
                                  max_iter=2000, random_state=42))
])

# обучение только на train
pipeline.fit(X_train, y_train)

# оценка на тесте
y_pred = pipeline.predict(X_test)

# расчет метрик
from sklearn.metrics import f1_score, balanced_accuracy_score

print({
    'f1': f1_score(y_test, y_pred, pos_label=1),
    'balanced_accuracy': balanced_accuracy_score(y_test, y_pred)
})
