import json
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', gamma='scale'))
])

param_grid = {'svc__C': [0.1, 1, 10]}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring='balanced_accuracy',
    cv=cv,
    n_jobs=1
)

grid.fit(X_train, y_train)
best_C = grid.best_params_['svc__C']
y_pred = grid.predict(X_test)
test_ba = balanced_accuracy_score(y_test, y_pred)

print(json.dumps({'best_C': best_C, 'balanced_accuracy': test_ba}))