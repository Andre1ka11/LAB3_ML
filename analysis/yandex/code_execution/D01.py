from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', gamma='scale'))
])

param_grid = {'svc__C': [0.1, 1, 10]}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='balanced_accuracy',
    cv=StratifiedKFold(3, shuffle=True, random_state=42),
    n_jobs=1
)

grid_search.fit(X_train, y_train)

best_C = grid_search.best_params_['svc__C']
y_pred = grid_search.predict(X_test)
balanced_acc = balanced_accuracy_score(y_test, y_pred)

print({
    'best_C': best_C,
    'balanced_accuracy': balanced_acc
})