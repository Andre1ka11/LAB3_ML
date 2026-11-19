import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(weights="uniform", metric="minkowski", p=2))
])

param_grid = {"knn__n_neighbors": [3, 5, 7, 9]}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=1
)
grid.fit(X_train, y_train)

best_k = grid.best_params_["knn__n_neighbors"]
test_accuracy = grid.score(X_test, y_test)

print(json.dumps({"best_k": best_k, "test_accuracy": test_accuracy}))