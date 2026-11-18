import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(weights="uniform", metric="minkowski", p=2))
cv = StratifiedKFold(3, shuffle=True, random_state=42)
search = GridSearchCV(pipe, {"kneighborsclassifier__n_neighbors": [3, 5, 7, 9]}, scoring="accuracy", cv=cv, n_jobs=1)
search.fit(Xtr, ytr)
print(json.dumps({"best_k": search.best_params_["kneighborsclassifier__n_neighbors"], "accuracy": accuracy_score(yte, search.predict(Xte))}))
