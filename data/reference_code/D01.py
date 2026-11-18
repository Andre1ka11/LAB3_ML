import json
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale"))
cv = StratifiedKFold(3, shuffle=True, random_state=42)
search = GridSearchCV(pipe, {"svc__C": [0.1, 1, 10]}, scoring="balanced_accuracy", cv=cv, n_jobs=1)
search.fit(Xtr, ytr)
pred = search.predict(Xte)
print(json.dumps({"best_C": search.best_params_["svc__C"], "balanced_accuracy": balanced_accuracy_score(yte, pred)}))
