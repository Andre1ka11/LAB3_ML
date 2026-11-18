import json
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
X, y = make_classification(n_samples=600, n_features=10, n_informative=5, n_redundant=2, weights=[0.9, 0.1], class_sep=1.0, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42, n_jobs=1)
model.fit(Xtr, ytr)
pred = model.predict(Xte)
print(json.dumps({"balanced_accuracy": balanced_accuracy_score(yte, pred), "f1": f1_score(yte, pred, pos_label=1)}))
