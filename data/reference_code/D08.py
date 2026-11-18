import json
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score, f1_score
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model = GaussianNB(var_smoothing=1e-9)
model.fit(Xtr, ytr)
pred = model.predict(Xte)
print(json.dumps({"balanced_accuracy": balanced_accuracy_score(yte, pred), "f1_malignant": f1_score(yte, pred, pos_label=0)}))
