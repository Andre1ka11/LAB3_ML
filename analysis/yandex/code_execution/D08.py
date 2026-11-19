from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score, f1_score
import json

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

model = GaussianNB(var_smoothing=1e-9)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

balanced_acc = balanced_accuracy_score(y_test, y_pred)
f1_malignant = f1_score(y_test, y_pred, pos_label=0)

results = {
    "balanced_accuracy": balanced_acc,
    "f1_malignant": f1_malignant
}

print(json.dumps(results, indent=4))