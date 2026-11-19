import json
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

X, y = make_classification(
    n_samples=600,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    weights=[0.9, 0.1],
    class_sep=1.0,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    class_weight='balanced',
    random_state=42,
    n_jobs=1,
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

result = {
    'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred, pos_label=1),
}
print(json.dumps(result))