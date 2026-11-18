import json
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score
X, y = make_classification(n_samples=500, n_features=12, n_informative=6, n_redundant=2, weights=[0.85, 0.15], random_state=42)
rng = np.random.default_rng(42)
X[rng.random(X.shape) < 0.1] = np.nan
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42))
model.fit(Xtr, ytr)
pred = model.predict(Xte)
print(json.dumps({"f1": f1_score(yte, pred, pos_label=1), "balanced_accuracy": balanced_accuracy_score(yte, pred)}))
