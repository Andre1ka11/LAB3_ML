import json
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X, y = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model = make_pipeline(StandardScaler(), PCA(n_components=2, svd_solver="full"), LogisticRegression(max_iter=2000, random_state=42))
model.fit(Xtr, ytr)
print(json.dumps({"accuracy": accuracy_score(yte, model.predict(Xte))}))
