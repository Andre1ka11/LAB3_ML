import json
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(Xtr, ytr)
pred = model.predict(Xte)
print(json.dumps({"rmse": root_mean_squared_error(yte, pred), "r2": r2_score(yte, pred)}))
