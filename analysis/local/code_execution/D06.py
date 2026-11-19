import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
X, y = load_diabetes(return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Predict on test set
y_pred = pipeline.predict(X_test)

# Calculate RMSE and R2
mse = np.mean((y_test - y_pred)**2)
rmse = np.sqrt(mse)
r2 = np.corrcoef(y_test, y_pred)[0, 1]

# Output JSON
import json
print(json.dumps({
    "rmse": rmse,
    "r2": r2
}, separators=','))