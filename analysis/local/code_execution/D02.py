import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Generate data
X, y = make_classification(n_samples=500, n_features=12, n_informative=6, n_redundant=2, weights=[0.85, 0.15], random_state=42)

# Introduce NaNs
rng = np.random.default_rng(42)
X[rng.random(X.shape) < 0.1] = np.nan

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Define pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42))
])

# Train model
pipeline.fit(X_train, y_train)

# Predict on test set
y_pred = pipeline.predict(X_test)

# Calculate metrics
from sklearn.metrics import f1_score, balanced_accuracy_score
f1 = f1_score(y_test, y_pred, pos_label=1)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

# Output result
print(f'{{"f1": {f1}, "balanced_accuracy": {balanced_accuracy}}}')