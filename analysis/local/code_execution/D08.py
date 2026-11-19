import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Load dataset
breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Train model
model = GaussianNB(var_smoothing=1e-9)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate metrics
from sklearn.metrics import balanced_accuracy_score, f1_score

balanced_acc = balanced_accuracy_score(y_test, y_pred)
f1_malignant = f1_score(y_test, y_pred, pos_label=0)

# Output JSON
import json

json_output = json.dumps({
    "balanced_accuracy": balanced_acc,
    "f1_malignant": f1_malignant
})

print(json_output)