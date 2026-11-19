from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

# создание синтетических данных
X, y = make_classification(n_samples=600, n_features=10, n_informative=5,
                           n_redundant=2, weights=[0.9, 0.1],
                           class_sep=1.0, random_state=42)

# разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=42, stratify=y)

# обучение модели
clf = RandomForestClassifier(n_estimators=200, max_depth=6, 
                             class_weight='balanced', random_state=42, n_jobs=1)
clf.fit(X_train, y_train)

# предсказание классов
y_pred = clf.predict(X_test)

# вычисление метрик
bal_acc = balanced_accuracy_score(y_test, y_pred)
f1_class_1 = f1_score(y_test, y_pred, pos_label=1)

print('{"balance_accuracy": %.2f, "f1": %.2f}' % (bal_acc, f1_class_1))
