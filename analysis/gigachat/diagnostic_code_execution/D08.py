from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score

# Загрузка набора данных
data = load_breast_cancer()
X = data.data
y = data.target

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Обучение модели Naive Bayes
gnb = GaussianNB(var_smoothing=1e-9)
gnb.fit(X_train, y_train)

# Прогнозирование на тестовой выборке
y_pred = gnb.predict(X_test)

# Вычисление сбалансированной точности и F1-мера для класса malignant
bal_acc = balanced_accuracy_score(y_test, y_pred)
f1_malignant = f1_score(y_test, y_pred, pos_label=0)

# Печать результатов в формате JSON
print('{"balanced_accuracy": %.3f, "f1_malignant": %.3f}' % (bal_acc, f1_malignant))
