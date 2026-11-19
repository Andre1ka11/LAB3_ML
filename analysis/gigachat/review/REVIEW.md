# Материалы независимой проверки

Модель и технические ID скрыты. Для настоящей слепой оценки нужно перемешать этот пакет с ответами остальных моделей; пакет только одного участника сам по себе этого не обеспечивает. Ключ review_key.json проверяющим не передавать.

B и C: score_0_2 по рубрике. D: шесть флагов ошибок 0/1; N/A там, где критерий неприменим. Не исправлять ответ или код. Два проверяющих заполняют отдельные CSV независимо. После сравнения оценок рассчитывается Cohen kappa; при <0.6 уточнить рубрику и переоценить. Не заменять пропуски нулями.

## R001 (тип D)

Напишите полный автономный Python-код на sklearn без загрузки данных из сети. Создайте make_classification(n_samples=600,n_features=10,n_informative=5,n_redundant=2,weights=[0.9,0.1],class_sep=1.0,random_state=42). Разбейте test_size=0.25,random_state=42,stratify=y. Обучите RandomForestClassifier(n_estimators=200,max_depth=6,class_weight='balanced',random_state=42,n_jobs=1) на train. Напечатайте JSON с balanced_accuracy и f1 для класса 1 на test. Все неуказанные параметры оставьте по умолчанию. Код должен запускаться без исправлений.

Ответ модели:

{"answer": {"balance_accuracy": 0.87, "f1": 0.83}, "explanation": "ответ содержит расчет balanced accuracy и f1 score для класса 1 после обучения модели на случайных данных", "confidence": 80}

```python
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
```

Эталон и рубрика:

```json
{
  "reference": {
    "answer": {
      "code_path": "data/reference_code/D04.py",
      "metrics": {
        "balanced_accuracy": 0.7425373134328358,
        "f1": 0.6153846153846154
      }
    },
    "solution_ru": "Эталонный код и фактические метрики получаются локальным запуском; см. reference.code_path и results/reference/code_metrics.json."
  },
  "rubric": {
    "pass_at_1": "Запуск извлечённого кода без правок с первой попытки (JSON-декодирование строки answer не является правкой).",
    "correctness": "Все метрики отличаются от эталона не более чем на 0.01 абсолютных; best_C/best_k должны совпасть точно.",
    "methodology": [
      "fit preprocessing только на train и внутри CV",
      "подбор только по train/CV",
      "оценка на отдельном test, кроме описательной D07",
      "правильная метрика и pos_label",
      "стратификация для классификации",
      "фиксированные random_state там, где применимо"
    ],
    "manual_review": "Два независимых слепых проверяющих; для каждой методологической ошибки 0/1, затем Cohen kappa. Если kappa<0.6, уточнить рубрику и переоценить. Это не оценка LLM самим эталонным кодом."
  }
}
```

## R002 (тип C)

Есть 25 000 независимых строк по деталям: 15 числовых и 4 категориальных признака, местами пропуски; целевая переменная - наличие дефекта. Связи предположительно нелинейны. Предложите сильный табличный baseline и честное сравнение с простой моделью.

Ответ модели:

{"answer": "[\"xgbregressor\", \"logisticregression\"]", "explanation": "предлагается использовать XGBoost для регрессии и логистическую регрессию для бинарной классификации как простые базовые модели для сравнения.", "confidence": 80}

Эталон и рубрика:

```json
{
  "reference": {
    "answer": "Градиентный бустинг по деревьям с корректной обработкой категорий/пропусков; допустим random forest. Сравнить с логистической регрессией в Pipeline на одном фиксированном стратифицированном разбиении и одной выбранной по балансу классов метрике. Подбор только по train/CV, проверить доступность признаков в момент прогноза.",
    "solution_ru": "Градиентный бустинг по деревьям с корректной обработкой категорий/пропусков; допустим random forest. Сравнить с логистической регрессией в Pipeline на одном фиксированном стратифицированном разбиении и одной выбранной по балансу классов метрике. Подбор только по train/CV, проверить доступность признаков в момент прогноза."
  },
  "rubric": {
    "0": "Непригодное решение или критическая методологическая ошибка.",
    "1": "Основной метод разумен, но один или несколько обязательных аспектов существенно не раскрыты.",
    "2": "Обоснованное решение раскрывает все три критерия без критических ошибок.",
    "criteria": [
      "нелинейная табличная модель",
      "корректная обработка категорий и пропусков",
      "единое разбиение/метрика и отсутствие утечки"
    ],
    "accepted_alternatives": "CatBoost допустим концептуально, но не требуется для запуска эталонного кода банка."
  }
}
```

## R003 (тип D)

Напишите полный автономный Python-код на sklearn без загрузки данных из сети. На load_iris отложите test_size=0.25 с random_state=42 и stratify=y. В Pipeline(StandardScaler(),KNeighborsClassifier(weights='uniform',metric='minkowski',p=2)) выберите n_neighbors из [3,5,7,9] через GridSearchCV(scoring='accuracy',cv=StratifiedKFold(3,shuffle=True,random_state=42),n_jobs=1) только на train. Напечатайте JSON с best_k и test accuracy. Все неуказанные параметры оставьте по умолчанию. Код должен запускаться без исправлений.

Ответ модели:

{"answer": {"best_k": 5, "test_accuracy": 0.8666666666666667}, "explanation": "Создан полный независимый скрипт с использованием sklearn, который выполняет задачу пошагово. Сначала создается pipeline со стандартным масштабированием и классификатором KNN. Затем выполняется поиск наилучшего параметра n_neighbors через кросс-валидацию. Наконец, тестовая точность вычисляется после обучения на обучающей выборке.", "confidence": 100}

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# Загрузка данных
data = load_iris()
X = data.data
y = data.target

# Разделение данных на train и test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Определение p-метрики как евклидова (Minkowski с p=2)
param_grid = {'knn__n_neighbors': [3, 5, 7, 9]}
pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(weights='uniform', metric='minkowski', p=2))])

# Поиск лучшего k через GridSearchCV
grid_search = GridSearchCV(pipe, param_grid, scoring='accuracy', cv=StratifiedKFold(3, shuffle=True, random_state=42), n_jobs=1)
grid_search.fit(X_train, y_train)

# Получение лучшей точности и значения k
best_k = grid_search.best_params_['knn__n_neighbors']
test_accuracy = grid_search.score(X_test, y_test)

# Форматированный вывод результата
result = {
    'best_k': best_k,
    'test_accuracy': test_accuracy
}
print(result)
```

Эталон и рубрика:

```json
{
  "reference": {
    "answer": {
      "code_path": "data/reference_code/D03.py",
      "metrics": {
        "best_k": 7,
        "accuracy": 0.9473684210526315
      }
    },
    "solution_ru": "Эталонный код и фактические метрики получаются локальным запуском; см. reference.code_path и results/reference/code_metrics.json."
  },
  "rubric": {
    "pass_at_1": "Запуск извлечённого кода без правок с первой попытки (JSON-декодирование строки answer не является правкой).",
    "correctness": "Все метрики отличаются от эталона не более чем на 0.01 абсолютных; best_C/best_k должны совпасть точно.",
    "methodology": [
      "fit preprocessing только на train и внутри CV",
      "подбор только по train/CV",
      "оценка на отдельном test, кроме описательной D07",
      "правильная метрика и pos_label",
      "стратификация для классификации",
      "фиксированные random_state там, где применимо"
    ],
    "manual_review": "Два независимых слепых проверяющих; для каждой методологической ошибки 0/1, затем Cohen kappa. Если kappa<0.6, уточнить рубрику и переоценить. Это не оценка LLM самим эталонным кодом."
  }
}
```

## R004 (тип D)

Напишите полный автономный Python-код на sklearn без загрузки данных из сети. Загрузите load_breast_cancer. Сохраните исходную кодировку y: 0=malignant,1=benign. Сделайте train_test_split(test_size=0.25,random_state=42,stratify=y). Обучите GaussianNB(var_smoothing=1e-9) на train без масштабирования. На test напечатайте JSON с balanced_accuracy и f1_malignant (pos_label=0). Не переобозначайте классы. Все неуказанные параметры оставьте по умолчанию. Код должен запускаться без исправлений.

Ответ модели:

{"answer": {"balanced_accuracy": 0.963, "f1_malignant": 0.978}, "explanation": "Реализован полный код с использованием sklearn, который выполняет все требуемые шаги.", "confidence": 100}

```
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
```

Эталон и рубрика:

```json
{
  "reference": {
    "answer": {
      "code_path": "data/reference_code/D08.py",
      "metrics": {
        "balanced_accuracy": 0.9228511530398322,
        "f1_malignant": 0.9108910891089109
      }
    },
    "solution_ru": "Эталонный код и фактические метрики получаются локальным запуском; см. reference.code_path и results/reference/code_metrics.json."
  },
  "rubric": {
    "pass_at_1": "Запуск извлечённого кода без правок с первой попытки (JSON-декодирование строки answer не является правкой).",
    "correctness": "Все метрики отличаются от эталона не более чем на 0.01 абсолютных; best_C/best_k должны совпасть точно.",
    "methodology": [
      "fit preprocessing только на train и внутри CV",
      "подбор только по train/CV",
      "оценка на отдельном test, кроме описательной D07",
      "правильная метрика и pos_label",
      "стратификация для классификации",
      "фиксированные random_state там, где применимо"
    ],
    "manual_review": "Два независимых слепых проверяющих; для каждой методологической ошибки 0/1, затем Cohen kappa. Если kappa<0.6, уточнить рубрику и переоценить. Это не оценка LLM самим эталонным кодом."
  }
}
```

## R005 (тип D)

Напишите полный автономный Python-код на sklearn без загрузки данных из сети. Создайте make_classification(n_samples=500,n_features=12,n_informative=6,n_redundant=2,weights=[0.85,0.15],random_state=42). Затем rng=np.random.default_rng(42); X[rng.random(X.shape)<0.1]=np.nan. Разбейте test_size=0.25,random_state=42,stratify=y. Обучите только на train Pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=2000,random_state=42)). На test напечатайте JSON с f1 (положительный класс 1) и balanced_accuracy. Не используйте test для fit. Все неуказанные параметры оставьте по умолчанию. Код должен запускаться без исправлений.

Ответ модели:

{"answer": {"f1": 0.73, "balanced_accuracy": 0.89}, "explanation": "создал классификационные данные, применил случайные пропуски, разбил данные на train/test, провел обучение и оценку модели на тестовой выборке.", "confidence": 80} 

```python
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

# создание синтетических данных
X, y = make_classification(n_samples=500, n_features=12, n_informative=6,
                           n_redundant=2, weights=[0.85, 0.15],
                           random_state=42)

# добавление случайных пропусков
rng = np.random.default_rng(42)
X[rng.random(X.shape) < 0.1] = np.nan

# разделение данных на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

# pipeline обучения
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(class_weight='balanced',
                                  max_iter=2000, random_state=42))
])

# обучение только на train
pipeline.fit(X_train, y_train)

# оценка на тесте
y_pred = pipeline.predict(X_test)

# расчет метрик
from sklearn.metrics import f1_score, balanced_accuracy_score

print({
    'f1': f1_score(y_test, y_pred, pos_label=1),
    'balanced_accuracy': balanced_accuracy_score(y_test, y_pred)
})
```

Эталон и рубрика:

```json
{
  "reference": {
    "answer": {
      "code_path": "data/reference_code/D02.py",
      "metrics": {
        "f1": 0.52,
        "balanced_accuracy": 0.7571996027805363
      }
    },
    "solution_ru": "Эталонный код и фактические метрики получаются локальным запуском; см. reference.code_path и results/reference/code_metrics.json."
  },
  "rubric": {
    "pass_at_1": "Запуск извлечённого кода без правок с первой попытки (JSON-декодирование строки answer не является правкой).",
    "correctness": "Все метрики отличаются от эталона не более чем на 0.01 абсолютных; best_C/best_k должны совпасть точно.",
    "methodology": [
      "fit preprocessing только на train и внутри CV",
      "подбор только по train/CV",
      "оценка на отдельном test, кроме описательной D07",
      "правильная метрика и pos_label",
      "стратификация для классификации",
      "фиксированные random_state там, где применимо"
    ],
    "manual_review": "Два независимых слепых проверяющих; для каждой методологической ошибки 0/1, затем Cohen kappa. Если kappa<0.6, уточнить рубрику и переоценить. Это не оценка LLM самим эталонным кодом."
  }
}
```

## R006 (тип D)

Напишите полный автономный Python-код на sklearn без загрузки данных из сети. На load_breast_cancer сделайте train_test_split(test_size=0.25, random_state=42, stratify=y). В Pipeline StandardScaler -> SVC(kernel='rbf', gamma='scale') подберите C из [0.1,1,10] через GridSearchCV(scoring='balanced_accuracy', cv=StratifiedKFold(3, shuffle=True, random_state=42), n_jobs=1). Поиск видит только train. На test один раз вычислите balanced_accuracy. Напечатайте JSON с best_C и balanced_accuracy. Все неуказанные параметры оставьте по умолчанию. Код должен запускаться без исправлений.

Ответ модели:

{"answer": {"best_C": 1, "balanced_accuracy": 0.96}, "explanation": "реализован полный независимый код с использованием sklearn, который выполняет все требуемые шаги и возвращает результат", "confidence": 100}

```
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Загрузка набора данных breast cancer
data = datasets.load_breast_cancer()
X = data.data
y = data.target

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Определение pipeline и параметров для GridSearchCV
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', gamma='scale'))
])
parameters = {'svc__C': [0.1, 1, 10]}

# Инициализация и выполнение GridSearchCV
grid_search = GridSearchCV(pipeline, parameters, scoring='balanced_accuracy', cv=StratifiedKFold(3, shuffle=True, random_state=42), n_jobs=1)
grid_search.fit(X_train, y_train)

# Оценка модели на тестовой выборке
balanced_acc = grid_search.best_estimator_.fit(X_train, y_train).score(X_test, y_test)

# Печать результата
print({"best_C": grid_search.best_params_['svc__C'], "balanced_accuracy": balanced_acc})
```

Эталон и рубрика:

```json
{
  "reference": {
    "answer": {
      "code_path": "data/reference_code/D01.py",
      "metrics": {
        "best_C": 10,
        "balanced_accuracy": 0.9794549266247379
      }
    },
    "solution_ru": "Эталонный код и фактические метрики получаются локальным запуском; см. reference.code_path и results/reference/code_metrics.json."
  },
  "rubric": {
    "pass_at_1": "Запуск извлечённого кода без правок с первой попытки (JSON-декодирование строки answer не является правкой).",
    "correctness": "Все метрики отличаются от эталона не более чем на 0.01 абсолютных; best_C/best_k должны совпасть точно.",
    "methodology": [
      "fit preprocessing только на train и внутри CV",
      "подбор только по train/CV",
      "оценка на отдельном test, кроме описательной D07",
      "правильная метрика и pos_label",
      "стратификация для классификации",
      "фиксированные random_state там, где применимо"
    ],
    "manual_review": "Два независимых слепых проверяющих; для каждой методологической ошибки 0/1, затем Cohen kappa. Если kappa<0.6, уточнить рубрику и переоценить. Это не оценка LLM самим эталонным кодом."
  }
}
```

## R007 (тип B)

Оцените утверждение: верно или неверно? Обоснуйте ответ. KMeans оптимизирует сумму квадратов расстояний до центроидов, однако стандартная итерационная процедура может остановиться в локальном минимуме.

Ответ модели:

{"answer": true, "explanation": "KMeans действительно стремится минимизировать сумму квадратов расстояний до центроидов (в евклидовом пространстве). Однако из-за случайного начального размещения центроидов алгоритм может застревать в локальных минимумах, что делает результат зависимым от начальных условий.", "confidence": 95}

Эталон и рубрика:

```json
{
  "reference": {
    "answer": true,
    "solution_ru": "Целевая функция - inertia. Начальная инициализация влияет на результат; несколько инициализаций уменьшают риск неудачного локального решения."
  },
  "rubric": {
    "0": "Вердикт неверен или отсутствует.",
    "1": "Вердикт верен, но обоснование ошибочно или существенно неполно.",
    "2": "Верный вердикт и корректное обоснование по обоим пунктам.",
    "criteria": [
      "сумма квадратов/inertia",
      "зависимость от инициализации/локальный минимум"
    ]
  }
}
```

## R008 (тип B)

Оцените утверждение: верно или неверно? Обоснуйте ответ. Случайный выбор подмножества признаков при построении деревьев случайного леса предназначен, в частности, для уменьшения корреляции между деревьями.

Ответ модели:

{"answer": true, "explanation": "Случайный выбор подмножества признаков уменьшает корреляцию между деревьями, так как каждое дерево обучается на уникальной комбинации признаков, что снижает вероятность дублирования структуры и повышает разнообразие моделей.", "confidence": 95}

Эталон и рубрика:

```json
{
  "reference": {
    "answer": true,
    "solution_ru": "Разные деревья не всегда выбирают один и тот же сильный признак. Менее коррелированные ошибки лучше усредняются; эффект не гарантирован на каждом датасете."
  },
  "rubric": {
    "0": "Вердикт неверен или отсутствует.",
    "1": "Вердикт верен, но обоснование ошибочно или существенно неполно.",
    "2": "Верный вердикт и корректное обоснование по обоим пунктам.",
    "criteria": [
      "разнообразие деревьев",
      "меньшая корреляция помогает усреднению"
    ]
  }
}
```

## R009 (тип C)

У 900 насосов измерены температура в K, вибрация в мм/с и мощность в кВт. Меток нет. Нужно найти группы похожих режимов; число групп неизвестно. Предложите метод, обработку масштаба, способ выбора числа групп и проверку инженерного смысла.

Ответ модели:

{"answer": "кластеризация k-средних", "explanation": "подходит для нахождения неизвестного числа групп", "confidence": 80}

Эталон и рубрика:

```json
{
  "reference": {
    "answer": "Стандартизация и KMeans для нескольких k; silhouette/стабильность кластеров, анализ центров в исходных единицах и обсуждение с инженером. Проверить выбросы и форму кластеров; DBSCAN/HDBSCAN допустимы с обоснованием плотности и масштабирования.",
    "solution_ru": "Стандартизация и KMeans для нескольких k; silhouette/стабильность кластеров, анализ центров в исходных единицах и обсуждение с инженером. Проверить выбросы и форму кластеров; DBSCAN/HDBSCAN допустимы с обоснованием плотности и масштабирования."
  },
  "rubric": {
    "0": "Непригодное решение или критическая методологическая ошибка.",
    "1": "Основной метод разумен, но один или несколько обязательных аспектов существенно не раскрыты.",
    "2": "Обоснованное решение раскрывает все три критерия без критических ошибок.",
    "criteria": [
      "метод без учителя с обоснованием",
      "обработка разных масштабов",
      "выбор параметров и предметная проверка; кластеры не равны автоматически классам неисправностей"
    ],
    "accepted_alternatives": "Принимаются KMeans, GMM и методы плотности, если их предпосылки и параметры объяснены."
  }
}
```

## R010 (тип B)

Оцените утверждение: верно или неверно? Обоснуйте ответ. ROC-AUC=0.95 гарантирует precision не ниже 0.95 при пороге классификации 0.5 независимо от доли положительного класса.

Ответ модели:

{"answer": false, "explanation": "ROC-AUC (площадь под кривой ROC) измеряет общую способность модели различать положительные и отрицательные классы, но не напрямую связана с точностью (precision). Даже высокий ROC-AUC не гарантирует высокую точность при конкретном пороге классификации.", "confidence": 80}

Эталон и рубрика:

```json
{
  "reference": {
    "answer": false,
    "solution_ru": "ROC-AUC описывает ранжирование по порогам; precision зависит от выбранного порога и базовой частоты. Такая гарантия отсутствует."
  },
  "rubric": {
    "0": "Вердикт неверен или отсутствует.",
    "1": "Вердикт верен, но обоснование ошибочно или существенно неполно.",
    "2": "Верный вердикт и корректное обоснование по обоим пунктам.",
    "criteria": [
      "ROC-AUC оценивает ранжирование",
      "precision зависит от порога и prevalence"
    ]
  }
}
```

## R011 (тип B)

Оцените утверждение: верно или неверно? Обоснуйте ответ. Обычный PCA использует метки классов и выбирает направления, лучше всего разделяющие классы.

Ответ модели:

{"answer": false, "explanation": "PCA (Principal Component Analysis) является методом снижения размерности и не использует метки классов. Он ищет направления, которые объясняют максимальную вариацию данных, независимо от класса.", "confidence": 100}

Эталон и рубрика:

```json
{
  "reference": {
    "answer": false,
    "solution_ru": "PCA не использует y, а максимизирует объяснённую дисперсию X. Направление наибольшей дисперсии не обязано разделять классы."
  },
  "rubric": {
    "0": "Вердикт неверен или отсутствует.",
    "1": "Вердикт верен, но обоснование ошибочно или существенно неполно.",
    "2": "Верный вердикт и корректное обоснование по обоим пунктам.",
    "criteria": [
      "PCA не использует метки",
      "дисперсия, а не разделение классов"
    ]
  }
}
```

## R012 (тип D)

Напишите полный автономный Python-код на sklearn без загрузки данных из сети. На load_wine сделайте стратифицированный test_size=0.25,random_state=42. Обучите только на train Pipeline(StandardScaler(),PCA(n_components=2,svd_solver='full'),LogisticRegression(max_iter=2000,random_state=42)). Напечатайте JSON с test accuracy. PCA не должен видеть test при fit. Все неуказанные параметры оставьте по умолчанию. Код должен запускаться без исправлений.

Ответ модели:

{"answer": {"accuracy": 0.9333333333333333}, "explanation": "Создан pipeline с этапами масштабирования, уменьшения размерности и логической регрессии. Тестовая точность вычислена после обучения модели.", "confidence": 100}

"```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# Загрузка датасета
data = load_wine()
X = data.data
y = data.target

# Стратифицированное разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Построение пайплайна
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2, svd_solver='full')),
    ('logreg', LogisticRegression(max_iter=2000, random_state=42))
])

# Обучение только на тренировочной выборке
pipeline.fit(X_train, y_train)

# Вычисление точности на тестовой выборке
test_accuracy = pipeline.score(X_test, y_test)

# Печать результата
print(json.dumps({"accuracy": test_accuracy}))
```"```

Эталон и рубрика:

```json
{
  "reference": {
    "answer": {
      "code_path": "data/reference_code/D05.py",
      "metrics": {
        "accuracy": 0.9111111111111111
      }
    },
    "solution_ru": "Эталонный код и фактические метрики получаются локальным запуском; см. reference.code_path и results/reference/code_metrics.json."
  },
  "rubric": {
    "pass_at_1": "Запуск извлечённого кода без правок с первой попытки (JSON-декодирование строки answer не является правкой).",
    "correctness": "Все метрики отличаются от эталона не более чем на 0.01 абсолютных; best_C/best_k должны совпасть точно.",
    "methodology": [
      "fit preprocessing только на train и внутри CV",
      "подбор только по train/CV",
      "оценка на отдельном test, кроме описательной D07",
      "правильная метрика и pos_label",
      "стратификация для классификации",
      "фиксированные random_state там, где применимо"
    ],
    "manual_review": "Два независимых слепых проверяющих; для каждой методологической ошибки 0/1, затем Cohen kappa. Если kappa<0.6, уточнить рубрику и переоценить. Это не оценка LLM самим эталонным кодом."
  }
}
```

## R013 (тип C)

Нужно предсказывать энергопотребление завода на следующий час по погоде, календарю и прошлому потреблению. Доступны два года почасовых наблюдений. Какую базовую модель и разбиение предложите? Как построить лаги и rolling-признаки без утечки?

Ответ модели:

{"answer": {"model": "ARIMAX (Autoregressive Integrated Moving Average with eXogenous inputs)", "split": "80% data for training and 20% for testing", "lags_and_rolling": "create lags by shifting consumption values backward in time and rolling features using window size equal to the forecast horizon (1 hour), ensuring no overlap between train and test sets"}, "explanation": "ARIMAX подходит для временных рядов с экзогенными переменными (погодой и календарем). Разделение данных на 80/20 стандартное соотношение для обучения и тестирования модели. Лаги создаются путем сдвига потребления назад во времени, а признаки rolling формируются с окном размером прогнозируемого горизонта, чтобы избежать утечки информации.", "confidence": 90}

Эталон и рубрика:

```json
{
  "reference": {
    "answer": "Сезонная наивная модель (тот же час прошлого дня/недели) и Ridge/линейная регрессия или бустинг. Финальный test - последний период, валидация walk-forward/TimeSeriesSplit. Лаги и rolling используют только прошлое (shift перед rolling); погода доступна как прогноз на момент решения, а не будущие фактические измерения. RMSE/MAE по цели.",
    "solution_ru": "Сезонная наивная модель (тот же час прошлого дня/недели) и Ridge/линейная регрессия или бустинг. Финальный test - последний период, валидация walk-forward/TimeSeriesSplit. Лаги и rolling используют только прошлое (shift перед rolling); погода доступна как прогноз на момент решения, а не будущие фактические измерения. RMSE/MAE по цели."
  },
  "rubric": {
    "0": "Непригодное решение или критическая методологическая ошибка.",
    "1": "Основной метод разумен, но один или несколько обязательных аспектов существенно не раскрыты.",
    "2": "Обоснованное решение раскрывает все три критерия без критических ошибок.",
    "criteria": [
      "временное разбиение и сезонный baseline",
      "разумная регрессионная модель/метрика",
      "лаги/rolling и погода доступны в момент прогноза"
    ],
    "accepted_alternatives": "Не требуется единственный конкретный алгоритм."
  }
}
```

## R014 (тип B)

Оцените утверждение: верно или неверно? Обоснуйте ответ. Для kNN с евклидовым расстоянием изменение единиц измерения одного признака может изменить набор ближайших соседей и предсказание.

Ответ модели:

{"answer": true, "explanation": "При изменении единиц измерения одного признака изменяется масштаб этого признака относительно других признаков, что может привести к изменению порядка расстояний между объектами и, соответственно, набора ближайших соседей.", "confidence": 90}

Эталон и рубрика:

```json
{
  "reference": {
    "answer": true,
    "solution_ru": "Евклидово расстояние суммирует квадраты разностей; масштаб признака меняет его вклад и может изменить порядок соседей."
  },
  "rubric": {
    "0": "Вердикт неверен или отсутствует.",
    "1": "Вердикт верен, но обоснование ошибочно или существенно неполно.",
    "2": "Верный вердикт и корректное обоснование по обоим пунктам.",
    "criteria": [
      "масштаб изменяет вклад в расстояние",
      "возможна смена порядка соседей"
    ]
  }
}
```

## R015 (тип C)

В обучающих данных по станкам 1% отказов. Пропуск отказа стоит 100 условных единиц, ложная тревога - 2. Доступны размеченные примеры обоих классов. Предложите стартовую модель, метрики и способ выбора порога. Можно ли выбрать модель по одной accuracy?

Ответ модели:

{"answer": "логистическая регрессия, AUC ROC, выбор порога на основе F1 score", "explanation": "Логистическая регрессия подходит для бинарной классификации отказов станков. Метрика AUC ROC позволяет оценить общую производительность модели независимо от выбранного порога. Выбор оптимального порога лучше всего осуществлять через балансировку между истинными положительными и отрицательными результатами (F1 score), учитывая стоимость пропуска и ложной тревоги.", "confidence": 85}

Эталон и рубрика:

```json
{
  "reference": {
    "answer": "Логистическая регрессия/бустинг с учётом дисбаланса; PR-AUC, recall/precision и стоимость 100FN+2FP. Порог выбирать на validation/OOF по стоимости, test оставить финальным. Accuracy одной недостаточно: константа даёт 99%. Для идеально калиброванной p(failure) и нулевой цены верных решений порог 2/102, но калибровку надо проверить.",
    "solution_ru": "Логистическая регрессия/бустинг с учётом дисбаланса; PR-AUC, recall/precision и стоимость 100FN+2FP. Порог выбирать на validation/OOF по стоимости, test оставить финальным. Accuracy одной недостаточно: константа даёт 99%. Для идеально калиброванной p(failure) и нулевой цены верных решений порог 2/102, но калибровку надо проверить."
  },
  "rubric": {
    "0": "Непригодное решение или критическая методологическая ошибка.",
    "1": "Основной метод разумен, но один или несколько обязательных аспектов существенно не раскрыты.",
    "2": "Обоснованное решение раскрывает все три критерия без критических ошибок.",
    "criteria": [
      "не ограничиваться accuracy и сравнить с константой",
      "стоимость/PR-метрики и обоснованная модель",
      "порог по validation/OOF, не по test"
    ],
    "accepted_alternatives": "Конкретное значение порога 2/102 необязательно для 2 баллов; необходим корректный способ его выбора."
  }
}
```

## R016 (тип C)

Нужно классифицировать 6000 коротких заявок ремонта на 4 категории. Представление - разреженные неотрицательные частоты 12 000 слов. Обучение и предсказание должны быть быстрыми. Предложите два разумных baseline и схему оценки; нужно ли превращать матрицу в плотную?

Ответ модели:

{"answer": {"baselines": ["использовать Naive Bayes classifier из-за его простоты и скорости обучения, особенно подходящим для больших разреженных матриц частот слов", "применить логистическую регрессию как простой и быстрый метод классификации"], "evaluation_scheme": "разделить данные на обучающий (80%) и тестовый (20%) наборы, использовать перекрестную проверку k-fold (k=5) для оценки точности модели", "плотность": false}, "explanation": "Naive Bayes и логистическая регрессия являются базовыми алгоритмами с низкой сложностью вычислений и подходят для работы с разреженными данными. Перекрестная проверка позволяет получить более надежную оценку качества модели, избегая переобучения. Преобразование матрицы в плотную форму может привести к увеличению памяти и времени обработки, поэтому лучше оставить ее разреженной.", "confidence": 90}

Эталон и рубрика:

```json
{
  "reference": {
    "answer": "MultinomialNB/ComplementNB и линейная логистическая регрессия/LinearSVC. Сохранять sparse; vectorizer учить внутри train/CV. Стратифицированное разбиение, macro-F1 и матрица ошибок; дубликаты заявок не должны пересекать части.",
    "solution_ru": "MultinomialNB/ComplementNB и линейная логистическая регрессия/LinearSVC. Сохранять sparse; vectorizer учить внутри train/CV. Стратифицированное разбиение, macro-F1 и матрица ошибок; дубликаты заявок не должны пересекать части."
  },
  "rubric": {
    "0": "Непригодное решение или критическая методологическая ошибка.",
    "1": "Основной метод разумен, но один или несколько обязательных аспектов существенно не раскрыты.",
    "2": "Обоснованное решение раскрывает все три критерия без критических ошибок.",
    "criteria": [
      "байесовский и линейный baseline для sparse текста",
      "не уплотнять; vectorizer без утечки",
      "стратификация/macro-F1 и контроль дубликатов"
    ],
    "accepted_alternatives": "Допустим TF-IDF вместо counts с корректным Pipeline; GaussianNB не является естественным baseline для разреженных частот."
  }
}
```

## R017 (тип D)

Напишите полный автономный Python-код на sklearn без загрузки данных из сети. Сгенерируйте X,y=make_blobs(n_samples=300,centers=3,n_features=2,cluster_std=0.8,random_state=42). Игнорируя y, примените StandardScaler к X, затем KMeans(n_clusters=3,n_init=10,random_state=42). Напечатайте JSON с silhouette по масштабированному X и предсказанным кластерам. Это описательная кластеризация всех 300 объектов; отдельный test не требуется. Не используйте accuracy по номерам кластеров. Все неуказанные параметры оставьте по умолчанию. Код должен запускаться без исправлений.

Ответ модели:

{"answer": {"silhouette": 0.56}, "explanation": "использование make_blobs для создания синтетических данных и kmeans для кластеризации; вычисление показателя silhouette для оценки качества кластеров", "confidence": 90}

```python
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# создание синтетических данных
X, _ = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=0.8, random_state=42)

# стандартная нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# выполнение кластеризации
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# расчет показателя silhouette
silhouette = silhouette_score(X_scaled, clusters)

# вывод результата в формате json
result = {
    'silhouette': silhouette
}
print(result)
```

Эталон и рубрика:

```json
{
  "reference": {
    "answer": {
      "code_path": "data/reference_code/D07.py",
      "metrics": {
        "silhouette": 0.8783134858331793
      }
    },
    "solution_ru": "Эталонный код и фактические метрики получаются локальным запуском; см. reference.code_path и results/reference/code_metrics.json."
  },
  "rubric": {
    "pass_at_1": "Запуск извлечённого кода без правок с первой попытки (JSON-декодирование строки answer не является правкой).",
    "correctness": "Все метрики отличаются от эталона не более чем на 0.01 абсолютных; best_C/best_k должны совпасть точно.",
    "methodology": [
      "fit preprocessing только на train и внутри CV",
      "подбор только по train/CV",
      "оценка на отдельном test, кроме описательной D07",
      "правильная метрика и pos_label",
      "стратификация для классификации",
      "фиксированные random_state там, где применимо"
    ],
    "manual_review": "Два независимых слепых проверяющих; для каждой методологической ошибки 0/1, затем Cohen kappa. Если kappa<0.6, уточнить рубрику и переоценить. Это не оценка LLM самим эталонным кодом."
  }
}
```

## R018 (тип B)

Оцените утверждение: верно или неверно? Обоснуйте ответ. StandardScaler можно обучить на объединении train и test до разбиения: утечки не будет, потому что scaler не использует целевые метки.

Ответ модели:

{"answer": false, "explanation": "StandardScaler не использует целевые метки, но он все равно должен обучаться только на тренировочном наборе данных, чтобы избежать смещения модели. Объединение train и test перед масштабированием приведет к утечке информации из тестового набора в тренировочный.", "confidence": 90}

Эталон и рубрика:

```json
{
  "reference": {
    "answer": false,
    "solution_ru": "Статистики тестовых признаков тоже являются информацией теста. Scaler обучают только на train и внутри каждого train-фолда CV."
  },
  "rubric": {
    "0": "Вердикт неверен или отсутствует.",
    "1": "Вердикт верен, но обоснование ошибочно или существенно неполно.",
    "2": "Верный вердикт и корректное обоснование по обоим пунктам.",
    "criteria": [
      "статистики test создают утечку",
      "fit только train/внутри CV"
    ]
  }
}
```

## R019 (тип B)

Оцените утверждение: верно или неверно? Обоснуйте ответ. В soft-margin SVM при прочих равных увеличение C усиливает штраф за нарушения зазора и соответствует более слабой регуляризации.

Ответ модели:

{"answer": false, "explanation": "В soft-margin SVM параметр C регулирует баланс между штрафом за ошибки (нарушения зазора) и сложностью гиперплоскости. Увеличение C увеличивает штраф за ошибки, что соответствует более жесткой регуляризации.", "confidence": 90}

Эталон и рубрика:

```json
{
  "reference": {
    "answer": true,
    "solution_ru": "В целевой функции 0.5*||w||^2 + C*sum(xi) вклад ошибок относительно нормы растёт. Это не гарантия роста качества на тесте."
  },
  "rubric": {
    "0": "Вердикт неверен или отсутствует.",
    "1": "Вердикт верен, но обоснование ошибочно или существенно неполно.",
    "2": "Верный вердикт и корректное обоснование по обоим пунктам.",
    "criteria": [
      "C увеличивает вес нарушений",
      "регуляризация относительно ошибки слабее"
    ]
  }
}
```

## R020 (тип D)

Напишите полный автономный Python-код на sklearn без загрузки данных из сети. Загрузите load_diabetes(return_X_y=True) с параметрами загрузчика по умолчанию. Разбейте test_size=0.2,random_state=42 без стратификации. Обучите Pipeline(StandardScaler(),LinearRegression()) только на train. На test напечатайте JSON с rmse и r2. RMSE должен быть корнем из MSE, а не MSE. Все неуказанные параметры оставьте по умолчанию. Код должен запускаться без исправлений.

Ответ модели:

{"answer": {"answer": """import json
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Загрузка данных
X, y = load_diabetes(return_X_y=True)

# Разбиение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Построение Pipeline
model = Pipeline([('scaler', StandardScaler()), ('linear_regression', LinearRegression())])

# Обучение модели
model.fit(X_train, y_train)

# Прогнозирование на тестовой выборке
y_pred = model.predict(X_test)

# Вычисление метрик
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# Форматируем результат в JSON
result = {
    'rmse': rmse,
    'r2': r2
}
print(json.dumps(result))
"""", "explanation": "Реализован полный автономный код на sklearn, который выполняет все требуемые шаги: загрузку данных, разбиение на обучающую и тестовую выборки, обучение Pipeline с масштабированием и линейной регрессией, вычисление RMSE и R2 и вывод результата в формате JSON.", "confidence": 100}

Эталон и рубрика:

```json
{
  "reference": {
    "answer": {
      "code_path": "data/reference_code/D06.py",
      "metrics": {
        "rmse": 53.85344583676594,
        "r2": 0.45260276297191926
      }
    },
    "solution_ru": "Эталонный код и фактические метрики получаются локальным запуском; см. reference.code_path и results/reference/code_metrics.json."
  },
  "rubric": {
    "pass_at_1": "Запуск извлечённого кода без правок с первой попытки (JSON-декодирование строки answer не является правкой).",
    "correctness": "Все метрики отличаются от эталона не более чем на 0.01 абсолютных; best_C/best_k должны совпасть точно.",
    "methodology": [
      "fit preprocessing только на train и внутри CV",
      "подбор только по train/CV",
      "оценка на отдельном test, кроме описательной D07",
      "правильная метрика и pos_label",
      "стратификация для классификации",
      "фиксированные random_state там, где применимо"
    ],
    "manual_review": "Два независимых слепых проверяющих; для каждой методологической ошибки 0/1, затем Cohen kappa. Если kappa<0.6, уточнить рубрику и переоценить. Это не оценка LLM самим эталонным кодом."
  }
}
```

## R021 (тип B)

Оцените утверждение: верно или неверно? Обоснуйте ответ. Наивный Байес автоматически выдаёт хорошо калиброванные вероятности на любом датасете, поскольку это вероятностный алгоритм.

Ответ модели:

{"answer": false, "explanation": "Наивный Байес — вероятностный алгоритм, но хорошая калибровка вероятностей зависит от качества обучающего набора данных и корректности реализации модели.", "confidence": 80}

Эталон и рубрика:

```json
{
  "reference": {
    "answer": false,
    "solution_ru": "Вероятностный выход не гарантирует калибровку; нарушение условной независимости и распределительных предположений может приводить к избыточной уверенности."
  },
  "rubric": {
    "0": "Вердикт неверен или отсутствует.",
    "1": "Вердикт верен, но обоснование ошибочно или существенно неполно.",
    "2": "Верный вердикт и корректное обоснование по обоим пунктам.",
    "criteria": [
      "вероятностная модель не гарантирует калибровку",
      "роль предположений/зависимостей признаков"
    ]
  }
}
```

## R022 (тип C)

У датчика 60 сильно коррелирующих числовых каналов. Для визуализации нужно двумерное представление без использования меток. Предложите метод и preprocessing. Гарантирует ли он сохранение различий между классами неисправностей?

Ответ модели:

{"answer": "t-SNE", "explanation": "t-SNE (техника стохастического соседства) является подходящим методом для визуализации высокомерных данных в двухмерном пространстве без необходимости использования меток. Он сохраняет локальные структуры данных, что может помочь сохранить различия между классами.", "confidence": 85}

Эталон и рубрика:

```json
{
  "reference": {
    "answer": "PCA до 2 компонент, центрирование, обычно стандартизация при разных масштабах, обработка пропусков по train. Показать explained_variance_ratio_. PCA сохраняет дисперсию, а не гарантирует разделение классов; при оценке классификатора всё преобразование обучать внутри train.",
    "solution_ru": "PCA до 2 компонент, центрирование, обычно стандартизация при разных масштабах, обработка пропусков по train. Показать explained_variance_ratio_. PCA сохраняет дисперсию, а не гарантирует разделение классов; при оценке классификатора всё преобразование обучать внутри train."
  },
  "rubric": {
    "0": "Непригодное решение или критическая методологическая ошибка.",
    "1": "Основной метод разумен, но один или несколько обязательных аспектов существенно не раскрыты.",
    "2": "Обоснованное решение раскрывает все три критерия без критических ошибок.",
    "criteria": [
      "PCA и обоснование масштаба/центрирования",
      "объяснённая дисперсия",
      "нет гарантии сохранения классов, соблюдение train/test"
    ],
    "accepted_alternatives": "Другой метод визуализации допустим при явном обсуждении отличий; PCA является простым эталонным вариантом."
  }
}
```

## R023 (тип C)

Для диагностики подшипников есть 180 размеченных записей, 320 числовых спектральных признаков и два класса. Требуется компактная модель с объяснимыми коэффициентами. Какой стартовый метод и схему проверки вы предложите? Укажите обработку признаков и основной риск.

Ответ модели:

{"answer": "логистическая регрессия", "explanation": "Логистическая регрессия является простым и интерпретируемым методом, подходящим для данной задачи из-за небольшого количества классов и необходимости объяснения коэффициентов модели.", "confidence": 90}

Эталон и рубрика:

```json
{
  "reference": {
    "answer": "Регуляризованная логистическая регрессия (L1/L2/elastic net) в Pipeline с масштабированием, стратифицированная CV и подбор силы регуляризации только внутри train. При повторных записях одного подшипника нужен group split. Основной риск: p>n, переобучение/нестабильность коэффициентов; корреляция ограничивает интерпретацию.",
    "solution_ru": "Регуляризованная логистическая регрессия (L1/L2/elastic net) в Pipeline с масштабированием, стратифицированная CV и подбор силы регуляризации только внутри train. При повторных записях одного подшипника нужен group split. Основной риск: p>n, переобучение/нестабильность коэффициентов; корреляция ограничивает интерпретацию."
  },
  "rubric": {
    "0": "Непригодное решение или критическая методологическая ошибка.",
    "1": "Основной метод разумен, но один или несколько обязательных аспектов существенно не раскрыты.",
    "2": "Обоснованное решение раскрывает все три критерия без критических ошибок.",
    "criteria": [
      "регуляризованная линейная модель и связь с p>n/интерпретацией",
      "масштабирование внутри Pipeline",
      "CV без утечки и риск нестабильности"
    ],
    "accepted_alternatives": "Допустима другая компактная регуляризованная модель, если объяснены ограничения; простое дерево допустимо с ограничением сложности и сравнением стабильности."
  }
}
```
