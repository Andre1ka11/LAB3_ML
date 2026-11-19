# Экспертная проверка по рубрикам задания

Содержательный просмотр всех первых ответов B/C/D; сверка с исходными условиями, эталонами и сохранёнными результатами запусков. Ответы и код не исправлялись; новых запусков моделей/их программ не было.

Выполнена одна экспертная проверка по содержательным рубрикам задания. Требование пункта 8 о двух независимых слепых оценках участников команды этой записью не подтверждается; kappa не вычисляется. Оценки созданы после просмотра результатов, не являются предрегистрацией. Дополнительный код GigaChat рассматривается диагностически и не исправляет первичный неуспех формата.

B: 0 неверный вердикт; 1 верный вердикт с ошибкой/существенным пропуском; 2 оба критерия раскрыты. C: 0 непригодное решение/критическая ошибка; 1 разумная основа с пропусками; 2 все критерии по смыслу раскрыты. Пропущенный этап не додумывается. D: 1 наблюдаемая ошибка, 0 ошибка в представленном коде не найдена, N/A критерий неприменим. Ноль не означает доказательство выполнения программы.

Проверено 92 первых ответа: по 8 B, 7 C и 8 D для каждой из четырёх моделей. Хеши ответа, задания и request_id проверяются при каждом пересчёте. Исходные формы двух участников сохранены отдельно и не заполнялись этой проверкой.

## Сводка

| Модель | B среднее / 2 | C среднее / 2 | C полностью / 7 | D строгий / 8 | Дополнительный результат / 39 |
| --- | --- | --- | --- | --- | --- |
| YandexGPT | 1.875 | 1.143 | 1 | 5 | 19 |
| Qwen3-4B | 1.250 | 0.571 | 1 | 2 | 12 |
| GigaChat-2 | 1.500 | 0.857 | 0 | 0 | 11 |
| DeepSeek | 2.000 | 1.429 | 3 | 8 | 35 |

39 = 24 A/B/F с автоматическими вердиктами B + 7 C (только 2 балла считаются полным успехом) + 8 D (первичный контракт, выполнение, численные метрики и методология одновременно). Это дополнительная оценка после просмотра результатов, а не финальная оценка двух слепых участников. Средний балл B отдельно учитывает качество обоснований.

Успех C строже разумного выбора алгоритма: нужны все обязательные аспекты рубрики. Поэтому подробный ответ также может получить 1 за конкретный пропуск. D=0 по флагу означает отсутствие найденной ошибки в тексте, а не успешный запуск. Для отсутствующего кода стоят N/A.

Giga D01 показывает, зачем нужен содержательный просмотр: best_estimator_.score выдаёт accuracy, хотя результат подписан balanced_accuracy. У Qwen D04 weighted-F1 заменяет F1 класса 1, D06 корреляция заменяет R². Статический просмотр не исправляет эти программы и не превращает диагностику Giga в primary pass.

## Построчные решения

### YandexGPT

#### B01 — 2/2

Правильно: PCA без меток, максимизирует вариативность, не разделимость классов.

Исходный request_id: `c580836777950786ac1559223dc929e6eff22916e99dad904900e34952b705d3`; SHA256 ответа: `b83d2d13c2c56881c459d1438ae650719c7e5695ddc8e8e67502f914c02433c8`.

#### B02 — 2/2

Указаны отсутствие гарантии калибровки и нарушение независимости признаков.

Исходный request_id: `dac375104849649228badba9e2a416faf90ff82d8ceedd330d0d545e6fe988e1`; SHA256 ответа: `6320c95f4d2093a74801c6b12af6c3ffd7862ee7639467521ea8023622905390`.

#### B03 — 1/2

Утечка через среднее и стандартное отклонение объяснена верно, но исправленная схема fit только на train/в CV не сформулирована.

Исходный request_id: `12becf0d1936428d81ccb49b7699f2ba9fac7e05a0ca7d4f43df1079e92ff46b`; SHA256 ответа: `aeb1177f38a8d5173434487a4ccbb90f40992510df832a2a98bf30f7bfda81ee`.

#### B04 — 2/2

Отделены общая различающая способность ROC-AUC и precision при пороге; названа зависимость от доли класса.

Исходный request_id: `e89b4b17918e40e8b622b6e70514da14819ff12a5d0cceb11e0d28e3c28d8f60`; SHA256 ответа: `aa7c179bede347be6eade055842a9f95212cd1c13eca4921357090da98e94eea`.

#### B05 — 2/2

Оба направления верны: штраф растёт, относительная регуляризация слабее.

Исходный request_id: `5073c9a874ecb74ddf5120a893d0f34bd253ce56c665675dbe2fd5808affba59`; SHA256 ответа: `74ce56d824bc03146c6df8bcbaf4599f9c2f78aa64629b7b18049b688af04888`.

#### B06 — 2/2

Масштаб меняет расстояния, соседей и итоговое предсказание.

Исходный request_id: `e516d41983674645ca1fcf133b792d0341338c9b90716d5a1314bdea1bc0bd64`; SHA256 ответа: `d89510fc48dbfd3ea14637198318e5ad49d88b1cad3f629eeeb1abb586dca38b`.

#### B07 — 2/2

Верно объяснены разнообразие и снижение корреляции. Упрощение «дерево на наборе признаков» следует уточнить: подмножество выбирается при разбиениях.

Исходный request_id: `20bcfb6b7bc41b21e9ba7fc3313f655c0a8be056dc46b5331cd985c3dbbc2082`; SHA256 ответа: `8aff2280ae644f41045c46005eb6c96867351f51823db33bbb9a880b83b141ad`.

#### B08 — 2/2

Указаны сумма квадратов и зависимость локального решения от инициализации.

Исходный request_id: `0f590ce082c32d3b344d23e55184e1a267e63c96b5e80b2d18f16f9418d1821f`; SHA256 ответа: `1f2829d75888053681f0377c4b482f5e9a667db9478aa868c9b1139a5cfd6c0d`.

#### C01 — 1/2

Логистическая регрессия, масштабирование и риск p>n разумны; нет Pipeline, подбора регуляризации внутри CV и защиты preprocessing от утечки.

Исходный request_id: `5d8966192d5e8f8f59c9766b4ea3a33d9b9f4e2bd4d0c31d75b89f5ac56d01a2`; SHA256 ответа: `3e8a99c14f45e92ff624a6f93cd237d93bf6a673c6d57c3bed45da3d140f95ca`.

#### C02 — 2/2

KMeans обоснован, разные единицы требуют стандартизации, k выбирается по локтю, предметная проверка по центроидам присутствует; группы трактуются как режимы.

Исходный request_id: `e39dc53d6b124cfb3bf2bef7c65e5962d12917c8fab03dad16bbdb4129150415`; SHA256 ответа: `66faca17551b0f8c9fbc95f129024d8082dd5cef3099e0d46a488713d7b31e09`.

#### C03 — 1/2

Бустинг и LR с OHE/импутацией разумны. Не закреплены общие фолды, подбор только по train и обучение preprocessing внутри train; порядок перечисления сам по себе не доказывает утечку.

Исходный request_id: `f33834c8f42a2a1bc9eebc8106c6158add610597cf40903200656667e732ef31`; SHA256 ответа: `4a52db80e66c9691a7143144f9eea23319bca952aeae5afdce5a0688a9645a6b`.

#### C04 — 1/2

Есть модель, отказ от accuracy и стоимости ошибок; не сказано выбирать порог на validation/OOF, сохранив test для финала.

Исходный request_id: `dcf829d7e1182c515cde40a58a7da4428c08be9649a0ec385204db25da0acfdb`; SHA256 ответа: `73c1bfbf20011ce05bd22d04aceb47775e5c3a31e798dae3c78e65682dfd4cca`.

#### C05 — 1/2

Хронологические train/validation/test и задержанные признаки верны; нет сезонного наивного baseline и метрики, не разобрана доступность прогноза погоды.

Исходный request_id: `5c0f8e6524400cded9579ecf9b2bb8e2942572620b67a87987c0ba1dcb4463a3`; SHA256 ответа: `0346bb044a7f80eb5e01777f0c6da24c0e2500c00d63d86cd2ef23dbadf26efb`.

#### C06 — 1/2

PCA и отсутствие гарантии разделения верны; отсутствуют preprocessing, выбор масштаба и обсуждение train/test.

Исходный request_id: `707132cc39f8c7156d6783e25457cc3cc6c99e625b0b76475dfb51afa3753bfe`; SHA256 ответа: `338eccdd6e5f3f305fabd81932048674315c984a8befbe0829c13566909ec70f`.

#### C07 — 1/2

MultinomialNB и L2-LR подходят sparse-частотам, macro-F1 назван; нет vectorizer внутри train, стратификации и контроля дубликатов.

Исходный request_id: `99d93d804e903ad16dd812bcfd14fe2bc88bd93455da2bbbb71ba2b74168de7c`; SHA256 ответа: `7125b937fb4f80c6b4bc92eb4e60f8973794ff28d393c0392992a68f5290f2d6`.

#### D01 — методология кода

Pipeline внутри GridSearchCV, CV и holdout корректны; print(dict) печатает Python repr, а не JSON. Методология верна, контракт вывода нарушен.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=0, missing_stratification=0, missing_seed=0.

Исходный request_id: `3be93d4ad01f53545c999092d0032a1f0692a7e99f0aea77a4f8ba495b7bf3a4`; SHA256 ответа: `98999399b5605c3d81168df1047fa8acab96852b65c1ee7d6488be9186d4da0c`.

#### D02 — методология кода

Imputer/scaler обучаются только на train, метрики и stratify верны. Нет import json: сохранённый запуск падает, исправления не вносятся.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=0, missing_stratification=0, missing_seed=0.

Исходный request_id: `653bf1a39abda58b9e231b79c3f265ea04998751c568d3d0eadf7251ad93f44b`; SHA256 ответа: `97603c47ef3dc9fc5f516b9c9e1608cbcf818ee39eb5f70fcaf5e725cfeeecf6`.

#### D03 — методология кода

Scaler внутри Pipeline/CV, k выбран только по train; holdout accuracy и stratify/seed корректны.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=0, missing_stratification=0, missing_seed=0.

Исходный request_id: `2ba2034c0ca4aaea8fedf2e2aaed9866349821421bd44200b58cb3ce95e160c0`; SHA256 ответа: `4c46a78edce439e6c4e40a4d22c99b9d277eb339808d6db19286ee29f1fcbee4`.

#### D04 — методология кода

RF обучен на train, balanced_accuracy и F1 класса 1 вычислены на test; параметры/seed соответствуют.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=0, missing_stratification=0, missing_seed=0.

Исходный request_id: `358ddabdfd2e4dc6084282c1c91388f16e4531a5b4a016a5d67a70b7512ce9a3`; SHA256 ответа: `1b695462f9e11b733ebbefacd38c56825abb66843b78c1d684876e6bb0411fcc`.

#### D05 — методология кода

Scaler/PCA/LR fit только на train; стратифицированный holdout, accuracy и seed корректны.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=0, missing_stratification=0, missing_seed=0.

Исходный request_id: `dc604616e7551accbfc637e668042dd8a8d000a38d642843522da7357922ad3c`; SHA256 ответа: `67719c0c57810d62b185590d7f2601691142c55a664ef4f258e828199881a2e0`.

#### D06 — методология кода

Train-only Pipeline, RMSE как sqrt(MSE) и r2_score правильны. squared=False несовместим с окружением сохранённого запуска; это API-сбой, не неправильная формула.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=0, missing_stratification=N/A, missing_seed=0.

Исходный request_id: `d4af9c7b3afdbb03a5fa137e1acb3db02d755074d094beccea94084861b121ec`; SHA256 ответа: `be5d44c6bece0959fab73db69bc4e14dcdf6a4f9ca32569d7711ca8411f0032e`.

#### D07 — методология кода

Описательная кластеризация всех 300 объектов допустима по условию; silhouette по scaled X и clusters, y не используется, seed/n_init верны.

Флаги: leakage=N/A, test_tuning=N/A, train_evaluation=N/A, wrong_metric=0, missing_stratification=N/A, missing_seed=0.

Исходный request_id: `02d5bc40544670186abe7ad418189b54b464280b7ac1ae15aafe1e3eadbab990`; SHA256 ответа: `eb250cdce7697d73f9f2f4feaded24b15918cf4fdddd409f854fab6f50f514dd`.

#### D08 — методология кода

GaussianNB без scaler, исходные метки сохранены, F1 pos_label=0 и balanced_accuracy на test; stratify/seed корректны.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=0, missing_stratification=0, missing_seed=0.

Исходный request_id: `7936f73b9adc273f7b7ea3034c02602274e52c17a103505145ab5de959cb7bdd`; SHA256 ответа: `5f77387b7123d759f4d5ee8578f602a5538531d62c704354930cbae198c2055e`.

### Qwen3-4B

#### B01 — 2/2

Верно противопоставлены PCA без меток с максимизацией вариации и LDA.

Исходный request_id: `679d72041694139afea31d059c5c1a0558c9713c07a5c63161e1d5f44b3bc0ba`; SHA256 ответа: `fbd12d6b32d7b759baab8e2a548ba6e46024e42ffed2c5dd484d4addb1675466`.

#### B02 — 1/2

Отсутствие гарантии признано, но роль условной независимости/коррелированных признаков не раскрыта; общие слова о качестве и настройке недостаточны.

Исходный request_id: `0737979593b5d43a6d9f22ff7d507b07bbb1aa97613800d819d6a9c216990685`; SHA256 ответа: `e1182c4dd198554c26418ac7c0a75f46f8c89d5b7309aeb42b513f25ad9cb6b6`.

#### B03 — 0/2

Вердикт true неверен: статистики test создают утечку даже без y.

Исходный request_id: `cfa15642c62e8d11f3f5f0f7baa682621324b7c39e92c42b40922dea50116d0a`; SHA256 ответа: `97346f751a6e9c7ced712db21ea65089f1942cc0874ff043e979164f47dba452`.

#### B04 — 2/2

Объяснена зависимость precision от порога и распределения классов, гарантия из ROC-AUC отвергнута.

Исходный request_id: `d91f4e9a6fa6c4f295e977da72c497448c9cb16ecb16a709087681a08151c488`; SHA256 ответа: `3d0223a3076788e8d7a7a7b4447ee19696515efd9d65a6efb4e0b5a1c8dd8ffe`.

#### B05 — 0/2

Неверный вердикт и перепутанные направления действия C: штраф растёт, регуляризация ослабевает.

Исходный request_id: `d2a1461c2c5806aff72c2493ee50ff0d7bb3a3b77d10654e505a39a13dc4678c`; SHA256 ответа: `3b0e8b1c4a3b4c99fdb9fd94beff6d84c30d15190138f1ad23059e30b986aff3`.

#### B06 — 2/2

Изменение масштаба меняет расстояния и ближайших соседей.

Исходный request_id: `aa61122164166368a565833ba20b16a26c7d84ca814916bf94c59a6f0d6e76d4`; SHA256 ответа: `3dcecdc89a1f64b8cf622aefacde0faf1f5d3bbec1d6a43e6410cc7b2768229b`.

#### B07 — 2/2

Есть разнообразие признаков, меньшая корреляция и выигрыш обобщения.

Исходный request_id: `23b8ed557faeeaf053d86d24cc40b5f530051c6a1b80f405231a9af4a9516d89`; SHA256 ответа: `6aa794ce4508790b91b7b1a04b4ea5ebf16fcdef48737d14c08f66742b00963e`.

#### B08 — 1/2

Вердикт и целевая сумма квадратов верны, но обычный KMeans ошибочно назван градиентным спуском; используется чередование назначений и пересчёта центров.

Исходный request_id: `d53c65879b024422e4b003946d863a1f4741a186ce36d657a772a3d86ea92f92`; SHA256 ответа: `1473dbddaac52d618c4ecefd9da600d8260ff04576df51a8376ea0951dc0d9b8`.

#### C01 — 0/2

Предложена регрессия Lasso для двух классов без правила классификации/порога и его проверки. Это не обоснованный классификатор; Pipeline также отсутствует.

Исходный request_id: `89c2160f7d72aacc8906301ab73f22321fdc4e29cfcc581ca7c27b1866654fa0`; SHA256 ответа: `c1e0553f5848bc549fb0d9b958e3d73adf5ff2d65832e2d5190a69934dbfa606`.

#### C02 — 2/2

Есть кластеризация без меток, нормализация, elbow/silhouette и проверка соответствия реальным режимам; словесная опечатка не меняет смысл.

Исходный request_id: `0b3aabfaed6c65a39dfd1cba39cfcc809f6709d23a30f0792cafd909ade7b8f3`; SHA256 ответа: `6d4f4015df57feb62350a8b7ee48f3a961028df621839a45d4b2177eb993c8a0`.

#### C03 — 1/2

RF и сравнение с LR разумны, но кодирование категорий, конкретная обработка пропусков и общие фолды/метрика без утечки не раскрыты.

Исходный request_id: `54df435d23e409d33b8fa690aa1c18494025418dae5c22436af189b4bfcfc3d8`; SHA256 ответа: `dea25df159de0fc8ea28a8b9e01defee281671d82004a889454ce5e4eb31bd89`.

#### C04 — 0/2

Необоснованный отказ: доля отказов и цены ошибок уже заданы; можно предложить baseline и validation-подбор порога.

Исходный request_id: `6cac4403e20660189e2b716b51fc8be45e21823d1205a74b967ccab1f3fe569d`; SHA256 ответа: `74ec9eedc1ed57ffd79bdd3e23e2e30a47ebf27eb626c30e5c759ee22fe55cf4`.

#### C05 — 0/2

Случайное разбиение временного ряда — критическая ошибка оценки прогноза будущего; оговорка про сезонность её не устраняет.

Исходный request_id: `22e39415c8492ff1e0a7ffcdb4f409fad0647f174016c3b660147833eb502359`; SHA256 ответа: `74120ce2f5c0b9d81c27bd312e85d1e9af7f28766272ab6966d7a7f06e1744ab`.

#### C06 — 0/2

Заявлено сохранение наиболее значимых различий классов, чего PCA без меток не гарантирует; preprocessing отсутствует.

Исходный request_id: `d0ef6e06b2dd29c2863b6cca51975bcc4cc08396de9b7df86ac815c560b9b026`; SHA256 ответа: `61bc6975bb3a26b6228609ab3d77dfed8640cff68ab80d683f78740fc7f4b07e`.

#### C07 — 1/2

LR и sparse-представление разумны, но второй baseline RF не соответствует требуемой байесовской паре; нет macro-F1, стратификации и защиты vectorizer от утечки. answer=1 само по себе ничего не объясняет.

Исходный request_id: `a0156f0e391387064ab76b2e847adb45f5649077b4fab9b1b31b1c156903bb7c`; SHA256 ответа: `9ab043039693b4bc5265d14bd8f59a5a36a9d4c155b5df6ddf38c55e3a6badab`.

#### D01 — методология кода

По тексту Pipeline/CV и holdout задуманы правильно, но y y_test — SyntaxError; никакой успешной методологии запуска не подтверждено.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=0, missing_stratification=0, missing_seed=0.

Исходный request_id: `b06a969b6e68c57d967ed52e3c1640fcf86e74c21eb83e53ea45acf493b62f99`; SHA256 ответа: `f5e5ff9e1c3de093985ccd1dd2acc108f2809e0e6a15210983d73f4ed0949911`.

#### D02 — методология кода

Train-only imputer/scaler/LR, stratify и seed; F1 класса 1 и balanced_accuracy на holdout корректны.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=0, missing_stratification=0, missing_seed=0.

Исходный request_id: `fbb137d4753580b9a6a954071f7753348aa393bab4546c5f1b410175568d8f50`; SHA256 ответа: `cdcdf4c2ae6a2c8effde3733aa4378a8d178e71ef1fdbf458606e93819f061b3`.

#### D03 — методология кода

Pipeline/CV и holdout корректны по исходнику, но json_result не напечатан; stdout пуст, JSON-контракт не выполнен.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=0, missing_stratification=0, missing_seed=0.

Исходный request_id: `22bb65fd87af9b92a5017fee11d98542a5d0d4309996fd48343c1ff9646b31e5`; SHA256 ответа: `1e3c3b02dc3a7739ab0d9e859e120a5c12899725e666e0fd27de7946485f6d6f`.

#### D04 — методология кода

В видимом коде F1 average=weighted вместо бинарного F1 класса 1. Дополнительно весь ответ не разбирается как JSON, вывод с шаблонами не является вычисленным результатом.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=1, missing_stratification=0, missing_seed=0.

Исходный request_id: `a2f57ec3f18652d3cbe0ed59311b250c596a40a309dcf4239a454fb086607223`; SHA256 ответа: `8d40915cc841bef72d00df95e9439a810abe55785c617b3bf88e8183b9a7027b`.

#### D05 — методология кода

В answer число 0.75 вместо программы; объяснение не заменяет исходник. Методологические флаги неприменимы: проверять нечего.

Флаги: leakage=N/A, test_tuning=N/A, train_evaluation=N/A, wrong_metric=N/A, missing_stratification=N/A, missing_seed=N/A.

Исходный request_id: `6bf54755238abdd452968d1e5900c7ffd58c7abffab6fa9e76485e640caa99e9`; SHA256 ответа: `0447608230f7373ffd806faa065161629a4be18d037973fec30990fda71b0d1e`.

#### D06 — методология кода

Train-only Pipeline и RMSE правильны, но np.corrcoef(...)[0,1] — корреляция, а не R². separators="," также ломает JSON-вывод в сохранённом запуске.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=1, missing_stratification=N/A, missing_seed=0.

Исходный request_id: `5a6f02b5958e76d63417f7caae2df9952c254cf2a676441a4d4d91501cbedd48`; SHA256 ответа: `aed6eab0facf67d808623a882a29ff2f6c1c3bf842ccc4b16b9f19a518f2bf6a`.

#### D07 — методология кода

Только литерал словаря с приблизительным числом, без make_blobs/scaler/KMeans/silhouette и print; exit=0 не означает выполнение задачи.

Флаги: leakage=N/A, test_tuning=N/A, train_evaluation=N/A, wrong_metric=N/A, missing_stratification=N/A, missing_seed=N/A.

Исходный request_id: `771952fba4ca1e71a61d87a869c644d3d8eac97b5a2492d5c3d06a89a6c1c8c6`; SHA256 ответа: `f48a7e6aeff69c72ac91f6bf4e8df8ecdef535d0c829758450fc47e987076dc5`.

#### D08 — методология кода

GaussianNB на train без scaling, test stratified, pos_label=0, balanced_accuracy и seed соответствуют.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=0, missing_stratification=0, missing_seed=0.

Исходный request_id: `62fc1f59e5eaa899adbfa42b01ba607c9b14ed5c644ef030c6512967331fad81`; SHA256 ответа: `c41c4ba2ef9f6ea991514657456d0968f311099a71d5ed56b2a0e6f5c203946c`.

### GigaChat-2

#### B01 — 2/2

PCA не использует классы и ищет максимальную вариацию.

Исходный request_id: `5770b51cc9c454e895f9c54619e92f362bfea6a2e8a05f2fb17cfbea44995b36`; SHA256 ответа: `de43c4653b9335451fde690da8e89a695f70d11f64cbb6a10271e6a1a8a8d696`.

#### B02 — 1/2

Вердикт верен, но ответ о качестве данных/реализации не объясняет предположения NB и зависимости признаков.

Исходный request_id: `b396faa5480e5e46e8ba3cb1639801153cb3a8b9d43535ee8f8ea00261122679`; SHA256 ответа: `fad4b222b7c81f8aaeba54dd29a09708990f29496071dbf872f2ff8c842892b6`.

#### B03 — 2/2

Верно названы утечка из test и необходимость обучать scaler только на train.

Исходный request_id: `154d7a2462610d8dce74e4a021fea1859ccee26b2f00640f291b5200423ff7da`; SHA256 ответа: `2b15f76001868576675416c0b06c383c054c1d465ef26908cb3be173d6f7737a`.

#### B04 — 1/2

Отсутствие гарантии при пороге объяснено, но обязательная зависимость precision от базовой частоты класса пропущена.

Исходный request_id: `48082d98dbd746ac50af85fbf8b072369971098d16d522951abe8cb28d4ae26e`; SHA256 ответа: `b553d63773ad0efa4f500e9458d9effa8304465a8bb51adb1c54952fe2814df5`.

#### B05 — 0/2

Вердикт false неверен: рост штрафа C означает более слабую, а не более жёсткую регуляризацию.

Исходный request_id: `cddd6483e64794c3fff988f1348d3f95a428b4596c5793eec623d80867619958`; SHA256 ответа: `fbf45d8a385b5337deef63a3b8f0f2faf3be83dbd8ee5b83f9d8e040c845e144`.

#### B06 — 2/2

Объяснены относительный масштаб, порядок расстояний и смена соседей.

Исходный request_id: `a6df97ca4dc934b4394ec926aa0f910eb95a735d44bae7f5f26793bf27557db5`; SHA256 ответа: `e774ef36d8ea64172982cb605cee96958beeacc228ada044ee8ed61303027ba8`.

#### B07 — 2/2

Верно названы снижение корреляции и разнообразие структур; усреднение выражено неявно. Уточнение: признаки выбираются на каждом разбиении.

Исходный request_id: `7acdb1adce23c8336ee9defa409fe0d3581d174ef1976bc79c10d7c791632ed3`; SHA256 ответа: `6e7469bcda7ff62a72f2313d7f19480692923967af595f78a42442dd0de4c11b`.

#### B08 — 2/2

Правильная цель KMeans, локальные минимумы и влияние начальных центров.

Исходный request_id: `5d76547afcc8ee54166c3d7628571f486f8e6ba2ed18f7b92e79259494c59e92`; SHA256 ответа: `304f524c88184e0f32ab5f9d1a25ee5d016f62b1e85ecc9ba529363aebb96207`.

#### C01 — 1/2

Логистическая регрессия разумна и интерпретируема, но нет регуляризации при p>n, масштабирования, CV и рисков.

Исходный request_id: `8eb515e8fb575b0f270f4b056bf9c57f0626d6ab3ff26f49a05d50f05b33f89d`; SHA256 ответа: `25dd4576ccd3a0bd8503e0aab402b67c5f6ab68179105992bddd32d8741fd7d8`.

#### C02 — 1/2

KMeans разумен, но сам не определяет неизвестное k; нет масштаба, выбора k и инженерной проверки.

Исходный request_id: `b3c3de1b19a3482fb329797d474d5b94695d0c90c2c96638c48c9614193e623a`; SHA256 ответа: `bc1d2e8f33a697f010bbe29122816f4106d7559265cc76729fdc19441adec585`.

#### C03 — 0/2

XGBRegressor назван как регрессионное решение бинарной задачи без порога/метрики и правил сравнения; категории и пропуски не обработаны.

Исходный request_id: `ffb0faeb4d7190ad4b9800813741d9c89a642c5ebcce5c87bae2e853cd8bf4d9`; SHA256 ответа: `4d78c641a8427613eb4cdb71971b121462db396c6c91f5db401b3c91e3766d43`.

#### C04 — 1/2

Логистическая регрессия разумна, но F1 не кодирует стоимость 100FN+2FP. Нет validation-подбора порога, константного baseline и явного ответа про accuracy.

Исходный request_id: `9981d2a1dcd29368401ee83c581e29963926e73fd4fbbbcb4e12f58cd22eb96a`; SHA256 ответа: `0c95162310bbb2057f308c42fc29763e7e9be6b97b9dbefc391d36419354fc54`.

#### C05 — 1/2

ARIMAX с внешними признаками допустим; соотношение 80/20 не задаёт хронологию. Нет сезонного baseline, метрики и ясного shift-before-rolling/доступности погоды.

Исходный request_id: `4a69c0ce7d766c402a63729eab9d9b42faebc9406018df00e809a298a2b0af09`; SHA256 ответа: `1d38090d5b09db4e896597526a62af387677a13dadcf176e07ccdfff813de459`.

#### C06 — 1/2

t-SNE допустима как альтернативная визуализация, но не раскрыты отличие от PCA, масштабирование и отсутствие гарантии сохранения классов.

Исходный request_id: `8b81446944a83492ce5143268d69e806765627707ffdafeb418f7b50a34be4aa`; SHA256 ответа: `6f0f87ff5ab75491a0e33edef0143732167447c67197e3e0e498dfcdcbacf8b3`.

#### C07 — 1/2

NB/LR и отказ от уплотнения верны; CV названа, но нет стратификации/macro-F1, обработки дубликатов и vectorizer без утечки.

Исходный request_id: `d84c394adaf081cd67451fe34fab70e18db146930722de55699df44e321faaf2`; SHA256 ответа: `94497481ff7896ea29121c0587aec4d6bc1cb3c812aadb8d3158b056fcd93907`.

#### D01 — методология кода

Проверен дословный дополнительный fenced-код: Pipeline/CV верны, но best_estimator_.score — accuracy, ошибочно названная balanced_accuracy. print(dict) не JSON. Первичный ответ не проходит контракт.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=1, missing_stratification=0, missing_seed=0.

Исходный request_id: `0058f42eccc7863972902b2f2589dc5b5b6d7bd6f8efaa4f2eefa7522e636719`; SHA256 ответа: `b153b3353bd3bb29c235d75b91e6db23798a920b96fdf6e7cad70edf44bd66f1`.

#### D02 — методология кода

Дополнительный код правильно разделяет train/test и Pipeline, но отсутствует импорт Pipeline; primary JSON-контракт также нарушен.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=0, missing_stratification=0, missing_seed=0.

Исходный request_id: `eb552a7819f7b546c161e9c6954c3c0eb8b8b9737b41a0793e02d41ff8b224d5`; SHA256 ответа: `09bd792881e0956a8d325fe13ce64830ed6911051edf092c2e69ca885e3a1222`.

#### D03 — методология кода

Дополнительный код: scaler внутри GridSearchCV, stratify/seed и holdout accuracy верны; print(dict) не JSON, первичный контракт нарушен.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=0, missing_stratification=0, missing_seed=0.

Исходный request_id: `465a17127b5af014b5d4d8100f6eedb0848bc2acdee246632441a5d1cbe3d155`; SHA256 ответа: `d63f9fb33d990f04f873dd61852ee2b54dd7e82fcb4e6d81d8a92717d1cb007c`.

#### D04 — методология кода

Дополнительный код: RF и метрики на holdout корректны; опечатка ключа balance_accuracy — проблема контракта, не формулы. Первичный ответ невалиден.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=0, missing_stratification=0, missing_seed=0.

Исходный request_id: `1a5edcbc7b364c51e0368deb300a72a48bc5be9281a247514c4552a2eed738d2`; SHA256 ответа: `be695fdcc3dacb1e770c6c52858655be8962ecd7edbc4dc48ba4655e1093b18e`.

#### D05 — методология кода

Дополнительный код: train-only PCA/Pipeline и split верны, но отсутствуют импорты Pipeline и json; primary JSON-контракт нарушен.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=0, missing_stratification=0, missing_seed=0.

Исходный request_id: `b123b966b9dfdefa05fdc4b5b0d7931ea94b8b1fc85731f7825e89874c3f8cf3`; SHA256 ответа: `11e87e8f50e8849c68856c4a1f51a625870807e0c5ad5559751ee102d1651872`.

#### D06 — методология кода

В дополнительном фрагменте train-only Pipeline, sqrt(MSE), r2_score и seed корректны; внешняя JSON-оболочка синтаксически нарушена, первичную оценку не повышаем.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=0, missing_stratification=N/A, missing_seed=0.

Исходный request_id: `9727ac378cdf346aca91e8d420615bbc5b9fad067b67101fde48e26cafac6287`; SHA256 ответа: `ca35cdb3f437c2bcc23302d815976a917a67d70f4cde238f1ae3dc6b3e99686e`.

#### D07 — методология кода

Дополнительный код правильно игнорирует y и считает silhouette на scaled X; holdout по условию не нужен. print(dict) не JSON; primary контракт нарушен.

Флаги: leakage=N/A, test_tuning=N/A, train_evaluation=N/A, wrong_metric=0, missing_stratification=N/A, missing_seed=0.

Исходный request_id: `3ec7f0a94ae7be7f1f460a8b21b6f2033b5a2c85784ff8bcfe01df03ffb2ec8a`; SHA256 ответа: `4e93c537e1e86de9876223cff1d6f64a1df4175aa1e5deca30a7989e1781c038`.

#### D08 — методология кода

Дополнительный код сохраняет метки и pos_label=0, no scaling, stratify и seed; округление 3 знака допустимо по числам, но primary контракт нарушен.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=0, missing_stratification=0, missing_seed=0.

Исходный request_id: `9bef8bc1f35205dff0f00098351a197bfb6f76a26863e909631f1f87d8af0196`; SHA256 ответа: `099798dadc56d62c8e4eb2603c92ecd9a568f9835585f8f3f0b1531603232374`.

### DeepSeek

#### B01 — 2/2

PCA без учителя, максимальная дисперсия, разделение классов относится к supervised-методам.

Исходный request_id: `4d24f2d698a4d58fba8b644f75083277f5ce87a9ef99b262325a123b234401db`; SHA256 ответа: `05a8bc54b501f42d5121b5c44d30036847f8840e4a87cb74566f631ea6c437ce`.

#### B02 — 2/2

Корректно раскрыты независимость, корреляции, смещение вероятностей и отдельная проверка калибровки.

Исходный request_id: `7b56ea263730a824ff675c86f77e001e5645021211fd56b9e2ba9d4b6904e728`; SHA256 ответа: `6d394c1984ec89c795f98b56f8dce7f0894767c25602796a83992eae7b4bf908`.

#### B03 — 2/2

Объяснены статистики test, fit scaler только на train и применение к test без переобучения.

Исходный request_id: `2231d3f0581ca74f9cddadbb3697aced2134ea54f9e039e288b7fc5a8b611c54`; SHA256 ответа: `7f35a72a19ac0a091cf878e6a08646aa9ed74d7779dff799a885e47e0e7ca4e6`.

#### B04 — 2/2

Есть ранжирование, порог и prevalence; гарантия precision отвергнута.

Исходный request_id: `bea768804e66f94cf6687a8e20ecff8ffeae2132c82b2cf4f4668ce60c77caa1`; SHA256 ответа: `adb7739a5366491c203657fa2ec89deac800080dbda27990eb6d999263ea22e8`.

#### B05 — 2/2

Верно: рост C увеличивает штраф и ослабляет регуляризацию.

Исходный request_id: `0af68c13f8304ab20e424c5b0d2b685e99e365758b7062c21da9cda833cc6b9e`; SHA256 ответа: `26635f932318f78aade33efbc87aed29b5ba4128608665b28a54fe44384361a5`.

#### B06 — 2/2

Квадраты разностей связывают масштаб с порядком соседей и прогнозом.

Исходный request_id: `305324d0f147a9cc4941d55ad0f5fa1d31347391ff86334ddf1bc09dda6596c7`; SHA256 ответа: `823ae5e366dab81dd39f2411a675735a738f3822b37ada968ea96634c60a644a`.

#### B07 — 2/2

Точно указан выбор признаков на каждом разбиении, корреляция ошибок и снижение дисперсии ансамбля.

Исходный request_id: `3854fefc06b19bb13ac1f157ea130523fc4f1ce3caa3443bdd5b752f96c4b1ab`; SHA256 ответа: `7704547967af4e266f6e28d35c777c50cee6f10189ba07bc72479f180fec2339`.

#### B08 — 2/2

Верны inertia, алгоритм Ллойда, зависимость от инициализации и отсутствие гарантии глобального оптимума.

Исходный request_id: `47a391117d99476832e7e34f0274d94d15abeac4172f403da5b4b408673d588e`; SHA256 ответа: `64b568a1677a00d456b74ce447784ecf48c3bfbd200e4709a2a2c5a69809878c`.

#### C01 — 2/2

Регуляризованная логистическая модель, вложенная стратифицированная CV, preprocessing внутри фолдов и риск нестабильности при p>n раскрыты. Название Pipeline не обязательно при эквивалентной схеме.

Исходный request_id: `ada96aea36245e9ab25d24d7f1d2626429f2b102b343539334d414ac96246b13`; SHA256 ответа: `6f48036aa22aaf74a5b0f82ae8d2040af7e84a8666eb384480adb985bf128754`.

#### C02 — 2/2

Масштаб, k по внутренним метрикам, устойчивость и предметная проверка объяснены; физические режимы отделены от артефактов.

Исходный request_id: `10b411d9461b0bdf19acea10fe86b8d846a9fcf3ed5c5b291cd537abacea2313`; SHA256 ответа: `bf09f1a09417d5d22732dd70c900c813d9ac003f365e1849a1f0f0d30d42804c`.

#### C03 — 1/2

Модели, категории/пропуски и одинаковые фолды разумны, но OOF после подбора на тех же фолдах не является независимой оценкой; внешний CV/нетронутый test явно не отделён. Для CatBoost категориальные пропуски также требуют конкретной обработки.

Исходный request_id: `d6df0cf410d930691c4d0e6a76ecefcc9e2f2a2f78024198029990deaffe5aba`; SHA256 ответа: `54bc111485de9f71c7c271f4b6940f049d04e09322b868a890aeaf74acd72146`.

#### C04 — 2/2

Указаны PR-метрики, константа 99%, стоимость 100FN+2FP, порог на validation/CV и корректный порог 2/102 при калибровке.

Исходный request_id: `8ef3a94793c8cd88a10e2285f9a4e78e996817498a17a9de311aeeb947e408ca`; SHA256 ответа: `aeee218ac2b5fd4ae1d1293c8dabd3c9be1ad96109537f8d4da47e968a00e2af`.

#### C05 — 1/2

Хронология, shift(1), погода-прогноз и fit preprocessing только на train раскрыты подробно; обязательный сезонный наивный baseline и MAE/RMSE пропущены.

Исходный request_id: `44a324ec20e6e275f55dc029ea419d9c77da5e004c84c442afe195624c55dc67`; SHA256 ответа: `e19f2e665152cdf764076f1fab848e588e9c2dda6bdf9faff4238828d7157cf8`.

#### C06 — 1/2

PCA, стандартизация, дисперсия и отсутствие гарантии классов верны; не разобрана защита train/test при использовании преобразования в модели.

Исходный request_id: `ca79676a62aa16a1ef22018716dd1bdfdd584891d997db78283f68b0a78d4cd6`; SHA256 ответа: `9508b029be1fee7b9233e7dbde8dd50e1011323342d2a4f7c436c9cb05fb14f9`.

#### C07 — 1/2

NB, линейные sparse-модели, macro-F1 и holdout разумны; не раскрыты vectorizer только на train и дубликаты. Связка multinomial+liblinear из перечня вариантов некорректна без OvR.

Исходный request_id: `1473badbd0380f1984ee3b0686f0309f53061b2c6b17bb7010afedfb3f4e1db3`; SHA256 ответа: `ff458579a0a158c14be1769f0f2f86d4d6c7fb18270e9f27ef5412e38b7675ed`.

#### D01 — методология кода

Pipeline в CV; C только на train; balanced_accuracy отдельно на test; stratify, seed и JSON корректны.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=0, missing_stratification=0, missing_seed=0.

Исходный request_id: `a6591d827b7573db9e9a2d0b78ab66955a48069ad88d91a0149f007a5dbefbf6`; SHA256 ответа: `013a83b6cc4fb3d8b76c79d4642883e3e3f09c6674417b8b2f1cef7f25b2f920`.

#### D02 — методология кода

Imputer/scaler/LR fit только на train; F1 класса 1 и balanced_accuracy на test; stratify/seed/JSON корректны.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=0, missing_stratification=0, missing_seed=0.

Исходный request_id: `8eb27ac4c588b2884098bd5d800ebecb0f9a928df1e66bdb8b336bf3b6cf5dfb`; SHA256 ответа: `ef787e15d0bce3949765f11a1ae6361a6b5f9d769874409e9fdf611c37ed8df6`.

#### D03 — методология кода

Scaler внутри Pipeline/CV, k только по train; GridSearchCV.score с scoring=accuracy применяется к holdout; JSON корректен.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=0, missing_stratification=0, missing_seed=0.

Исходный request_id: `3da3bfdd0a0668cc5ed08ac36655e42bc0db36640e324e0d585b6df898f896f2`; SHA256 ответа: `e86e0337636d6d3fc2fc7b034b0045d353a14040af412236260ca3a7961cdb6c`.

#### D04 — методология кода

RF только на train, balanced_accuracy и бинарный F1 класса 1 на test; все заданные параметры, stratify и seed соблюдены.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=0, missing_stratification=0, missing_seed=0.

Исходный request_id: `a83f08c013db1f34ae41e875ee3c7ff9ffe444ce70c23cf0a4164bb0cd7add77`; SHA256 ответа: `1b7ca055a9510ff3cc012e038dca4b1c51475d5f3a27f0cacb9fe35adea56b1f`.

#### D05 — методология кода

Scaler и PCA обучаются в train-only Pipeline; holdout accuracy, stratify и seed корректны.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=0, missing_stratification=0, missing_seed=0.

Исходный request_id: `a4c574daa10d6aa585e33aea9756d1e1a7b1ddbaca203e951492ee30ee1c3dbf`; SHA256 ответа: `177d125e426cf0d3d3a527196cf7b8ead440a342837548adf043ad4a6f6236ef`.

#### D06 — методология кода

Pipeline обучен на train; RMSE через sqrt(MSE), R² через r2_score на test; случайный split без stratify соответствует регрессии.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=0, missing_stratification=N/A, missing_seed=0.

Исходный request_id: `f70ca858a81530fd6fbb3a01aeaced51dca5c414d6bcf44b35ac4af240512bc8`; SHA256 ответа: `cc115da8cf48f93cc6467350967d1391e1099aa8610e56ad5dff899a6179af6c`.

#### D07 — методология кода

Описательная кластеризация всех 300 строк соответствует условию; y игнорируется, silhouette по scaled X и clusters; n_init/seed/JSON корректны.

Флаги: leakage=N/A, test_tuning=N/A, train_evaluation=N/A, wrong_metric=0, missing_stratification=N/A, missing_seed=0.

Исходный request_id: `cd032cd86485ff258f75024f500cb59b4859c6d82afd3d0394e89d40a8fc4860`; SHA256 ответа: `08965ece509cfb2feb4d816d35980a48e32c6d5298318c3720b914787e005bb6`.

#### D08 — методология кода

Исходная кодировка классов сохранена, GaussianNB без scaling; holdout balanced_accuracy и F1 pos_label=0, stratify и seed корректны.

Флаги: leakage=0, test_tuning=0, train_evaluation=0, wrong_metric=0, missing_stratification=0, missing_seed=0.

Исходный request_id: `c07910b6e76883e1ba2c9927d74e88042e4fc30b69cbc22f0f766562d721f5f3`; SHA256 ответа: `40982406f8f98dd2b68ad3e46d131ef20893cb587c4de398b15bc66e9f830830`.
