# Локальная часть ЛР3

Основной файл для просмотра: `notebooks/local_analysis.ipynb`. Он выполнен и содержит таблицы, графики, все ответы C/D, сохранённые результаты запуска кода и проверки H5/H10/H12. Run All выполняет только анализ сохранённых данных.

Модель: [официальная Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF), Q4_K_M, ревизия `bc640142c66e1fdd12af0bd68f40445458f3869b`. SHA256: `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`. Размер 2 497 280 256 байт. Веса не включены в Git.

Модель запускается через установленный `llama-cpp-python` с CUDA на RTX 4080 SUPER. Общие запросы, system prompt, temperature=0 и max_tokens=4096 совпадают с Yandex. Режим без рассуждений задаётся пустым think-блоком в chat template Qwen3. В [карточке модели](https://huggingface.co/Qwen/Qwen3-4B-GGUF#best-practices) рекомендованы другие sampling-параметры для обычного использования; greedy выбран ради общего протокола и может ухудшать качество. Параметры не менялись после просмотра основных ответов.

## Пересчёт анализа

В текущем рабочем клоне используется `.venv/Scripts/python.exe` с numpy 2.3.5, scipy 1.16.3 и sklearn 1.7.2. Окружение создано с доступом к системным пакетам, чтобы использовать уже установленную CUDA-сборку llama-cpp-python; scipy/sklearn установлены локально в venv. Системные версии не заменяются.

```powershell
python -m venv --system-site-packages .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install matplotlib==3.10.8 psutil==5.9.5
.venv/Scripts/python.exe scripts/analyze_local.py
.venv/Scripts/python.exe scripts/build_local_notebook.py
.venv/Scripts/python.exe scripts/execute_local_notebook.py
.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_*.py'
```

Точные фактические версии сохранены в `analysis/local/environment.json`; полный список установленных версий (`pip list --format=freeze`) в `analysis/local/package-versions.txt`. Обычный `pip freeze` падает на устаревшей editable-ссылке стороннего системного проекта; `pip list` позволяет записать версии без обращения к его Git. Сторонние проекты не используются в эксперименте. Для анализа весов и GPU не требуется. Скрипт исполнения ноутбука выбирает именно Python, которым вызван, без изменения глобального kernelspec.

## Повтор генерации

```powershell
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('Qwen/Qwen3-4B-GGUF','Qwen3-4B-Q4_K_M.gguf',revision='bc640142c66e1fdd12af0bd68f40445458f3869b',local_dir='../models'))"
.venv/Scripts/python.exe local_runner.py --model-path ../models/Qwen3-4B-Q4_K_M.gguf --stage pilot
.venv/Scripts/python.exe scripts/run_local_monitored.py --model-path ../models/Qwen3-4B-Q4_K_M.gguf
```

Существующие 344 ответа пропускаются. Результаты первого содержательного ответа не перезапрашиваются. Если журнал оборван после started, автоматическое продолжение остановится. Не удалять журнал, чтобы получить более удобный результат; новый независимый эксперимент требует отдельной регистрации и каталога.

Контрольные суммы банка проверяются перед запуском. Общая функция `prepare` импортирована из Yandex runner, поэтому правила банка и parser едины. Локальный runner импортирует только функцию подготовки, не обращается к API Yandex и не читает ключи.

Параметры и прогнозы зарегистрированы коммитом `16950bc` до любых локальных генераций. Pilot: 4/5 ответов нужного типа, P04 содержит число вместо кода. До main коммитом `e508e14` раскрыто изменение допуска пилота: проверяется получение всех пяти ответов, а ошибки типа не исправляются и не переспрашиваются. Тексты и параметры остались прежними.

## Данные и ограничения

- `results/models/local/qwen3-4b-q4-v1/events.jsonl`: исходные запросы, ответы, tokens, latency, UTC и commit регистрации.
- `analysis/local/`: метрики, парное сравнение с Yandex, bootstrap CI, ECE, предсказания E, графики, provenance, окружение и мониторинг ресурсов.
- `analysis/local/code_execution/`: шесть извлечённых без правок строк кода, stdout/stderr. Два других ответа не предоставили код в допустимом формате. Запускаемость, JSON stdout и совпадение метрик считаются раздельно.
- `docs/local_code_review.json`: технический просмотр и hashes до исполнения. Это не независимая оценка участников. Код выполнялся в отдельных процессах с очищенным окружением, без сетевых/файловых операций в просмотренных исходниках; это не ОС-контейнер.
- `analysis/local/hardware.json`: пиковый RSS процесса и использованная память всего GPU, включая рабочий стол; выборка с интервалом около секунды для классификации. Не выдаётся за точный максимум памяти процесса GPU или профиль первых pilot/main.
- Цена API локально равна нулю; полная себестоимость не измерена. График API cost не сравнивает электричество и амортизацию с облачным тарифом.

Для окончательной сдачи два участника должны независимо заполнить формы `analysis/local/review/reviewer_1.csv` и `reviewer_2.csv`, перемешав ответы разных моделей для слепой проверки. Анализ не заполняет их вымышленными оценками. Финальные выводы отчёта и причины пяти ошибок команда формулирует самостоятельно, как требует PDF. Технические данные и расчётные статусы гипотез уже готовы.
