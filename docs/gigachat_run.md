# GigaChat-2: запуск и анализ

Основной файл: **notebooks/gigachat_analysis.ipynb**. Ноутбук выполнен, Run All читает сохранённые ответы и пересчитывает показатели, без обращений к API и повторного исполнения кода модели.

## Эксперимент

- Запрошена `GigaChat-2` (Lite); сервер вернул `GigaChat-2:2.0.28.2` во всех 344 ответах. Дата: 19 ноября 2026. Пять pilot исключены, 39 main и 300 E (100 на каждый shots=0/5/20).
- System/user полностью совпадают с общими запросами Yandex/Qwen; temperature=0, лимит 4096, новый диалог на задачу, без tools/search. Сервер не возвращает effective temperature; поле оставлено null.
- Прогнозы и конфигурация зафиксированы до генераций коммитом `9b810f0`. Результаты других моделей и baseline уже были известны. Pilot раскрыт коммитом `6e14f62`; ни prompts, ни параметры после него не менялись. Ошибочный P04 не переспросили.
- Первичное H5 остаётся Yandex vs Qwen, как выбрано командой; GigaChat vs Qwen — явно вторичное сравнение. H10/H12 повторены по прежним критериям. Три доступные парные проверки A/B/F: exact McNemar + Holm.
- Применён официальный, ещё поддерживаемый `https://gigachat.devices.sberbank.ru/api/v1`: новый `https://api.giga.chat/v1` дал TLS timeout в данном окружении. [Документация адресов](https://developers.sber.ru/docs/ru/gigachat/api/reference/rest/gigachat-api).

## Расходы и токены

Использована бесплатная квота Lite, покупок не было. Остаток до/после сохранён в results/models/gigachat/gigachat-2-v1/balance_before.json и balance_after.json. Уменьшение равно 60 822 — сумме usage.total_tokens всех 344 ответов. Эквивалент по тарифу пакета Lite 1300 руб./20 млн токенов, включая НДС: 3.95343 руб. Это оценка использованного объёма, не реальное списание денег и не цена приобретения пакета. [Тариф](https://developers.sber.ru/docs/ru/gigachat/tariffs/individual-tariffs).

`prompt_tokens` уже исключает `precached_prompt_tokens`, `total_tokens` содержит тарифицируемый объём. Кэш бесплатен; он и полный объём входа/выхода показаны отдельно. Повторно прибавлять кэш к цене нельзя. Сохранённое стандартное поле журнала input_tokens равно provider prompt_tokens, то есть здесь это вход без кэша. [Правила подсчёта](https://developers.sber.ru/docs/ru/gigachat/guides/counting-tokens).

## Пересчёт

```powershell
python -m venv --system-site-packages .venv
.venv/Scripts/python.exe -m pip install -r requirements-analysis.txt
.venv/Scripts/python.exe -m pip install requests
.venv/Scripts/python.exe scripts/analyze_gigachat.py
.venv/Scripts/python.exe scripts/build_gigachat_notebook.py
.venv/Scripts/python.exe scripts/execute_gigachat_notebook.py
.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_*.py'
```

Фактическое окружение сохранено в analysis/gigachat/environment.json и package-versions.txt. Для анализа ключ, сертификаты и доступ к сети не требуются.

## Генерация и ключ

Локальный `.env.gigachat` (игнорируется Git):

```dotenv
GIGACHAT_CREDENTIALS=<ключ авторизации из личного кабинета>
GIGACHAT_SCOPE=GIGACHAT_API_PERS
```

Можно вместо файла задать переменные среды с этими именами. Ключ и access token не сохраняются в журнал запросов/ответов. Токен обновляется в памяти; токены OAuth в Git отсутствуют.

CA-сертификаты скачаны по ссылкам [официальной инструкции](https://developers.sber.ru/docs/ru/gigachat/certificates) с gu-st.ru в `.cache/gigachat-ca/`, подключены только к requests-сессии. Проверка сертификатов включена; системное хранилище не изменяется. Для другого окружения можно передать `GIGACHAT_CA_BUNDLE_FILE` с путём к доверенному CA bundle.

```powershell
.venv/Scripts/python.exe scripts/prepare_gigachat_ca.py
.venv/Scripts/python.exe gigachat_runner.py --stage pilot
.venv/Scripts/python.exe gigachat_runner.py --stage all --budget-rub 100
```

Полученные ответы пропускаются. Некорректный формат и неправильные решения сохраняются, не перезапрашиваются. На 429 разрешены до трёх попыток с ожиданием. При неизвестном исходе сетевого запроса runner останавливается и сохраняет uncertain; нельзя обходить это удалением журнала ради лучшего ответа. В этом прогоне повторов/uncertain не было.

## Код D: две разные проверки

Все восемь ответов D нарушили общий формат, поэтому первичный end-to-end pass@1=0/8. Эти оценки не изменены. Диагностически извлечены точные подстроки Python из fenced/triple-quoted блоков; offsets и SHA256 записаны в docs/gigachat_code_review.json. Без исправлений запустились 6/8; 2/8 завершились NameError из-за отсутствующего Pipeline. Вывод, JSON и совпадение чисел показаны раздельно в diagnostic_code_execution.json. D01 печатает обычную accuracy под именем balanced_accuracy, хотя число близко к эталону; совпадение чисел не заменяет проверки методологии.

Для воспроизведения диагностики есть scripts/execute_gigachat_code.py; обычный Run All её не выполняет. Код просмотрен технически, запускается в отдельных процессах с очищенным окружением; это не ОС-контейнер. Извлечение — post-hoc диагностика, не расширение разрешённого parser и не новая генерация.

Для окончательной сдачи два участника независимо оценивают B/C/D по формам analysis/gigachat/review/, перемешав ответы разных моделей для слепой проверки. Их оценки, kappa и собственные выводы не подменяются оценками ассистента. Пять исходных автоматических ошибок сохранены в error_examples.json; student_analysis оставлено команде согласно заданию.
