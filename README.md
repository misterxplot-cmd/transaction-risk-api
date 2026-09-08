# Transaction Risk API

Учебный сервис на FastAPI, который загружает обученную модель, проверяет входные данные и возвращает класс риска транзакции. Поддерживаются одиночные запросы и пакеты до 100 объектов.

Этот репозиторий демонстрирует Python-бэкенд вокруг ML-модели: схемы Pydantic, жизненный цикл приложения, обработку ошибок и интеграционные тесты. Код подготовлен с помощью ИИ и проверен локально.

## Два режима

| Режим | Источник модели | Назначение |
|---|---|---|
| `synthetic_demo` | `train_demo.py`, искусственные данные | Самостоятельный быстрый запуск сервиса |
| `metaverse_educational` | Артефакт `behavior_only` из проекта Transaction Risk ML | Связь обучения и сервиса |

Режим указан в каждом ответе API. Значения `class_scores` — оценки классификатора, а не калиброванные вероятности реального мошенничества. При отсутствии модели сервис возвращает 503 и не подменяет результат случайным ответом.

## Быстрый запуск

Проверено с Python 3.12. Команды выполняются из папки репозитория.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python train_demo.py
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Откройте http://127.0.0.1:8000/docs. Раскройте `POST /predict`, нажмите **Try it out**, вставьте пример из `examples/transaction.json` и нажмите **Execute**.

На Linux/macOS замените `.venv\Scripts\python` на `.venv/bin/python`.

Пример запроса из другого окна PowerShell:

```powershell
$body = Get-Content examples/transaction.json -Raw
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body $body
```

Реальные ответы контрольного запуска сохранены в `examples/response_synthetic_demo.json` и `examples/response_metaverse_educational.json`.

## Модель из первого проекта

Сначала обучите `behavior_only` в Transaction Risk ML. Если два репозитория находятся рядом, перед запуском сервера укажите путь:

```powershell
$env:MODEL_PATH = '../transaction-risk-ml/artifacts/model.joblib'
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Версии scikit-learn при обучении и обслуживании должны совпадать. Сервис проверяет схему входов, имена классов и версию библиотеки. Загружайте только собственные доверенные артефакты: формат joblib основан на pickle.

## Маршруты

| Метод и путь | Поведение |
|---|---|
| `GET /health` | Приложение запущено |
| `GET /ready` | Модель готова, иначе 503 |
| `GET /model` | Метаданные загруженной модели |
| `POST /predict` | Одна транзакция |
| `POST /predict/batch` | Объект с массивом `transactions`, от 1 до 100 элементов |
| `GET /docs` | Интерактивная документация |

Отрицательные суммы, час вне 0–23, лишние поля и некорректные пакеты получают 422. Поля `risk_score`, `anomaly` и `transaction_type` не принимаются. Неизвестные категории допускаются обученным OneHotEncoder.

## Тесты

```powershell
.venv\Scripts\python -m pytest -q
```

Тесты проверяют согласованность одиночного и пакетного ответа, сумму оценок классов, отклонение некорректных данных, поведение без модели и доступность OpenAPI. Дополнительно проведён локальный интеграционный запуск с артефактом первого проекта.

## Ограничения

Это локальный учебный сервис без авторизации, очереди задач и мониторинга. Он по умолчанию слушает только 127.0.0.1. Обучение выполняется отдельным скриптом, а не во время HTTP-запроса. Модельные файлы и внешние данные не включаются в Git.

[Разбор архитектуры](EXPLAIN.md) · [Документация FastAPI](https://fastapi.tiangolo.com/tutorial/)
