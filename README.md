# LLM_Experiment

Серия опытов с языковыми моделями. Каждый опыт лежит в `experiments/NN_name/`
(скрипт `run.py` + результаты `results.json` / `results.md`).

## Запуск

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python experiments/01_tokenization/run.py
```

## Опыты

1. [Токенизация EN / RU / RU-латиницей / KK](experiments/01_tokenization/results.md)
2. [Случайность: «Придумай название для кофейни» ×3](experiments/02_randomness/results.md)
