# Imob AI

An **AI engineering** project based on a synthetic real-estate use case. Its
focus is technically sound software for AI, machine learning, reproducible
evaluation, semantic search, and source-grounded RAG.

> Status: repository foundation. No dataset, trained model, or API exists yet.

## Technical goal

The project will grow through reviewable increments:

1. a fictional catalog and structured property search as the baseline;
2. semantic search evaluated with retrieval metrics;
3. RAG grounded in retrieved evidence;
4. a message classifier trained only on synthetic data;
5. an API, tests, documentation, and manually published artifacts.

Each increment must include a technical rationale, tests, and evaluation
evidence before the next one starts.

## Local setup

Python 3.9 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## Initial checks

```bash
python -m unittest discover -s tests -v
pytest
ruff check .
```

## Data policy

No real customer or real-estate agency data will be used. Datasets will be
synthetic, documented, and validated before model training.

