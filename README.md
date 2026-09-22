# Zepto AI/ML Engineering Capstone

This repository contains the three required modules:
- `data_pipeline` — scraping, cleaning, currency conversion, SQLite normalization and SQL/pandas querying.
- `analytics` — Titanic profiling, EDA, preprocessing, classification, imbalance comparison, tuning, regression and model persistence.
- `support_assistant` — local embeddings + ChromaDB + LangGraph + FastAPI RAG assistant with deterministic `MOCK_LLM` mode.

## Setup

Create and activate a virtual environment, then install the consolidated requirements:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

### 1. Data pipeline
```bash
cd data_pipeline
python pipeline.py
```

This scrapes the first 5 catalogue pages (normally 100 books), cleans them, converts GBP to INR using the required fixed rate `1 GBP = 105.50 INR`, creates `output/books.db`, executes six SQL queries, saves their outputs, and verifies SQL JOIN vs pandas merge.

### 2. Analytics
```bash
cd analytics
python 01_eda.py
python 02_modeling.py
python verify_model.py
```

The first script loads `sns.load_dataset("titanic")` once and immediately saves `titanic.csv`. If network access is unavailable later, the modeling script uses that committed CSV.

### 3. Support assistant
```bash
cd support_assistant
python ingest.py
uvicorn main:app --reload
```

Then POST to `/ask`, for example:
```bash
curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What is the delivery fee below INR 149?\"}"
```

For a general question:
```bash
curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What is your favorite color?\"}"
```

`MOCK_LLM` is intentionally unset by default. This means the graded deterministic mock path is used. The optional real-LLM path is present in code but requires additional configuration.

## Docker
From `support_assistant`:
```bash
docker build -t zepto-support .
docker run -p 7860:7860 zepto-support
```

## Design decisions

### Data pipeline
The first five catalogue pages provide at least 60 records without requiring category URL discovery. SQLite uses normalized `categories` and `books` tables with a foreign key. Numeric parsing failures are median-imputed; rows with missing required categorical fields are dropped.

### Analytics
EDA cleaning follows the requested missing-value thresholds. Modeling starts with a stratified split before preprocessing. A `ColumnTransformer` inside a scikit-learn `Pipeline` guarantees train-only fitting of imputers, one-hot encoding and scaling. SMOTE is applied only to training data.

### Support assistant
The eight supplied policy documents are ingested as one chunk per document into ChromaDB using `all-MiniLM-L6-v2`. LangGraph routes policy questions to retrieval and general questions directly. Mock mode is deterministic and makes no LLM API call.

## Git workflow requirement

The assignment requires at least one feature branch with two commits merged into `main`. Perform that workflow in your own Git repository rather than fabricating history in the project files.
