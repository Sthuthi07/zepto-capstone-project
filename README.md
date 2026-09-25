# Zepto AI/ML Engineering Capstone

This repository contains the three required modules:

* `data_pipeline` — scraping, cleaning, currency conversion, SQLite normalization, and SQL/pandas querying.
* `analytics` — Titanic profiling, EDA, preprocessing, classification, imbalance comparison, tuning, regression, and model persistence.
* `support_assistant` — local embeddings, ChromaDB, LangGraph, and FastAPI RAG assistant with deterministic `MOCK_LLM` mode.

## Setup

Create and activate a virtual environment, then install the consolidated requirements:

```bash
python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

No API key is required for the graded baseline. The Support Assistant uses deterministic local mock behavior when `MOCK_LLM` is at its default value.

## Run

### 1. Data pipeline

Run from the repository root:

```bash
python data_pipeline/pipeline.py
```

On Windows PowerShell:

```powershell
python data_pipeline\pipeline.py
```

This scrapes the first 5 catalogue pages from Books to Scrape, normally producing 100 books. It:

* extracts title, price, star rating, availability, and category
* cleans and validates the scraped fields
* converts GBP to INR using the required fixed rate `1 GBP = 105.50 INR`
* creates the normalized SQLite database at `data_pipeline/output/books.db`
* executes six SQL queries covering filtering, ordering, limiting, distinct values, range/membership filtering, and joins
* saves the SQL query strings and outputs
* compares the SQL JOIN result with the equivalent pandas merge

### 2. Analytics

Run from the repository root:

```bash
python analytics/01_eda.py
python analytics/02_modeling.py
python analytics/verify_model.py
```

On Windows PowerShell:

```powershell
python analytics\01_eda.py
python analytics\02_modeling.py
python analytics\verify_model.py
```

The EDA script loads the Titanic dataset using:

```python
sns.load_dataset("titanic")
```

and immediately saves `analytics/titanic.csv` as the committed offline fallback dataset.

The analytics module performs:

* data cleaning and missing-value handling
* exploratory data analysis
* required visualizations
* correlation analysis
* train/test splitting
* preprocessing with `ColumnTransformer`
* Logistic Regression
* Decision Tree
* Random Forest
* ROC/AUC evaluation
* class-imbalance comparison
* SMOTE on training data
* hyperparameter tuning
* regression evaluation
* residual analysis
* persistence and reload verification of the fitted model pipeline

### 3. Support assistant

Run from the repository root.

First ingest the eight policy documents:

```bash
python -m support_assistant.ingest
```

Then start FastAPI:

```bash
uvicorn support_assistant.main:app --reload
```

The API is available at:

```text
http://127.0.0.1:8000
```

The main endpoint is:

```text
POST /ask
```

### Example 1 — Policy question

Request:

```json
{
  "query": "What is the delivery fee below INR 149?"
}
```

Actual response with the default `MOCK_LLM` behavior:

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pincodes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del...",
  "sources": ["doc_01", "doc_03", "doc_05"],
  "confidence": 1.0
}
```

This query is classified as `policy_question` by the keyword-based intent classifier and routed through the `retrieve_and_answer` node.

### Example 2 — General question

Request:

```json
{
  "query": "What is your favorite color?"
}
```

Actual response with the default `MOCK_LLM` behavior:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

This query is classified as `general_question` and routed through the `direct_answer` node.

No external LLM call is required in the default mock mode.

## Support Assistant Architecture

The Support Assistant implements the following RAG pipeline:

```text
8 policy documents
       |
       v
   Ingestion
       |
       | support_assistant/ingest.py
       v
SentenceTransformer
all-MiniLM-L6-v2
       |
       v
    ChromaDB
zepto_policies collection
       |
       v
    FastAPI
      /ask
       |
       v
  LangGraph StateGraph
       |
       v
 classify_intent
       |
       +-----------------------------+
       |                             |
       v                             v
policy_question                general_question
       |                             |
       v                             v
retrieve_and_answer             direct_answer
       |                             |
       v                             |
Retrieved policy context             |
       |                             |
       +-------------+---------------+
                     |
                     v
              Pydantic response
       answer / sources / confidence
```

### RAG pipeline stages

**1. Ingestion**

`support_assistant/ingest.py` loads all eight supplied policy documents from the `support_assistant/docs/` directory and stores them in the ChromaDB collection.

**2. Embedding**

The local Sentence Transformers model `all-MiniLM-L6-v2` converts the policy document content into embeddings. Incoming policy questions are also embedded for retrieval.

**3. Retrieval**

The `retrieve_and_answer` LangGraph node queries the `zepto_policies` ChromaDB collection and retrieves the most relevant policy chunks for a policy question.

The retrieval stage runs for real in both the default mock mode and the optional real-LLM mode.

**4. Generation**

In the required default `MOCK_LLM` mode, `retrieve_and_answer` builds a deterministic response using the top retrieved chunk:

```text
Based on the retrieved context: ...
```

For unrelated questions, `direct_answer` returns the fixed canned response:

```text
I can only answer questions about Zepto policies right now.
```

### LangGraph routing

The graph contains three named nodes:

* `classify_intent`
* `retrieve_and_answer`
* `direct_answer`

The `classify_intent` node uses a keyword-based heuristic in the default mock mode. Policy-related keywords such as `delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, and `support hours` route the query to `policy_question`.

Unrelated questions are routed to `general_question`.

The conditional edge then sends:

```text
policy_question → retrieve_and_answer
general_question → direct_answer
```

No LLM call is made for either route in the default mock mode.

### Optional real-LLM path

The project also contains the optional `MOCK_LLM=0` extension.

The structured prompt template in `support_assistant/prompt.py` contains the required prompt skeleton components, negative constraint, and few-shot example. The optional real-LLM path can use this structured prompt for classification/generation.

The retry-on-failure and Pydantic response-validation logic is also present in the optional real-LLM path.

The graded baseline does not require an external API key.

## Response schema

The FastAPI response is validated using Pydantic and follows this structure:

```json
{
  "answer": "string",
  "sources": ["string"],
  "confidence": 1.0
}
```

## Docker

The Dockerfile is located at:

```text
support_assistant/Dockerfile
```

Build the image **from the repository root**:

```bash
docker build -f support_assistant/Dockerfile -t zepto-support .
```

Run the container:

```bash
docker run -p 7860:7860 zepto-support
```
