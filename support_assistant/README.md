# Support Assistant

## Required baseline

The application defaults to `MOCK_LLM=1`. This is the deterministic graded path and does not call a cloud LLM.

Before starting the API, build the local ChromaDB index:

```bash
python ingest.py
```

Then from the repository root:

```bash
uvicorn support_assistant.main:app --reload
```

The FastAPI application exposes:

```text
POST /ask
```

## Example call transcripts

The following examples were run with `MOCK_LLM` left at its default value.

### Policy question

Request:

```json
{
  "query": "What is the delivery fee below INR 149?"
}
```

Response:

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pincodes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del...",
  "sources": ["doc_01", "doc_03", "doc_05"],
  "confidence": 1.0
}
```

This query is classified as `policy_question` by the keyword-based intent classifier and routed through the `retrieve_and_answer` node.

### General question

Request:

```json
{
  "query": "What is your favorite color?"
}
```

Response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

This query is classified as `general_question` and routed through the `direct_answer` node.

No external LLM call is made for either example in the default mock mode.

## Response schema

All responses are validated using the Pydantic `AskResponse` schema:

```json
{
  "answer": "string",
  "sources": ["string"],
  "confidence": 1.0
}
```

## Architecture

```text
8 policy documents
       |
       v
   INGESTION
support_assistant/ingest.py
       |
       v
   EMBEDDING
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
   RETRIEVAL                         |
       |                             |
       +-------------+---------------+
                     |
                     v
                GENERATION
                     |
                     v
             Pydantic AskResponse
```

### RAG pipeline stages

**1. Ingestion**

`support_assistant/ingest.py` loads all eight supplied policy documents from the `docs/` directory and stores their content in the ChromaDB collection.

One chunk is stored per policy document.

**2. Embedding**

The Sentence Transformers model `all-MiniLM-L6-v2` creates local embeddings for the policy documents and incoming policy questions.

**3. Retrieval**

The `retrieve_and_answer` LangGraph node embeds each incoming policy question and queries the `zepto_policies` ChromaDB collection for the most relevant chunks.

The retrieval stage runs for real for policy questions in both the default mock mode and the optional real-LLM mode.

**4. Generation**

In the required default `MOCK_LLM=1` mode, `retrieve_and_answer` uses the top retrieved chunk to construct the deterministic response:

```text
Based on the retrieved context: ...
```

For unrelated questions, `direct_answer` returns the fixed canned response:

```text
I can only answer questions about Zepto policies right now.
```

## LangGraph routing

The LangGraph graph contains three named nodes:

* `classify_intent`
* `retrieve_and_answer`
* `direct_answer`

The `classify_intent` node uses a keyword-based heuristic in the default mock mode.

Policy-related keywords include:

* `delivery`
* `return`
* `refund`
* `membership`
* `tracking`
* `cancel`
* `gift card`
* `support hours`

A policy-style query is routed to:

```text
policy_question → retrieve_and_answer
```

An unrelated query is routed to:

```text
general_question → direct_answer
```

The two example calls above demonstrate both conditional routes.

No external LLM call is made by the intent classifier in the default mock mode.

## MOCK_LLM behavior

With `MOCK_LLM=1`, classification is performed using the deterministic keyword heuristic and final answers are generated deterministically.

The retrieval stage still runs normally for policy questions.

The optional `MOCK_LLM=0` path contains the real-LLM integration point and structured prompt template.

The structured prompt template in `support_assistant/prompt.py` contains the required prompt skeleton components, negative constraint, and few-shot example for the optional real-LLM path.

The optional real-LLM path also contains retry-on-failure handling and Pydantic response validation.

The graded baseline does not require an external API key.

## Docker

From the repository root:

```bash
docker build -f support_assistant/Dockerfile -t zepto-support .
```

Run the container:

```bash
docker run -p 7860:7860 zepto-suppor
```
