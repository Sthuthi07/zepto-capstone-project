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

Example requests:

```json
{"query":"What is the delivery fee below INR 149?"}
```

Expected response shape:

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": ["doc_01"],
  "confidence": 1.0
}
```

General question:

```json
{"query":"What is your favorite color?"}
```

Expected response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## Architecture

```text
8 policy documents
       |
       v
 ingestion.py
       |
       v
all-MiniLM-L6-v2 embeddings
       |
       v
ChromaDB collection: zepto_policies
       |
       v
FastAPI /ask
       |
       v
LangGraph StateGraph
       |
       +--> classify_intent
       |       |
       |       +--> policy_question --> retrieve_and_answer --> JSON
       |       |
       |       +--> general_question --> direct_answer ----------> JSON
       |
       v
Pydantic AskResponse
```

Ingestion loads the eight documents and stores one chunk per document. The Sentence Transformers model creates local embeddings. `retrieve_and_answer` embeds each incoming policy question and asks ChromaDB for the top three cosine-similar chunks. In mock mode the first retrieved chunk is used to build the deterministic answer. `direct_answer` handles unrelated questions without retrieval.

`MOCK_LLM` branches the classification/generation behavior. In the required default mode, classification is keyword based and final answers are deterministic. The retrieval stage always runs for policy questions. The code contains the optional real-LLM branch location and structured prompt template, while the offline path remains the required baseline.

## Docker

From the repository root:

```bash
docker build -f support_assistant/Dockerfile -t zepto-support .
docker run -p 7860:7860 zepto-support
```

The container exposes `POST /ask` on port 7860.
