import os
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parent
DOCS = BASE / "docs"
DB = BASE / "chroma_db"
COLLECTION_NAME = "zepto_policies"

def ingest():
    client = chromadb.PersistentClient(path=str(DB))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    model = SentenceTransformer("all-MiniLM-L6-v2")

    ids, texts, metas = [], [], []
    for path in sorted(DOCS.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        ids.append(path.stem)
        texts.append(text)
        metas.append({"document_id": path.stem, "chunk_id": path.stem + "_chunk_0"})

    embeddings = model.encode(texts, normalize_embeddings=True).tolist()
    collection.add(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)
    print(f"Ingested {len(ids)} documents into {COLLECTION_NAME}.")
    return collection

if __name__ == "__main__":
    ingest()
