import json

from fastapi.testclient import TestClient

from support_assistant.main import app


client = TestClient(app)

for q in [
    "What is the delivery fee below INR 149?",
    "What is your favorite color?",
]:
    response = client.post("/ask", json={"query": q})
    print(json.dumps(response.json(), indent=2))