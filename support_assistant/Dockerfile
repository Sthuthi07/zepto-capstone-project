FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY support_assistant /app/support_assistant

ENV MOCK_LLM=1

CMD ["uvicorn", "support_assistant.main:app", "--host", "0.0.0.0", "--port", "7860"]