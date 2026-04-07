FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir openai pydantic httpx

# Copy server code and shared models
COPY . /app/
COPY ../models.py /app/models.py
COPY ../client.py /app/client.py
COPY ../openenv.yaml /app/openenv.yaml

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]
