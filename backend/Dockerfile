FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so Docker caches this layer —
# it only re-runs when requirements.txt actually changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the backend code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
