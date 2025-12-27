FROM python:3.14-slim

RUN apt-get update && apt-get install -y \
    libgomp1 \
    libstdc++6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH="/app"

COPY brinicle.*.so ./

COPY ref/ ./ref/

RUN mkdir -p ./app/data/indices

ENV PYTHONPATH="${PYTHONPATH}:/app"

EXPOSE 1984

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:1984/')" || exit 1

# run the FastAPI application. workers must be one
CMD ["uvicorn", "ref.api:app", "--host", "0.0.0.0", "--port", "1984", "--no-access-log", "--workers", "1"]
