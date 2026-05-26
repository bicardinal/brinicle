# FROM python:3.14-slim
FROM mirror2.chabokan.net/python:3.14-slim-trixie

RUN set -eux; \
    rm -f /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list; \
    rm -rf /var/lib/apt/lists/*; \
    printf '%s\n' \
      'Acquire::Check-Valid-Until "false";' \
      'Acquire::Retries "3";' \
      'Acquire::http::No-Cache "true";' \
      'Acquire::https::No-Cache "true";' \
      > /etc/apt/apt.conf.d/99mirror-workarounds; \
    printf '%s\n' \
      'deb https://repo.abrha.net/debian trixie main contrib non-free' \
      'deb https://repo.abrha.net/debian trixie-updates main contrib non-free' \
      'deb https://repo.abrha.net/debian-security trixie-security main contrib non-free' \
      > /etc/apt/sources.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        libstdc++6 \
        curl; \
    rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install -y \
#     libgomp1 \
#     libstdc++6 \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH="/app"

COPY brinicle/ ./brinicle/

RUN mkdir -p ./app/data/indices

ENV PYTHONPATH="${PYTHONPATH}:/app"

EXPOSE 1984

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:1984/')" || exit 1

# run the FastAPI application. workers must be one
# CMD ["uvicorn", "brinicle.ref.api:app", "--host", "0.0.0.0", "--port", "1984", "--no-access-log", "--workers", "1"]
CMD ["uvicorn", "brinicle.ref.item_search_api:app", "--host", "0.0.0.0", "--port", "1984", "--no-access-log", "--workers", "1"]

# ----------------------------------
