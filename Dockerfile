# ====== Stage 1: Build deps layer ======
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.2.1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
# Copier uniquement les fichiers de conf pour optimiser le cache
COPY pyproject.toml poetry.lock* ./

# Installer les dépendances (pas le code)
RUN poetry install --no-interaction --no-ansi --only main --no-root

# ====== Stage 2: Runtime ======
FROM python:3.11-slim AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
# Copier les deps installées depuis le builder
COPY --from=builder /usr/local /usr/local

# Copier le code de l’application
COPY . .

EXPOSE 8003
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8003/').read()" || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]