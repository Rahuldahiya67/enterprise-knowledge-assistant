FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[all]"

COPY data/sample_docs ./data/sample_docs
COPY eval ./eval

# Build the demo index at image build time so the container answers immediately.
RUN eka ingest data/sample_docs --reset

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["eka", "serve", "--host", "0.0.0.0", "--port", "8000"]
