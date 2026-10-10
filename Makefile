.PHONY: install test ingest eval serve docker

install:
	pip install -e ".[all,dev]"

test:
	pytest -q

ingest:
	eka ingest data/sample_docs --reset

eval: ingest
	eka eval --cases eval/cases.jsonl --report reports

serve:
	eka serve --port 8000

docker:
	docker compose up --build
