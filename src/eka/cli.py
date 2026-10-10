"""Command line interface.

    eka ingest data/sample_docs --reset
    eka ask "How many days of annual leave do I get?" --department HR
    eka eval --cases eval/cases.jsonl --report reports
    eka serve --port 8000
    eka setup-mongo            # create Atlas Vector Search + Atlas Search indexes
"""

from __future__ import annotations

import argparse
import json
import sys

from .models import SearchFilters


def _pipeline():
    from .pipeline import RAGPipeline

    return RAGPipeline()


def cmd_ingest(args) -> int:
    rag = _pipeline()
    if args.reset:
        rag.store.clear()
    stats = rag.ingest(args.path)
    print(f"Ingested {stats.documents} documents -> {stats.chunks} chunks (index now has {stats.total_chunks}).")
    return 0


def cmd_ask(args) -> int:
    rag = _pipeline()
    filters = SearchFilters(department=args.department, doc_type=args.doc_type)
    if args.stream:
        for event in rag.stream(args.question, filters):
            if event["type"] == "token":
                print(event["text"], end="", flush=True)
            else:
                print()
                for c in event["citations"]:
                    print(f"  [{c['index']}] {c['title']} ({c['source']})")
        return 0
    ans = rag.ask(args.question, filters)
    if args.json:
        print(json.dumps(ans.model_dump(mode="json"), indent=2))
        return 0
    print(ans.answer)
    for c in ans.citations:
        print(f"  [{c.index}] {c.title} ({c.source})")
    return 0


def cmd_eval(args) -> int:
    from .evaluation import run_evaluation

    rag = _pipeline()
    if rag.store.count() == 0:
        print("Index is empty. Run `eka ingest data/sample_docs` first.", file=sys.stderr)
        return 1
    out = run_evaluation(rag, args.cases, args.report)
    summary = out["summary"]
    width = max(len(k) for k in summary)
    for key, value in summary.items():
        print(f"{key.ljust(width)}  {value}")
    if args.report:
        print(f"\nReport written to {args.report}/eval_report.md")
    if args.min_accuracy is not None and (summary["answer_accuracy"] or 0) < args.min_accuracy:
        print(f"answer_accuracy below threshold {args.min_accuracy}", file=sys.stderr)
        return 2
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("eka.api:app_factory", factory=True, host=args.host, port=args.port, reload=False)
    return 0


def cmd_setup_mongo(args) -> int:
    from .config import get_settings
    from .embeddings import build_embedder
    from .stores import MongoAtlasStore

    settings = get_settings()
    store = MongoAtlasStore(
        settings.mongo_uri, settings.mongo_db, settings.mongo_collection,
        settings.mongo_vector_index, settings.mongo_text_index,
    )
    store.ensure_indexes(build_embedder(settings).dim)
    print("Search indexes requested. Atlas builds them in the background (usually under a minute).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eka", description="Enterprise Knowledge Assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="index a file or folder")
    p.add_argument("path")
    p.add_argument("--reset", action="store_true", help="clear the index first")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="ask a question")
    p.add_argument("question")
    p.add_argument("--department")
    p.add_argument("--doc-type")
    p.add_argument("--stream", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="run the evaluation suite")
    p.add_argument("--cases", default="eval/cases.jsonl")
    p.add_argument("--report", default="reports")
    p.add_argument("--min-accuracy", type=float, default=None, help="exit non-zero below this answer accuracy")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("serve", help="run the API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("setup-mongo", help="create MongoDB Atlas search indexes")
    p.set_defaults(func=cmd_setup_mongo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
