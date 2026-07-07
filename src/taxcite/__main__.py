"""CLI: python -m taxcite [ingest|eval|serve]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from taxcite.chunk import chunk_pages
from taxcite.fetch import fetch_publication
from taxcite.manifest import CORPUS, get_publication
from taxcite.parse import parse_pdf


def cmd_ingest(pub_ids: list[str]) -> int:
    from taxcite import db
    from taxcite.embed import embed_texts

    pubs = [get_publication(p) for p in pub_ids] if pub_ids else list(CORPUS)
    conn = db.get_connection()
    try:
        db.run_migration(conn)
        for pub in pubs:
            path = fetch_publication(pub)
            pages = parse_pdf(path)
            chunks = chunk_pages(pub.pub_id, pages)
            texts = [c.text for c in chunks]
            embeddings = embed_texts(texts)
            for chunk, embedding in zip(chunks, embeddings):
                db.upsert_chunk(conn, chunk, embedding)
            db.prune_chunks(conn, pub.pub_id, len(chunks))
            total_chars = sum(len(c.text) for c in chunks)
            print(
                f"{pub.pub_id:>6}  {len(pages):>4} pages  {len(chunks):>5} chunks  "
                f"{total_chars // max(len(chunks), 1):>5} avg chars  {pub.title}"
            )
    finally:
        conn.close()
    return 0


def cmd_eval(dataset: str, out: str) -> int:
    from taxcite.eval_harness import run_eval

    run_eval(dataset_path=Path(dataset), report_path=Path(out))
    return 0


def cmd_serve(host: str, port: int) -> int:
    import uvicorn

    uvicorn.run("taxcite.server:app", host=host, port=port, reload=False)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="taxcite")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="fetch, parse, chunk, and embed publications")
    ingest_p.add_argument("pubs", nargs="*", help="pub ids (default: whole corpus)")

    eval_p = sub.add_parser("eval", help="run Ragas eval harness and write report")
    eval_p.add_argument("--dataset", default="eval/dataset.jsonl")
    eval_p.add_argument("--out", default="eval/report.json")

    serve_p = sub.add_parser("serve", help="start the FastAPI server")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "ingest":
        return cmd_ingest(args.pubs)
    if args.command == "eval":
        return cmd_eval(args.dataset, args.out)
    if args.command == "serve":
        return cmd_serve(args.host, args.port)
    return 1


if __name__ == "__main__":
    sys.exit(main())
