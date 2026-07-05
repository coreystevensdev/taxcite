"""CLI for corpus operations: python -m taxcite ingest p501 [p590a ...]"""

from __future__ import annotations

import argparse
import sys

from taxcite.chunk import chunk_pages
from taxcite.fetch import fetch_publication
from taxcite.manifest import CORPUS, get_publication
from taxcite.parse import parse_pdf


def cmd_ingest(pub_ids: list[str]) -> int:
    pubs = [get_publication(p) for p in pub_ids] if pub_ids else list(CORPUS)
    for pub in pubs:
        path = fetch_publication(pub)
        pages = parse_pdf(path)
        chunks = chunk_pages(pub.pub_id, pages)
        total_chars = sum(len(c.text) for c in chunks)
        print(
            f"{pub.pub_id:>6}  {len(pages):>4} pages  {len(chunks):>5} chunks  "
            f"{total_chars // max(len(chunks), 1):>5} avg chars  {pub.title}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="taxcite")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="fetch, parse, and chunk publications")
    ingest.add_argument("pubs", nargs="*", help="pub ids (default: whole corpus)")
    args = parser.parse_args()

    if args.command == "ingest":
        return cmd_ingest(args.pubs)
    return 1


if __name__ == "__main__":
    sys.exit(main())
