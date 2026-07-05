"""Extract per-page text from publication PDFs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pdfplumber


@dataclass(frozen=True)
class Page:
    number: int
    text: str


def parse_pdf(path: Path) -> list[Page]:
    """Extract text page by page, preserving 1-based page numbers for citations.

    IRS pubs render body text in two or three columns. pdfplumber's default
    layout ordering handles the column flow well enough for chunking; table
    fidelity is a known weak spot tracked for a dedicated pass.
    """
    pages: list[Page] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append(Page(number=i, text=text.strip()))
    return pages
