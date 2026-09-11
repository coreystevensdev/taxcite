"""Extract per-page text from publication PDFs."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

# A gutter has to be this much emptier than the page's typical word coverage
# before it counts as a column break rather than ordinary spacing.
_GUTTER_MAX_DENSITY = 0.45

# Below this, a page is a cover, a figure, or a stub, and column detection reads
# noise as structure.
_MIN_WORDS_FOR_COLUMNS = 40

# IRS pubs run to three columns at most; the recursion guard keeps a pathological
# density profile from splitting a page into slivers.
_MAX_COLUMN_DEPTH = 2


@dataclass(frozen=True)
class Page:
    number: int
    text: str


def _gutter_x(page, x0: float, x1: float) -> float | None:
    """Find the column break between x0 and x1, or None if the band is one column.

    Works off word coverage rather than whitespace. IRS pages rarely have a clean
    empty stripe between columns: headers, rules, and tables span the gutter, so a
    run of zero-coverage pixels finds only the page margins. What does show up is a
    trough, coverage dropping to roughly a third of typical right where the columns
    meet.
    """
    words = [w for w in page.extract_words() if x0 <= w["x0"] < x1]
    if len(words) < _MIN_WORDS_FOR_COLUMNS:
        return None

    width = int(x1 - x0) + 1
    coverage = [0] * width
    for w in words:
        lo = max(0, int(w["x0"] - x0))
        hi = min(width, int(w["x1"] - x0) + 1)
        for x in range(lo, hi):
            coverage[x] += 1

    inked = [x for x in range(width) if coverage[x] > 0]
    if not inked:
        return None
    left, right = inked[0], inked[-1]
    typical = statistics.median([coverage[x] for x in range(left, right + 1) if coverage[x] > 0])

    # Only the middle half. A trough near either edge is a margin, not a gutter.
    quarter = (right - left) // 4
    search = range(left + quarter, right - quarter + 1)
    if len(search) < 2:
        return None

    candidate = min(search, key=lambda x: coverage[x])
    if coverage[candidate] >= typical * _GUTTER_MAX_DENSITY:
        return None
    return x0 + candidate


def _columns(page, x0: float, x1: float, depth: int = 0) -> list[str]:
    """Text of each column between x0 and x1, left to right."""
    gutter = _gutter_x(page, x0, x1) if depth < _MAX_COLUMN_DEPTH else None
    if gutter is None:
        text = page.crop((x0, 0, x1, page.height)).extract_text() or ""
        return [text.strip()] if text.strip() else []
    return _columns(page, x0, gutter, depth + 1) + _columns(page, gutter, x1, depth + 1)


def parse_pdf(path: Path) -> list[Page]:
    """Extract text page by page, preserving 1-based page numbers for citations.

    IRS pubs render body text in two or three columns, and pdfplumber's default
    ordering reads straight across the page. That zippers the columns together a
    line at a time: "Home destroyed. You may be able to continue treating" comes
    back joined to "Example. Sasha and Harper Smith sold their". Every chunk built
    from that text is two half-sentences from unrelated passages, which caps
    retrieval quality no matter how good the embeddings are.

    So each column is cropped and extracted on its own, then concatenated in
    reading order. Table fidelity is still a known weak spot, tracked separately.
    """
    pages: list[Page] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            columns = _columns(page, 0.0, float(page.width))
            pages.append(Page(number=i, text="\n\n".join(columns).strip()))
    return pages
