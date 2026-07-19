"""Split parsed pages into retrieval chunks with page-accurate citations.

Chunks pack whole paragraphs up to a size target, with one paragraph of
overlap between neighbors so a fact straddling a boundary survives in at
least one chunk. Page ranges are tracked per chunk because every answer
cites (publication, pages).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from taxcite.parse import Page

# Fixed constants tuned for the IRS pub corpus, not exposed as config: this
# tool ingests one corpus, so there is no second use case to parameterize for.
TARGET_CHARS = 1600
MIN_CHARS = 200
MAX_PARAGRAPH_CHARS = 400


@dataclass(frozen=True)
class Chunk:
    pub_id: str
    ordinal: int
    first_page: int
    last_page: int
    text: str


@dataclass(frozen=True)
class _Paragraph:
    page: int
    text: str


def chunk_pages(pub_id: str, pages: list[Page]) -> list[Chunk]:
    paragraphs = _split_paragraphs(pages)
    chunks: list[Chunk] = []
    buffer: list[_Paragraph] = []
    size = 0

    for para in paragraphs:
        if buffer and size + len(para.text) > TARGET_CHARS:
            chunks.append(_flush(pub_id, len(chunks), buffer))
            buffer = [buffer[-1]]  # one-paragraph overlap
            size = len(buffer[0].text)
        buffer.append(para)
        size += len(para.text)

    if buffer:
        tail = _flush(pub_id, len(chunks), buffer)
        # a tail that is pure overlap of the previous chunk adds nothing
        if len(chunks) == 0 or len(tail.text) >= MIN_CHARS or len(buffer) > 1:
            chunks.append(tail)
    return chunks


def _split_paragraphs(pages: list[Page]) -> list[_Paragraph]:
    """Split page text into packable units.

    PDF text extraction rarely produces blank-line paragraph breaks, so a
    page often arrives as one block. Oversized blocks are subdivided on
    line boundaries; without this, every page becomes a single chunk and
    the size target is meaningless.
    """
    paragraphs: list[_Paragraph] = []
    for page in pages:
        for block in re.split(r"\n\s*\n", page.text):
            text = block.strip()
            if not text:
                continue
            if len(text) <= MAX_PARAGRAPH_CHARS:
                paragraphs.append(_Paragraph(page=page.number, text=text))
            else:
                paragraphs.extend(
                    _Paragraph(page=page.number, text=piece)
                    for piece in _pack_lines(text)
                )
    return paragraphs


def _pack_lines(block: str) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        if current and size + len(line) > MAX_PARAGRAPH_CHARS:
            pieces.append("\n".join(current))
            current = []
            size = 0
        current.append(line)
        size += len(line)
    if current:
        pieces.append("\n".join(current))
    return pieces


def _flush(pub_id: str, ordinal: int, buffer: list[_Paragraph]) -> Chunk:
    return Chunk(
        pub_id=pub_id,
        ordinal=ordinal,
        first_page=buffer[0].page,
        last_page=buffer[-1].page,
        text="\n\n".join(p.text for p in buffer),
    )
