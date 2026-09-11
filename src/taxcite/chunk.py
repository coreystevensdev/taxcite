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
                    for piece in _pack_sentences(text)
                )
    return paragraphs


# A period only ends a sentence when what follows looks like a new one. IRS text is
# dense with "U.S.", "Pub. 501", "No." and "e.g.", and treating those as breaks
# scatters fragments through the corpus.
_ABBREVIATIONS = r"(?<!\bU\.S)(?<!\bPub)(?<!\bNo)(?<!\bSec)(?<!\be\.g)(?<!\bi\.e)(?<!\betc)(?<!\bMr)(?<!\bMrs)(?<!\bDr)"
_SENTENCE_BREAK = re.compile(rf"{_ABBREVIATIONS}(?<=[.?!])\s+(?=[A-Z(\u201c\"])")


def _reflow(block: str) -> str:
    """Undo PDF line wrapping so sentences are continuous again.

    A PDF carries hard line breaks wherever the text happened to wrap, including
    mid-word: "de-\nduction" is one word, not a hyphenated compound. Packing on
    those breaks is what left 55% of chunks starting on a lowercase letter and 77%
    ending without terminal punctuation.
    """
    block = re.sub(r"(\w)-\n(\w)", r"\1\2", block)
    return re.sub(r"\s*\n\s*", " ", block).strip()


def _pack_sentences(block: str) -> list[str]:
    """Subdivide an oversized block on sentence boundaries, not line boundaries.

    A sentence longer than the target on its own still becomes its own piece:
    splitting inside one puts the reader back where this started.
    """
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in _SENTENCE_BREAK.split(_reflow(block)):
        sentence = sentence.strip()
        if not sentence:
            continue
        if current and size + len(sentence) > MAX_PARAGRAPH_CHARS:
            pieces.append(" ".join(current))
            current = []
            size = 0
        current.append(sentence)
        size += len(sentence)
    if current:
        pieces.append(" ".join(current))

    # Sentence detection needs a capital after the period. Tables, index runs and
    # list fragments have no sentences to find, and without a backstop the whole
    # block comes back as one piece and the size target stops meaning anything.
    # That is what the line packing this replaced was guarding against.
    out: list[str] = []
    for piece in pieces:
        out.extend([piece] if len(piece) <= TARGET_CHARS else _pack_words(piece))
    return out


def _pack_words(piece: str) -> list[str]:
    """Last resort for a block with no sentence boundaries: split between words.

    Still never mid-word, which is the thing that made fragments unreadable.
    """
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for word in piece.split():
        if current and size + len(word) + 1 > MAX_PARAGRAPH_CHARS:
            pieces.append(" ".join(current))
            current = []
            size = 0
        current.append(word)
        size += len(word) + 1
    if current:
        pieces.append(" ".join(current))
    return pieces


def _flush(pub_id: str, ordinal: int, buffer: list[_Paragraph]) -> Chunk:
    return Chunk(
        pub_id=pub_id,
        ordinal=ordinal,
        first_page=buffer[0].page,
        last_page=buffer[-1].page,
        text="\n\n".join(p.text for p in buffer),
    )
