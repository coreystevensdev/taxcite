"""Column-aware PDF extraction.

These run against a real IRS publication rather than a synthetic fixture, because
the bug they cover only exists in real two-column layouts and a hand-built PDF
would encode whatever assumption the parser already makes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taxcite.parse import _columns, _gutter_x, parse_pdf

PUB = Path(__file__).parent.parent / "data" / "raw" / "p936.pdf"

pytestmark = pytest.mark.skipif(not PUB.exists(), reason="p936.pdf not cached locally")


def test_parses_every_page_with_1_based_numbers():
    pages = parse_pdf(PUB)
    assert len(pages) > 10
    assert [p.number for p in pages] == list(range(1, len(pages) + 1))


def test_body_text_is_not_two_columns_zippered_together():
    """The original failure: pdfplumber reads across the page, so a line came back
    as the left column's sentence joined to the right column's unrelated one."""
    page = parse_pdf(PUB)[4]
    assert "Home destroyed. You may be able to continue treating" in page.text
    # the right column's text at that same vertical position, which used to be
    # appended onto the line above
    assert "Home destroyed. You may be able to continue treating Example." not in page.text


def test_sentences_run_on_within_a_column():
    page = parse_pdf(PUB)[4]
    assert (
        "your home as a qualified home even after it is destroyed in\n"
        "a fire, storm, tornado, earthquake, or other casualty."
    ) in page.text


def test_detects_the_gutter_on_a_two_column_page():
    import pdfplumber

    with pdfplumber.open(PUB) as pdf:
        page = pdf.pages[4]
        gutter = _gutter_x(page, 0.0, float(page.width))
        assert gutter is not None
        # near the middle, not out at a margin
        assert 0.4 < gutter / page.width < 0.6


def test_reports_one_column_when_there_is_no_gutter():
    """A page of full-width tables or a cover has no column break, and inventing
    one would split rows down the middle."""
    import pdfplumber

    with pdfplumber.open(PUB) as pdf:
        single = [
            n
            for n, page in enumerate(pdf.pages)
            if _gutter_x(page, 0.0, float(page.width)) is None
        ]
        assert single, "expected at least one single-column page in this pub"
        page = pdf.pages[single[0]]
        assert len(_columns(page, 0.0, float(page.width))) <= 1
