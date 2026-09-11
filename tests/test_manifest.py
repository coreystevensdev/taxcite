"""The corpus is addressed by tax year, not by whatever is current."""

from __future__ import annotations

import importlib

from taxcite import manifest


def test_urls_are_pinned_to_a_year_not_the_current_revision():
    pub = manifest.CORPUS[0]
    assert "/irs-prior/" in pub.url
    assert f"--{manifest.TAX_YEAR}.pdf" in pub.url
    # the unversioned path is what silently changed under the dataset
    assert "/irs-pdf/" not in pub.url


def test_cache_filename_carries_the_year():
    """Without this, switching TAX_YEAR reuses the other year's download and the
    corpus quietly disagrees with the URL it claims to come from."""
    pub = manifest.CORPUS[0]
    assert pub.filename == f"{pub.pub_id}--{manifest.TAX_YEAR}.pdf"


def test_tax_year_is_overridable(monkeypatch):
    monkeypatch.setenv("TAXCITE_TAX_YEAR", "2025")
    reloaded = importlib.reload(manifest)
    try:
        assert reloaded.TAX_YEAR == "2025"
        assert reloaded.CORPUS[0].url.endswith("--2025.pdf")
    finally:
        monkeypatch.delenv("TAXCITE_TAX_YEAR", raising=False)
        importlib.reload(manifest)


def test_every_publication_has_a_distinct_pinned_url():
    urls = [p.url for p in manifest.CORPUS]
    assert len(urls) == len(set(urls)) == 14
