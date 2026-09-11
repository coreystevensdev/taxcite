"""Corpus manifest: which IRS publications are in scope, and for which tax year.

pub/irs-pdf/p501.pdf always serves whatever revision is current, so an ingest run
in December and the same run in February fetch different documents under the same
name. This module used to use that URL and track revisions by content hash after
the fact, which detects the change but does not stop it: the eval dataset's ground
truth is pinned to a tax year, and once the IRS published the 2025 revisions the
agent started correctly refusing to state 2024 figures and scoring zero for it.

pub/irs-prior carries every publication addressed by year, so the corpus is
reproducible. TAX_YEAR is the one the dataset is written against, and the two have
to move together.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

IRS_PRIOR_BASE = "https://www.irs.gov/pub/irs-prior"

# The tax year eval/dataset.jsonl's ground truth states. Changing this means
# rewriting the dataset's dollar figures to match, or the eval scores correct
# answers as wrong.
TAX_YEAR = os.environ.get("TAXCITE_TAX_YEAR", "2024")


@dataclass(frozen=True)
class Publication:
    pub_id: str
    title: str

    @property
    def url(self) -> str:
        return f"{IRS_PRIOR_BASE}/{self.pub_id}--{TAX_YEAR}.pdf"

    @property
    def filename(self) -> str:
        # Year in the cache name, so switching TAX_YEAR does not silently reuse
        # the other year's download.
        return f"{self.pub_id}--{TAX_YEAR}.pdf"


CORPUS: tuple[Publication, ...] = (
    Publication("p17", "Your Federal Income Tax (For Individuals)"),
    Publication("p501", "Dependents, Standard Deduction, and Filing Information"),
    Publication("p502", "Medical and Dental Expenses"),
    Publication("p503", "Child and Dependent Care Expenses"),
    Publication("p505", "Tax Withholding and Estimated Tax"),
    Publication("p523", "Selling Your Home"),
    Publication("p525", "Taxable and Nontaxable Income"),
    Publication("p526", "Charitable Contributions"),
    Publication("p550", "Investment Income and Expenses"),
    Publication("p590a", "Contributions to Individual Retirement Arrangements (IRAs)"),
    Publication("p590b", "Distributions from Individual Retirement Arrangements (IRAs)"),
    Publication("p596", "Earned Income Credit (EIC)"),
    Publication("p936", "Home Mortgage Interest Deduction"),
    Publication("p970", "Tax Benefits for Education"),
)


def get_publication(pub_id: str) -> Publication:
    for pub in CORPUS:
        if pub.pub_id == pub_id:
            return pub
    known = ", ".join(p.pub_id for p in CORPUS)
    raise KeyError(f"unknown publication {pub_id!r}; corpus contains: {known}")
