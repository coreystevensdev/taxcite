"""Corpus manifest: which IRS publications are in scope.

irs.gov serves the current revision of every publication at a stable
URL (pub/irs-pdf/p<number>.pdf), so the manifest pins identity, not
version. Revision tracking happens at ingest time via content hash.
"""

from __future__ import annotations

from dataclasses import dataclass

IRS_PDF_BASE = "https://www.irs.gov/pub/irs-pdf"


@dataclass(frozen=True)
class Publication:
    pub_id: str
    title: str

    @property
    def url(self) -> str:
        return f"{IRS_PDF_BASE}/{self.pub_id}.pdf"

    @property
    def filename(self) -> str:
        return f"{self.pub_id}.pdf"


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
