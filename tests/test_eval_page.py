"""The browsable eval report.

taxcite is the only pinned repo a reader cannot click into, and the thing worth
showing was already committed: 50 scored runs with their answers, their
retrieved passages and their per-question scores. These pin the contract
between the page and that file, because a key rename on either side empties a
panel with nothing failing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPORT = Path(__file__).resolve().parent.parent / "eval" / "report.json"


@pytest.fixture
def client():
    from taxcite.server import app

    return TestClient(app)


def served(client):
    """The page plus everything it links, which is what a browser actually runs.

    The script and stylesheet were inline until the page got a
    Content-Security-Policy, and a policy permitting its own inline <script>
    permits an injected one too. These assertions follow the page's own
    references now rather than assuming the code sits in the document, which is
    the contract that matters and does not care where it lives.
    """
    html = client.get("/").text
    parts = [html]
    for ref in re.findall(r'(?:src|href)="(/[^"]+\.(?:js|css))"', html):
        parts.append(client.get(ref).text)
    return "\n".join(parts)


def test_the_root_serves_the_page(client):
    resp = client.get("/")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "<!doctype html>" in resp.text.lower()


def test_the_report_is_served_as_committed(client):
    """Served from the file rather than rebuilt, so what a reader sees is what
    the repo contains."""
    resp = client.get("/eval/report.json")

    assert resp.status_code == 200
    assert resp.json() == json.loads(REPORT.read_text())


def test_every_field_the_page_reads_exists_in_the_report(client):
    page = served(client)
    report = json.loads(REPORT.read_text())

    for key in ("metrics", "n_questions", "corpus_tax_year", "per_question"):
        assert key in report, f"report lost {key}"
        assert key in page, f"page stopped reading {key}"

    entry = report["per_question"][0]
    for key in ("user_input", "response", "reference", "retrieved_contexts", "scores"):
        assert key in entry, f"per_question lost {key}"
        assert key in page, f"page stopped reading {key}"

    for metric in ("faithfulness", "answer_relevancy", "context_precision"):
        assert metric in report["metrics"], f"metrics lost {metric}"
        assert metric in entry["scores"], f"per-question scores lost {metric}"
        assert metric in page


def test_the_page_offers_the_report_path_it_fetches(client):
    assert "/eval/report.json" in served(client)


def test_no_question_is_missing_a_score(client):
    """A null would render as "n/a" and quietly shrink what the page claims to
    have measured."""
    report = json.loads(REPORT.read_text())

    for i, q in enumerate(report["per_question"]):
        for metric in ("faithfulness", "answer_relevancy", "context_precision"):
            assert q["scores"].get(metric) is not None, f"question {i} has no {metric}"


def test_the_aggregate_matches_the_per_question_scores(client):
    """The headline numbers are a mean of what is listed below them. If they
    ever disagree, the page is showing two different evaluations."""
    report = json.loads(REPORT.read_text())
    pq = report["per_question"]

    for metric, stated in report["metrics"].items():
        actual = sum(q["scores"][metric] for q in pq) / len(pq)
        assert abs(actual - stated) < 0.01, (
            f"{metric}: report says {stated}, the 50 questions average {actual:.4f}"
        )


def test_the_page_names_no_colour_the_tokens_do_not(client):
    body = client.get("/").text  # markup only; the CSS is a file now

    assert not re.search(r"#[0-9A-Fa-f]{6}\b", body), "a literal colour escaped into the markup"


def test_the_exact_zeros_are_still_there_and_still_contradicted(client):
    """Two scores came back as exactly 0.000 beside a faithfulness that is
    computed from the same retrieved passages. Both cannot be right, so the
    page labels them rather than printing 0.000 as a measurement.

    If a re-run ever removes them, this fails and the page's note should go
    with them."""
    report = json.loads(REPORT.read_text())
    pq = report["per_question"]

    contradicted = [
        (m, q["scores"]["faithfulness"], q["user_input"])
        for q in pq
        for m in ("answer_relevancy", "context_precision")
        if q["scores"][m] == 0.0 and q["scores"]["faithfulness"] >= 0.9
    ]

    assert contradicted, "no contradicted zeros left; drop the page's note too"
    assert len(contradicted) == 2, f"expected 2, found {len(contradicted)}: {contradicted}"

    page = served(client)
    assert "Exact zero while faithfulness" in page


def test_the_page_leads_with_the_ask_box_not_the_eval(client):
    """The eval is the stronger artifact but it is the second thing, because a
    reader has to see what the product does before a score about it means
    anything."""
    body = served(client)

    ask_at = body.index('id="askform"')
    eval_at = body.index('id="agg"')
    assert ask_at < eval_at, "the eval strip comes before the ask box"


def test_the_page_reads_the_answering_mode(client):
    body = served(client)

    assert "/status" in body
    assert "answering" in body
    assert "recorded" in body, "no fallback for a deployment without keys"


def test_the_recorded_fallback_cannot_pass_as_live(client):
    """A recorded answer shown without saying so is a claim about what this
    deployment can do."""
    body = served(client)

    assert "Answering is not configured on this deployment" in body
    assert "Not generated now." in body


def test_the_page_handles_the_human_review_interrupt(client):
    """/ask stops before generating and returns awaiting_review with a
    thread_id. A page that only read d.answer would show an empty card and
    hide the one step that makes this different from a chat box."""
    body = served(client)

    assert "awaiting_review" in body
    assert "/ask/resume" in body
    assert "thread_id" in body
    assert "chunks_preview" in body


def test_the_gate_offers_both_answers(client):
    """Declining has to be possible, or it is not a gate."""
    body = served(client)

    assert "Approve and generate" in body
    assert "Decline" in body
    assert "nothing was spent" in body


def test_the_page_carries_no_inline_script_or_style(client):
    """The property the Content-Security-Policy depends on.

    `served()` deliberately does not care whether the code is inline or linked,
    which is right for the content assertions and useless as a guard here: when
    the script was put back inline during a check, all of them still passed and
    the page would have been broken in a browser, because script-src 'self'
    blocks an inline block while CI sees nothing.
    """
    html = client.get("/").text

    assert "<script>" not in html, (
        "an inline <script> is back; script-src 'self' will block it and the "
        "page will render dead with every test still green"
    )
    assert "<style>" not in html, "an inline <style> is back"
    assert '<script src="/app.js">' in html
    assert '<link rel="stylesheet" href="/app.css">' in html


def test_the_policy_leaves_no_escape_for_script(client):
    """script-src carries no 'unsafe-inline' and no 'unsafe-eval'.

    style-src keeps 'unsafe-inline' on purpose: five style= attributes and one
    element.style assignment remain in the markup, and style injection is a far
    smaller problem than script injection. That asymmetry is the point, so it is
    asserted rather than left to a reader to notice.
    """
    csp = client.get("/").headers["content-security-policy"]
    directives = dict(
        (d.split(" ", 1) + [""])[:2] for d in (p.strip() for p in csp.split(";")) if d
    )

    assert "'unsafe-inline'" not in directives["script-src"]
    assert "'unsafe-eval'" not in directives["script-src"]
    assert directives["script-src"].strip() == "'self'"
    assert directives["object-src"].strip() == "'none'"
    assert directives["frame-ancestors"].strip() == "'none'"
    # Google Fonts is the one pair that cannot be 'self'.
    assert "https://fonts.googleapis.com" in directives["style-src"]
    assert "https://fonts.gstatic.com" in directives["font-src"]
