const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const band = (v) => v >= 0.85 ? "strong" : v >= 0.6 ? "mid" : "weak";
const n3 = (v) => v == null ? "n/a" : Number(v).toFixed(3);

const METRICS = [
  ["faithfulness", "Faithfulness", "Is every claim in the answer supported by a retrieved passage?"],
  ["answer_relevancy", "Answer relevancy", "Ragas generates questions from the answer and compares them to the real one, so a thorough answer scores lower."],
  ["context_precision", "Context precision", "How much of what retrieval returned was actually useful?"],
];

let data = null, order = "worst";

// The passages are raw publication text. Surfacing the publication number makes
// each one a citation rather than an anonymous blob, which is the repo's claim.
function pubOf(text) {
  const m = String(text).match(/Publication\s+(\d{3,4})|Pub\.?\s*(\d{3,4})/);
  return m ? "Pub. " + (m[1] || m[2]) : null;
}

function renderAggregate() {
  const m = data.metrics;
  const zeros = data.per_question.flatMap(q =>
    METRICS.filter(([k]) => q.scores[k] === 0).map(([k]) => k));
  // A zero beside a high faithfulness is the scorer failing: both are computed
  // against the same passages, so they cannot both be right.
  const suspect = data.per_question.filter(
    q => METRICS.some(([k]) => q.scores[k] === 0) && q.scores.faithfulness >= 0.9);
  $("corpus").textContent =
    `corpus tax year ${data.corpus_tax_year} · ${data.n_questions} questions · ` +
    `${data.per_question.reduce((a, q) => a + q.retrieved_contexts.length, 0)} passages retrieved in total`;

  $("agg").innerHTML = METRICS.map(([k, label]) => {
    const v = m[k];
    return `<div>
      <div class="k">${esc(label)}</div>
      <div class="v" style="color:var(--${band(v)})">${n3(v)}</div>
      <div class="n">mean of ${data.n_questions}</div>
    </div>`;
  }).join("") + `<div>
      <div class="k">Faithfulness at 1.0</div>
      <div class="v">${data.per_question.filter(q => q.scores.faithfulness === 1).length} / ${data.n_questions}</div>
      <div class="n">perfect runs</div>
    </div>`;

  $("caveat").innerHTML =
    `<strong>Answer relevancy is the weak one, and it is the metric to read carefully.</strong> ` +
    `Ragas scores it by generating questions from the answer and comparing them to the one asked, ` +
    `so a thorough answer that covers conditions and exceptions scores lower than a terse one. ` +
    `Sort worst first and read a few: the low scorers are mostly long, not wrong.` +
    (zeros.length ? `<br><br><strong>${zeros.length} scores came back as exactly 0.000.</strong> ` +
      (suspect.length
        ? `${suspect.length} of them sit beside a faithfulness of ` +
          suspect.map(q => n3(q.scores.faithfulness)).join(" and ") +
          `, which are computed against the same retrieved passages, so those cannot both be ` +
          `right. They look like the scorer failing rather than an answer scoring nothing. `
        : "") +
      `They are left in the published means above; dropping all ${zeros.length} would read ` +
      METRICS.filter(([k]) => data.per_question.some(q => q.scores[k] === 0))
        .map(([k, label]) => {
          const kept = data.per_question.map(q => q.scores[k]).filter(v => v !== 0);
          return `${label.toLowerCase()} ${(kept.reduce((a, b) => a + b, 0) / kept.length).toFixed(3)}`;
        }).join(" and ") + `.` : "");
}

function renderList() {
  const qs = data.per_question.map((q, i) => ({...q, _i: i}));
  if (order === "worst") qs.sort((a, b) => a.scores.faithfulness - b.scores.faithfulness);
  if (order === "best") qs.sort((a, b) => b.scores.faithfulness - a.scores.faithfulness);

  $("qs").innerHTML = qs.map(q => `
    <li>
      <button class="q" aria-expanded="false" data-i="${q._i}">
        <span class="num">${String(q._i + 1).padStart(2, "0")}</span>
        <span class="text">${esc(q.user_input)}</span>
        <span class="marks">${METRICS.map(([k]) =>
          `<span class="mark ${band(q.scores[k])}">${n3(q.scores[k])}</span>`).join("")}</span>
      </button>
      <div class="detail" id="d${q._i}"></div>
    </li>`).join("");
  $("loading").style.display = "none";
}

function renderDetail(i, el) {
  const q = data.per_question[i];
  el.innerHTML = `
    <h3>Scores for this question</h3>
    <div class="scorecard">
      ${METRICS.map(([k, label, note]) => {
        const v = q.scores[k];
        // An exact zero beside a high faithfulness is the scorer failing, not a
        // measurement: faithfulness is computed against these same passages, so
        // it cannot be high while they are worthless or the answer is unrelated.
        const suspect = v === 0 && q.scores.faithfulness >= 0.9;
        return `<div class="s">
          <div class="lbl">${esc(label)}</div>
          <div class="val" style="color:var(--${band(v)})">${n3(v)}</div>
          <div class="track"><span class="${band(v)}" style="width:${Math.max(2, v * 100)}%"></span></div>
          ${suspect ? `<div class="flag">Exact zero while faithfulness is
            ${n3(q.scores.faithfulness)}. Both are computed against the same passages, so this
            reads as the scorer failing rather than a measurement. It is left in the mean.</div>` : ""}
        </div>`;
      }).join("")}
    </div>

    <h3>What TaxCite answered</h3>
    <div class="answer">${esc(q.response)}</div>

    <h3>Reference answer it was scored against</h3>
    <div class="ref">${esc(q.reference)}</div>

    <h3>${q.retrieved_contexts.length} passages retrieval returned</h3>
    <div class="ctxs">
      ${q.retrieved_contexts.map((c, j) => {
        const pub = pubOf(c);
        return `<details class="ctx">
          <summary>
            <span>passage ${j + 1}</span>
            ${pub ? `<span class="pub">${esc(pub)}</span>` : ""}
            <span>${c.length.toLocaleString()} chars</span>
          </summary>
          <div class="body">${esc(c)}</div>
        </details>`;
      }).join("")}
    </div>`;
}

$("qs").addEventListener("click", (e) => {
  const btn = e.target.closest(".q");
  if (!btn) return;
  const i = Number(btn.dataset.i);
  const el = $("d" + i);
  const open = el.classList.toggle("open");
  btn.setAttribute("aria-expanded", String(open));
  if (open && !el.innerHTML.trim()) renderDetail(i, el);
});

$("sorts").addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (!btn) return;
  order = btn.dataset.sort;
  [...$("sorts").children].forEach(b => b.setAttribute("aria-pressed", String(b === btn)));
  renderList();
});

const RECORDED = new Map();
let answering = "not_configured";

function citeChips(contexts) {
  const pubs = [];
  for (const c of contexts) {
    const p = pubOf(c);
    if (p && !pubs.includes(p)) pubs.push(p);
  }
  return pubs;
}

function showAnswer({mode, why, text, cites}) {
  $("ansmode").textContent = mode;
  $("answhy").textContent = why;
  $("anstext").textContent = text;
  $("anscites").innerHTML = (cites || [])
    .map(c => `<span class="cite-chip">${esc(c)}</span>`).join("");
  $("review").innerHTML = "";
  $("answer").classList.add("on");
}

function renderGate(threadId) {
  $("review").innerHTML = `
    <div class="gate">
      <p>Nothing has been generated yet. Approving runs the model against the passages above;
         declining ends the run without spending anything.</p>
      <button type="button" id="approve">Approve and generate</button>
      <button type="button" class="no" id="decline">Decline</button>
    </div>`;

  const decide = async (approved) => {
    for (const id of ["approve", "decline"]) $(id).disabled = true;
    if (approved) $("approve").textContent = "Generating";
    try {
      const res = await fetch("/ask/resume", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({thread_id: threadId, approved}),
      });
      const d = await res.json();
      if (!res.ok) {
        showAnswer({mode: "unavailable", why: d.detail || `HTTP ${res.status}`, text: "", cites: []});
        return;
      }
      showAnswer({
        mode: approved ? "generated" : "declined",
        why: approved
          ? "Generated from the passages you approved."
          : "The run ended at the gate. Nothing was generated and nothing was spent.",
        text: d.answer || "",
        cites: (d.citations || []).map(c => `Pub. ${c.pub_id} p.${c.first_page}`),
      });
    } catch (e) {
      showAnswer({mode: "error", why: "The request did not complete.", text: "", cites: []});
    }
  };

  $("approve").addEventListener("click", () => decide(true));
  $("decline").addEventListener("click", () => decide(false));
}

async function ask(question) {
  const btn = $("go");
  btn.disabled = true;
  const was = btn.textContent;
  btn.textContent = "Asking";

  try {
    if (answering === "live") {
      const res = await fetch("/ask", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question}),
      });
      const d = await res.json();
      if (!res.ok) {
        showAnswer({mode: "unavailable", why: d.detail || `HTTP ${res.status}`, text: "", cites: []});
        return;
      }
      // The graph interrupts before generating. That pause is the product's
      // design rather than a loading state, so the page shows what retrieval
      // found and waits, instead of hiding it behind a spinner.
      if (d.status === "awaiting_review") {
        showAnswer({
          mode: "paused for review",
          why: "Retrieval ran and the graph stopped before generating. These are the passages it would use.",
          text: (d.chunks_preview || []).map(c => `${c.pub_id} ${c.pages}\n${c.excerpt}`).join("\n\n"),
          cites: [...new Set((d.chunks_preview || []).map(c => `Pub. ${c.pub_id.replace(/^p/, "")}`))],
        });
        renderGate(d.thread_id);
        return;
      }

      showAnswer({
        mode: "live",
        why: "Generated now, from the embedded corpus.",
        text: d.answer || "",
        cites: (d.citations || []).map(c => `Pub. ${c.pub_id} p.${c.first_page}`),
      });
      return;
    }

    // Answering is off, so the only honest answers are the ones already run.
    // Matched loosely, because nobody types one of fifty questions verbatim.
    const key = question.trim().toLowerCase().replace(/[^a-z0-9 ]/g, "");
    let hit = RECORDED.get(key);
    if (!hit) {
      const words = key.split(/\s+/).filter(w => w.length > 3);
      let best = 0;
      for (const [k, v] of RECORDED) {
        const score = words.filter(w => k.includes(w)).length / Math.max(words.length, 1);
        if (score > best) { best = score; hit = v; }
      }
      if (best < 0.5) hit = null;
    }

    if (hit) {
      showAnswer({
        mode: "recorded",
        why: `Answering is not configured on this deployment, so this is the run already on record for "${hit.user_input}". Not generated now.`,
        text: hit.response,
        cites: citeChips(hit.retrieved_contexts),
      });
    } else {
      showAnswer({
        mode: "not recorded",
        why: "Answering is not configured on this deployment, and that question is not one of the fifty already run. Pick one below, or read the scored runs underneath.",
        text: "",
        cites: [],
      });
    }
  } catch (e) {
    showAnswer({mode: "error", why: "The request did not complete.", text: "", cites: []});
  } finally {
    btn.disabled = false;
    btn.textContent = was;
  }
}

$("askform").addEventListener("submit", (e) => {
  e.preventDefault();
  ask($("q").value.trim() || $("q").placeholder);
});

function wireAsk() {
  for (const q of data.per_question) {
    RECORDED.set(q.user_input.trim().toLowerCase().replace(/[^a-z0-9 ]/g, ""), q);
  }
  // Three that read as things a person would actually ask.
  const picks = data.per_question.slice(0, 3);
  $("suggest").innerHTML = '<span class="lbl">Try</span>' + picks.map(q =>
    `<button type="button" data-q="${esc(q.user_input)}">${esc(q.user_input.length > 58 ? q.user_input.slice(0, 55) + "..." : q.user_input)}</button>`
  ).join("");
  $("suggest").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    $("q").value = b.dataset.q;
    ask(b.dataset.q);
  });

  const m = data.metrics;
  $("proof").innerHTML =
    `Every answer above comes from the same pipeline that was scored below: ` +
    `<strong>${data.n_questions} questions</strong>, faithfulness <strong>${n3(m.faithfulness)}</strong>, ` +
    `${data.per_question.filter(q => q.scores.faithfulness === 1).length} of them perfect. ` +
    `<a href="#evalhead">See where it did worst</a>.`;
}

fetch("/status")
  .then(r => r.ok ? r.json() : {answering: "not_configured"})
  .then(s => { answering = s.answering || "not_configured"; })
  .catch(() => {});

fetch("/eval/report.json")
  .then(r => r.ok ? r.json() : Promise.reject(new Error(r.status)))
  .then(d => { data = d; renderAggregate(); renderList(); wireAsk(); })
  .catch(() => { $("loading").textContent = "The eval report could not be loaded."; });
