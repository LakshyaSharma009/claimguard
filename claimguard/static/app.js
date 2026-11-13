"use strict";

// ---------- helpers ----------
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 1) => (x == null ? "–" : `${(100 * x).toFixed(d)}%`);
const fmtTime = (iso) => (iso ? new Date(iso).toLocaleString() : "–");

const BAND_MEANING = {
  LOW: "Auto-approve. Nothing in the claim stands out.",
  MEDIUM: "Quick review by a claims handler before payment.",
  HIGH: "Send to the investigation team.",
};

async function api(path, options = {}) {
  const res = await fetch(path, options);
  let body = null;
  try { body = await res.json(); } catch { /* empty body */ }
  if (!res.ok) {
    const err = new Error(typeof body?.detail === "string" ? body.detail : `Request failed (${res.status})`);
    err.status = res.status;
    err.detail = body?.detail;
    throw err;
  }
  return body;
}

const state = { health: null, schema: null, batchResults: [], selectedDecision: null };

// ---------- tabs ----------
function showTab(name) {
  $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $$(".tab-panel").forEach((p) => { p.hidden = p.id !== `tab-${name}`; });
  if (name === "queue") loadQueue();
  if (name === "monitor") loadMonitor();
}
$$(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ---------- header ----------
async function loadHealth() {
  try {
    const h = await api("/health");
    state.health = h;
    const years = h.training_years ? `${h.training_years[0]}–${h.training_years.at(-1)}` : "?";
    $("#model-status").innerHTML =
      `Model <strong>${esc(h.model_version)}</strong> · trained on ${esc(years)}` +
      (h.training_claims ? ` (${h.training_claims.toLocaleString()} claims)` : "") +
      (h.config_matches_training ? "" : ` · <span class="warn">config changed since training: retrain</span>`);
  } catch (e) {
    $("#model-status").innerHTML = `<span class="warn">API unreachable: ${esc(e.message)}</span>`;
  }
}

// ---------- score form ----------
const LOW_RISK = {
  Month: "Mar", WeekOfMonth: 2, DayOfWeek: "Tuesday", Make: "Toyota", AccidentArea: "Urban",
  DayOfWeekClaimed: "Wednesday", MonthClaimed: "Mar", WeekOfMonthClaimed: 2, Sex: "Male",
  MaritalStatus: "Married", Age: 45, Fault: "Third Party", PolicyType: "Sedan - Liability",
  VehicleCategory: "Sedan", VehiclePrice: "20000 to 29000", Deductible: 400, DriverRating: 2,
  Days_Policy_Accident: "more than 30", Days_Policy_Claim: "more than 30", PastNumberOfClaims: "1",
  AgeOfVehicle: "5 years", AgeOfPolicyHolder: "41 to 50", PoliceReportFiled: "Yes", WitnessPresent: "Yes",
  AgentType: "External", NumberOfSuppliments: "none", AddressChange_Claim: "no change",
  NumberOfCars: "1 vehicle", BasePolicy: "Liability",
};
const HIGH_RISK = {
  ...LOW_RISK, Fault: "Policy Holder", BasePolicy: "All Perils", PolicyType: "Sedan - All Perils",
  AddressChange_Claim: "under 6 months", PoliceReportFiled: "No", WitnessPresent: "No",
  Days_Policy_Accident: "1 to 7",
};

function buildForm(schema) {
  const host = $("#form-groups");
  host.innerHTML = schema.groups.map((g) => `
    <fieldset>
      <legend>${esc(g.name)}</legend>
      <div class="grid">
        ${g.fields.map((f) => {
          const cls = `field${f.used_by_model ? "" : " unused"}`;
          const title = f.used_by_model ? "" : ' title="Recorded, not used by the model"';
          const label = `<span>${esc(f.label)}${f.optional ? " <em>(optional)</em>" : ""}</span>`;
          if (f.type === "choice") {
            return `<label class="${cls}"${title}>${label}<select name="${esc(f.name)}" required>
              <option value="" disabled selected>Choose…</option>
              ${f.options.map((o) => `<option>${esc(o)}</option>`).join("")}</select></label>`;
          }
          return `<label class="${cls}"${title}>${label}<input type="number" name="${esc(f.name)}"
            min="${f.min}" max="${f.max}" step="1" ${f.optional ? 'placeholder="not recorded"' : "required"}></label>`;
        }).join("")}
      </div>
    </fieldset>`).join("");
}

function fillForm(values) {
  const form = $("#claim-form");
  for (const [k, v] of Object.entries(values)) {
    const el = form.elements[k];
    if (el) el.value = v;
  }
  $$(".field.invalid", form).forEach((f) => f.classList.remove("invalid"));
  $("#form-error").textContent = "";
}

function readForm() {
  const form = $("#claim-form");
  const claim = {};
  const missing = [];
  $$(".field.invalid", form).forEach((f) => f.classList.remove("invalid"));
  for (const g of state.schema.groups) {
    for (const f of g.fields) {
      const el = form.elements[f.name];
      const raw = el.value.trim();
      if (raw === "") {
        if (!f.optional) { missing.push(f.label); el.closest(".field").classList.add("invalid"); }
        continue;
      }
      if (f.type === "number") {
        const n = Number(raw);
        if (!Number.isInteger(n) || n < f.min || n > f.max) {
          missing.push(`${f.label} (${f.min}–${f.max})`);
          el.closest(".field").classList.add("invalid");
          continue;
        }
        claim[f.name] = n;
      } else {
        claim[f.name] = raw;
      }
    }
  }
  const ref = form.elements.claim_ref.value.trim();
  if (ref) claim.claim_ref = ref;
  return { claim, missing };
}

function showServerErrors(err) {
  if (Array.isArray(err.detail)) {
    const fields = err.detail.map((d) => d.loc?.at(-1)).filter(Boolean);
    fields.forEach((name) => $("#claim-form").elements[name]?.closest(".field")?.classList.add("invalid"));
    return `The API rejected: ${fields.join(", ")}`;
  }
  return err.message;
}

$("#claim-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const { claim, missing } = readForm();
  if (missing.length) {
    $("#form-error").textContent = `Please fill in: ${missing.slice(0, 6).join(", ")}${missing.length > 6 ? "…" : ""}`;
    return;
  }
  $("#form-error").textContent = "";
  const btn = $("#claim-form button[type=submit]");
  btn.disabled = true;
  try {
    const d = await api("/score", { method: "POST", headers: { "Content-Type": "application/json" },
                                    body: JSON.stringify(claim) });
    renderResult(d);
  } catch (e) {
    $("#form-error").textContent = showServerErrors(e);
  } finally {
    btn.disabled = false;
  }
});
$$("[data-fill]").forEach((b) => b.addEventListener("click", () =>
  fillForm(b.dataset.fill === "high" ? HIGH_RISK : LOW_RISK)));

function gaugeHtml(score) {
  const t = state.health?.thresholds;
  if (!t) return "";
  const max = Math.max(0.4, score * 1.15, t.high * 1.6);
  const pos = (x) => `${Math.min(100, (100 * x) / max).toFixed(2)}%`;
  return `<div class="gauge" style="--m:${pos(t.medium)};--h:${pos(t.high)}" aria-hidden="true">
      <div class="marker" style="left:${pos(score)}"></div>
      <span class="tick" style="left:${pos(t.medium)}">MEDIUM ≥ ${t.medium.toFixed(3)}</span>
      <span class="tick top" style="left:${pos(t.high)}">HIGH ≥ ${t.high.toFixed(3)}</span>
    </div>`;
}

function rulesHtml(hits) {
  if (!hits?.length) return `<p class="muted">No red-flag rules fired.</p>`;
  return hits.map((r) => `<div class="rule${r.strong ? " strong" : ""}">${esc(r.reason)}
      <div class="rid">${esc(r.id)}${r.strong ? " · strong rule: at least MEDIUM" : ""}</div></div>`).join("");
}

function reasonsHtml(reasons) {
  if (!reasons?.length) return `<p class="muted">No feature pushed this claim towards fraud.</p>`;
  const top = Math.max(...reasons.map((r) => r.impact));
  return reasons.map((r) => `<div class="reason"><span>${esc(r.text)}</span>
      <div class="bar" title="SHAP impact ${r.impact}"><div style="width:${(100 * r.impact / top).toFixed(0)}%"></div></div></div>`).join("");
}

function renderResult(d) {
  const el = $("#result");
  el.classList.remove("empty");
  el.innerHTML = `
    <div class="result-head">
      <span class="band band-big ${esc(d.band)}">${esc(d.band)}</span>
      <div style="text-align:right">
        <div class="score-num">${d.score.toFixed(3)}</div>
        <div class="score-sub">estimated fraud probability</div>
      </div>
    </div>
    <p class="band-meaning">${esc(BAND_MEANING[d.band])}</p>
    ${gaugeHtml(d.score)}
    <div class="section-title">Red-flag rules</div>
    ${rulesHtml(d.rule_hits)}
    <div class="section-title">Top model reasons</div>
    ${reasonsHtml(d.model_reasons)}
    <div class="meta">Decision <strong>${esc(d.decision_id)}</strong>${d.claim_ref ? ` · ref ${esc(d.claim_ref)}` : ""}
      · model ${esc(d.model_version)} · logged to the audit trail.
      <br><a href="#" data-open="${esc(d.decision_id)}">Open in review queue →</a></div>`;
  $("[data-open]", el).addEventListener("click", (e) => { e.preventDefault(); openDecision(d.decision_id); });
}

// ---------- batch ----------
$("#batch-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const file = $("#batch-file").files[0];
  if (!file) return;
  const btn = $("#batch-form button");
  btn.disabled = true;
  $("#batch-status").textContent = `Scoring ${file.name}…`;
  const t0 = performance.now();
  try {
    const fd = new FormData();
    fd.append("file", file);
    const res = await api("/score/batch", { method: "POST", body: fd });
    $("#batch-status").textContent = `Done in ${((performance.now() - t0) / 1000).toFixed(1)} s.`;
    renderBatch(res);
  } catch (e) {
    $("#batch-status").innerHTML = `<span class="error">${esc(e.message)}</span>`;
  } finally {
    btn.disabled = false;
  }
});

function renderBatch(res) {
  $("#batch-result").hidden = false;
  const total = res.scored || 1;
  $("#batch-stats").innerHTML = `
    <div class="stat"><div class="v">${res.scored.toLocaleString()}</div><div class="k">claims scored</div></div>
    ${["HIGH", "MEDIUM", "LOW"].map((b) => `<div class="stat ${b}"><div class="v">${res.bands[b].toLocaleString()}</div>
      <div class="k">${b} · ${pct(res.bands[b] / total)}</div></div>`).join("")}
    <div class="stat"><div class="v">${res.rejected}</div><div class="k">rows rejected</div></div>`;

  $("#batch-errors-card").hidden = !res.errors.length;
  $("#batch-errors").innerHTML = res.errors.slice(0, 50).map((e) =>
    `<div class="rule">Row ${e.row}: ${e.errors.map((x) => `${esc(x.field)}: ${esc(x.msg)}`).join("; ")}</div>`).join("") +
    (res.errors.length > 50 ? `<p class="muted">…and ${res.errors.length - 50} more.</p>` : "");

  state.batchResults = [...res.results].sort((a, b) => b.score - a.score);
  const rows = state.batchResults.slice(0, 500);
  $("#batch-table").innerHTML = `
    <thead><tr><th class="num">Row</th><th>Claim ref</th><th>Band</th><th class="num">Score</th>
      <th>Rules</th><th>Top reason</th><th>Decision</th></tr></thead>
    <tbody>${rows.map((r) => `<tr class="clickable" data-id="${esc(r.decision_id)}">
      <td class="num">${r.row}</td><td>${esc(r.claim_ref ?? "")}</td>
      <td><span class="band ${esc(r.band)}">${esc(r.band)}</span></td>
      <td class="num">${r.score.toFixed(3)}</td>
      <td>${r.rule_hits.map((h) => esc(h.id.split("_")[0])).join(", ")}</td>
      <td>${esc(r.model_reasons[0]?.text ?? "")}</td><td>${esc(r.decision_id)}</td></tr>`).join("")}</tbody>`;
  if (state.batchResults.length > 500) {
    $("#batch-table").insertAdjacentHTML("beforeend",
      `<tfoot><tr class="empty-row"><td colspan="7">Showing the top 500 of ${state.batchResults.length}. Download the CSV for all rows.</td></tr></tfoot>`);
  }
  $$("#batch-table tbody tr").forEach((tr) => tr.addEventListener("click", () => openDecision(tr.dataset.id)));
}

$("#batch-download").addEventListener("click", () => {
  const cols = ["row", "claim_ref", "decision_id", "band", "score", "model_probability", "rule_score", "rules", "reasons"];
  const q = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const lines = [cols.join(",")].concat(state.batchResults.map((r) => [
    r.row, r.claim_ref, r.decision_id, r.band, r.score, r.model_probability, r.rule_score,
    r.rule_hits.map((h) => h.id).join("; "), r.model_reasons.map((m) => m.text).join("; "),
  ].map(q).join(",")));
  const url = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: "claimguard_results.csv" });
  a.click();
  URL.revokeObjectURL(url);
});

// ---------- review queue ----------
async function loadQueue() {
  const bands = $$('#tab-queue input[name=band]:checked').map((c) => c.value);
  const params = new URLSearchParams();
  bands.forEach((b) => params.append("band", b));
  const reviewed = $("#queue-reviewed").value;
  if (reviewed) params.set("reviewed", reviewed);
  params.set("sort", $("#queue-sort").value);
  params.set("limit", "200");
  const table = $("#queue-table");
  try {
    const rows = bands.length ? await api(`/decisions?${params}`) : [];
    table.innerHTML = `
      <thead><tr><th>Decision</th><th>Claim ref</th><th>Band</th><th class="num">Score</th>
        <th class="num">Rules</th><th>Status</th><th>Scored</th></tr></thead>
      <tbody>${rows.length ? rows.map((r) => `<tr class="clickable${r.decision_id === state.selectedDecision ? " selected" : ""}" data-id="${esc(r.decision_id)}">
        <td>${esc(r.decision_id)}</td><td>${esc(r.claim_ref ?? "")}</td>
        <td><span class="band ${esc(r.band)}">${esc(r.band)}</span></td>
        <td class="num">${r.score.toFixed(3)}</td><td class="num">${r.rules_hit}</td>
        <td>${r.latest_outcome ? `${esc(r.latest_outcome === "CONFIRMED_FRAUD" ? "Confirmed fraud" : "Cleared")} · ${esc(r.reviewer)}` : '<span class="muted">Awaiting review</span>'}</td>
        <td>${esc(fmtTime(r.created_at))}</td></tr>`).join("")
        : `<tr class="empty-row"><td colspan="7">Nothing here. Score some claims first, or change the filters.</td></tr>`}</tbody>`;
    $$("#queue-table tbody tr.clickable").forEach((tr) => tr.addEventListener("click", () => openDecision(tr.dataset.id, false)));
  } catch (e) {
    table.innerHTML = `<tbody><tr class="empty-row"><td class="error">${esc(e.message)}</td></tr></tbody>`;
  }
}
$$("#tab-queue input[name=band], #queue-reviewed, #queue-sort").forEach((el) => el.addEventListener("change", loadQueue));
$("#queue-refresh").addEventListener("click", loadQueue);

async function openDecision(id, switchTab = true) {
  state.selectedDecision = id;
  if (switchTab) showTab("queue");
  $$("#queue-table tbody tr").forEach((tr) => tr.classList.toggle("selected", tr.dataset.id === id));
  const el = $("#decision-detail");
  el.classList.remove("empty");
  el.innerHTML = `<p class="muted">Loading ${esc(id)}…</p>`;
  let d;
  try { d = await api(`/decisions/${encodeURIComponent(id)}`); } catch (e) {
    el.innerHTML = `<p class="error">${esc(e.message)}</p>`;
    return;
  }
  const labels = Object.fromEntries((state.schema?.groups ?? []).flatMap((g) => g.fields.map((f) => [f.name, f.label])));
  const claimRows = Object.entries(d.claim).filter(([, v]) => v !== null && v !== undefined)
    .map(([k, v]) => `<dt>${esc(labels[k] ?? k)}</dt><dd>${esc(v)}</dd>`).join("");
  el.innerHTML = `
    <div class="result-head">
      <div><h2 style="margin:0">${esc(d.decision_id)}</h2>
        <div class="score-sub">${d.claim.claim_ref ? `ref ${esc(d.claim.claim_ref)} · ` : ""}scored ${esc(fmtTime(d.created_at))}</div></div>
      <div style="text-align:right"><span class="band band-big ${esc(d.band)}">${esc(d.band)}</span>
        <div class="score-sub">score ${d.score.toFixed(3)}</div></div>
    </div>
    <p class="band-meaning">${esc(BAND_MEANING[d.band])}</p>
    <div class="section-title">Red-flag rules</div>${rulesHtml(d.rule_hits)}
    <div class="section-title">Top model reasons</div>${reasonsHtml(d.model_reasons)}
    <details><summary>Full claim</summary><dl class="claim-kv">${claimRows}</dl></details>
    <div class="section-title">Investigation</div>
    ${d.reviews.length ? d.reviews.map((r) => `<div class="review-item ${esc(r.outcome)}">
        <strong>${r.outcome === "CONFIRMED_FRAUD" ? "Confirmed fraud" : "Cleared"}</strong> by ${esc(r.reviewer)}
        · ${esc(fmtTime(r.created_at))}${r.notes ? `<br>${esc(r.notes)}` : ""}</div>`).join("")
      : '<p class="muted">No outcome recorded yet.</p>'}
    <form class="review-form" id="review-form">
      <div class="outcome-choice" role="radiogroup" aria-label="Outcome">
        <label class="fraud"><input type="radio" name="outcome" value="CONFIRMED_FRAUD" required><span>Confirmed fraud</span></label>
        <label class="cleared"><input type="radio" name="outcome" value="CLEARED"><span>Cleared</span></label>
      </div>
      <label class="field"><span>Reviewer</span><input name="reviewer" maxlength="100" required
        value="${esc(localStorage.getItem("cg-reviewer") ?? "")}"></label>
      <label class="field"><span>Notes</span><textarea name="notes" rows="3" maxlength="2000"></textarea></label>
      <div class="form-footer"><span id="review-msg" role="status"></span>
        <button type="submit" class="primary">${d.reviews.length ? "Add another outcome" : "Record outcome"}</button></div>
      <p class="hint">Outcomes are appended, never edited: a new outcome supersedes the previous one in the metrics.</p>
    </form>
    <div class="meta">Model ${esc(d.model_version)} · config ${esc(d.config_hash)}<br>row hash ${esc(d.row_hash)}</div>`;

  $("#review-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const f = ev.target;
    const outcome = f.elements.outcome.value;
    const reviewer = f.elements.reviewer.value.trim();
    if (!outcome || !reviewer) {
      $("#review-msg").innerHTML = '<span class="error">Choose an outcome and enter your name.</span>';
      return;
    }
    localStorage.setItem("cg-reviewer", reviewer);
    try {
      await api(`/decisions/${encodeURIComponent(id)}/review`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ outcome, reviewer, notes: f.elements.notes.value.trim() }),
      });
      await openDecision(id, false);
      $("#decision-detail").insertAdjacentHTML("afterbegin", '<p class="ok">Outcome recorded.</p>');
      loadQueue();
    } catch (e) {
      $("#review-msg").innerHTML = `<span class="error">${esc(e.message)}</span>`;
    }
  });
}

// ---------- monitoring ----------
async function loadMonitor() {
  try {
    const [m, h] = await Promise.all([api("/metrics"), api("/health")]);
    state.health = h;
    const b = m.bands;
    $("#monitor-stats").innerHTML = `
      <div class="stat"><div class="v">${m.total_decisions.toLocaleString()}</div><div class="k">decisions logged</div></div>
      ${["HIGH", "MEDIUM", "LOW"].map((k) => `<div class="stat ${k}"><div class="v">${pct(b[k].confirmed_fraud_rate)}</div>
        <div class="k">${k}: confirmed fraud among ${b[k].reviewed} reviewed</div></div>`).join("")}`;
    $("#monitor-table").innerHTML = `
      <thead><tr><th>Band</th><th class="num">Decisions</th><th class="num">Share</th><th class="num">Reviewed</th>
        <th class="num">Confirmed fraud</th><th class="num">Confirmed-fraud rate</th></tr></thead>
      <tbody>${["HIGH", "MEDIUM", "LOW"].map((k) => `<tr><td><span class="band ${k}">${k}</span></td>
        <td class="num">${b[k].decisions.toLocaleString()}</td>
        <td class="num">${pct(m.total_decisions ? b[k].decisions / m.total_decisions : null)}</td>
        <td class="num">${b[k].reviewed}</td><td class="num">${b[k].confirmed_fraud}</td>
        <td class="num">${pct(b[k].confirmed_fraud_rate)}</td></tr>`).join("")}</tbody>`;
    const kv = {
      "Model version": h.model_version,
      "Trained": fmtTime(h.trained_at),
      "Training data": h.training_years ? `${h.training_years.join(", ")} · ${h.training_claims?.toLocaleString()} claims` : "–",
      "MEDIUM threshold": h.thresholds.medium.toFixed(4),
      "HIGH threshold": h.thresholds.high.toFixed(4),
      "Model weight in score": h.combine_weight,
      "Rules active": h.rules_loaded.join(", "),
      "Config hash": `${h.config_hash} ${h.config_matches_training ? "(matches training)" : "(CHANGED since training)"}`,
    };
    $("#model-info").innerHTML = Object.entries(kv).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  } catch (e) {
    $("#monitor-table").innerHTML = `<tbody><tr class="empty-row"><td class="error">${esc(e.message)}</td></tr></tbody>`;
  }
}
$("#monitor-refresh").addEventListener("click", loadMonitor);

// ---------- start ----------
(async function init() {
  await loadHealth();
  try {
    state.schema = await api("/schema");
    buildForm(state.schema);
  } catch (e) {
    $("#form-groups").innerHTML = `<p class="error">Could not load the form: ${esc(e.message)}</p>`;
  }
})();
