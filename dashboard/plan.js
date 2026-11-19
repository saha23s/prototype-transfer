// Renders dashboard/plan.html from window.PLAN_DATA (built by scripts/build_plan_data.py). No dependencies, no fetch.
const D = window.PLAN_DATA;
const A = D.analysis;
const SVGNS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);
const fmt = (x, n = 3) => (x < 0 ? "−" : "") + Math.abs(x).toFixed(n);
const STATUS_TEXT = { "to-verify": "to verify" };
const MODEL = Object.fromEntries(A.models.map((m) => [m.name, m.label]));

function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  for (const c of children.flat()) if (c != null) n.append(c.nodeType ? c : document.createTextNode(String(c)));
  return n;
}
function svg(tag, attrs = {}, ...children) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  for (const c of children.flat()) if (c != null) n.append(c.nodeType ? c : document.createTextNode(String(c)));
  return n;
}
const text = (x, y, s, cls = "", anchor = "start") => svg("text", { x, y, class: cls, "text-anchor": anchor }, s);
const badge = (status) => el("span", { class: `badge ${status}` }, STATUS_TEXT[status] || status);
function legend(figId, items) {
  const box = $(figId).querySelector(".legend");
  for (const { cls, label } of items) box.append(el("span", {}, cls ? el("i", { class: `swatch ${cls}` }) : null, label));
}
function fill(tableId, rows) {
  const tb = $(tableId).querySelector("tbody");
  for (const r of rows) tb.append(el("tr", {}, ...r.map((c) => el("td", {}, c))));
}

// ---------- header and problem ----------
document.title = D.title;
$("title").textContent = D.title;
$("subtitle").textContent = D.subtitle;
$("statement").textContent = D.problem.statement;
$("transferable").append(el("b", {}, "When is a prototype transferable? "), D.problem.transferable);
const listCard = (title, items, cls = "") => el("div", { class: `col-card ${cls}` }, el("h3", {}, title), el("ul", {}, items.map((t) => el("li", {}, t))));
$("problem-cols").append(listCard("Given", D.problem.given), listCard("Not allowed", D.problem.forbidden, "forbidden"), listCard("Goal", D.problem.goal));
fill("shift-table", D.shift_factors.map((s) => [s.name, s.evidence, s.label_free_fix]));

// ---------- decisions ----------
const ORDER = ["decided", "proposed", "hypothesis", "to-verify", "open"];
const counts = Object.fromEntries(ORDER.map((s) => [s, D.decisions.filter((d) => d.status === s).length]));
for (const s of ORDER) {
  $("decision-tiles").append(el("div", { class: "stat-tile" }, el("span", { class: "value" }, counts[s]), el("span", { class: "label" }, STATUS_TEXT[s] || s)));
}
fill("decision-table", [...D.decisions].sort((a, b) => ORDER.indexOf(a.status) - ORDER.indexOf(b.status))
  .map((d) => [d.title, d.choice, d.why, badge(d.status)]));

// ---------- findings ----------
for (const f of A.findings) $("plan-findings").append(el("div", { class: "finding" }, el("h3", {}, f.title), el("p", {}, f.text)));

// ---------- chart: failing / improving prototype counts ----------
(function chartCounts() {
  const W = 520, left = 175, right = 50, pitch = 52, H = pitch * A.proto_counts.length + 34;
  const x = (v, total) => left + (v / total) * (W - left - right);
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Failing and improving prototype counts per model" });
  const total = A.proto_counts[0].total;
  for (let v = 0; v <= total; v += 15) {
    s.append(svg("line", { x1: x(v, total), x2: x(v, total), y1: 6, y2: H - 24, class: "gridline" }), text(x(v, total), H - 8, v, "", "middle"));
  }
  A.proto_counts.forEach((m, i) => {
    const y = 14 + i * pitch;
    s.append(text(left - 10, y + 22, MODEL[m.name], "row-label", "end"));
    s.append(svg("rect", { x: left, y, width: x(m.fail, total) - left, height: 14, class: `m-${m.name}` }), text(x(m.fail, total) + 5, y + 11, `${m.fail} fail`, "val-label"));
    s.append(svg("rect", { x: left, y: y + 19, width: Math.max(1, x(m.improve, total) - left), height: 14, class: `m-${m.name} stroke bar-outline` }), text(x(m.improve, total) + 5, y + 30, `${m.improve} improve`, "val-label"));
  });
  $("ch-counts").append(s);
  legend("ch-counts", [{ label: `filled bar: Δ < ${fmt(A.thresholds.fail, 2)} (fails)` }, { label: `outline bar: Δ > +${A.thresholds.improve} (improves)` }]);
})();

// ---------- chart: failed-prototype groups (dumbbell) ----------
(function chartGroups() {
  const W = 520, L = 24, R = 24, pitch = 86, H = pitch * A.groups.length + 36, lo = 0.6, hi = 1.0;
  const x = (v) => L + ((v - lo) / (hi - lo)) * (W - L - R);
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Mean soundscape prototype AUC by prototype group and model" });
  for (const t of [0.6, 0.7, 0.8, 0.9, 1.0]) s.append(svg("line", { x1: x(t), x2: x(t), y1: 4, y2: H - 24, class: "gridline" }), text(x(t), H - 8, t.toFixed(1), "", "middle"));
  A.groups.forEach((g, i) => {
    const y = 18 + i * pitch, cy = y + 36;
    const vals = A.models.map((m) => g.sc_auc[m.name]);
    s.append(text(L, y, g.label, "row-label"));
    s.append(svg("line", { x1: x(Math.min(...vals)), x2: x(Math.max(...vals)), y1: cy, y2: cy, class: "connector" }));
    A.models.forEach((m, j) => {
      const v = g.sc_auc[m.name];
      s.append(svg("circle", { cx: x(v), cy, r: 7, class: `m-${m.name} pt-dot` }, svg("title", {}, `${MODEL[m.name]}: ${v}`)));
      s.append(text(x(v), cy + (j === 1 ? 24 : -13), v.toFixed(3), "val-label", "middle"));
    });
  });
  $("ch-groups").append(s);
  legend("ch-groups", A.models.map((m) => ({ cls: m.name === "pretrained-xcl" ? "pretrained" : m.name, label: m.label })));
})();

// ---------- chart: delta scatter, pretrained vs each fine-tuned model ----------
(function chartScatter() {
  const all = A.proto.flatMap((p) => Object.values(p.delta));
  const lo = Math.floor((Math.min(...all) - 0.03) * 10) / 10, hi = Math.ceil((Math.max(...all) + 0.03) * 10) / 10;
  const SW = 560, SH = 420, m = { l: 52, r: 14, t: 30, b: 46 };
  const sx = (v) => m.l + ((v - lo) / (hi - lo)) * (SW - m.l - m.r), sy = (v) => SH - m.b - ((v - lo) / (hi - lo)) * (SH - m.t - m.b);
  const ticks = []; for (let t = Math.ceil(lo * 5) / 5; t <= hi + 1e-9; t += 0.2) ticks.push(Math.round(t * 10) / 10);
  const rho = { "focal-only": A.rho.pretrained_vs_focal_only, "soundscape-only": A.rho.pretrained_vs_soundscape_only };
  for (const other of ["focal-only", "soundscape-only"]) {
    const s = svg("svg", { viewBox: `0 0 ${SW} ${SH}`, role: "img", "aria-label": `Delta AUC under pretrained vs ${MODEL[other]}` });
    s.append(text(m.l, 15, `Pretrained vs ${MODEL[other]}: Spearman ρ = ${rho[other]}`, "panel-title"));
    for (const t of ticks) {
      s.append(svg("line", { x1: sx(t), x2: sx(t), y1: m.t, y2: SH - m.b, class: "gridline" }), svg("line", { x1: m.l, x2: SW - m.r, y1: sy(t), y2: sy(t), class: "gridline" }),
        text(sx(t), SH - m.b + 14, fmt(t, 1), "", "middle"), text(m.l - 6, sy(t) + 3, fmt(t, 1), "", "end"));
    }
    s.append(svg("line", { x1: sx(lo), y1: sy(lo), x2: sx(hi), y2: sy(hi), class: "diag" }));
    s.append(svg("line", { x1: sx(A.thresholds.fail), x2: sx(A.thresholds.fail), y1: m.t, y2: SH - m.b, class: "thr" }), svg("line", { x1: m.l, x2: SW - m.r, y1: sy(A.thresholds.fail), y2: sy(A.thresholds.fail), class: "thr" }));
    for (const p of A.proto) {
      s.append(svg("circle", { cx: sx(p.delta["pretrained-xcl"]), cy: sy(p.delta[other]), r: 4.5, class: `m-${other} pt-dot` },
        svg("title", {}, `${p.species} P${p.k}: Δ pretrained ${fmt(p.delta["pretrained-xcl"])}, Δ ${MODEL[other]} ${fmt(p.delta[other])}`)));
    }
    s.append(text((m.l + SW - m.r) / 2, SH - 8, "\u0394 under pretrained (soundscape AUC \u2212 focal AUC)", "axis-title", "middle"));
    const yt = text(14, (m.t + SH - m.b) / 2, `Δ under ${MODEL[other]}`, "axis-title", "middle");
    yt.setAttribute("transform", `rotate(-90 14 ${(m.t + SH - m.b) / 2})`);
    s.append(yt);
    $("ch-scatter").querySelector(".scatter-pair").append(s);
  }
})();

// ---------- chart: positives per species (log scale) ----------
(function chartSpecies() {
  const rows = [...A.species].sort((a, b) => b.n_soundscape - a.n_soundscape);
  const W = 760, left = 80, right = 70, pitch = 24, H = rows.length * pitch + 44, maxLog = 3;
  const lg = (n) => Math.log10(Math.max(1, n)), x = (n) => left + (lg(n) / maxLog) * (W - left - right);
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Soundscape and focal test positives per species, log scale" });
  for (const t of [1, 10, 100, 1000]) s.append(svg("line", { x1: x(t), x2: x(t), y1: 6, y2: H - 28, class: "gridline" }), text(x(t), H - 12, t, "", "middle"));
  s.append(svg("line", { x1: x(A.primary_min_pos), x2: x(A.primary_min_pos), y1: 6, y2: H - 28, class: "thr" }), text(x(A.primary_min_pos) + 4, 14, `${A.primary_min_pos}`, "val-label"));
  rows.forEach((r, i) => {
    const y = 18 + i * pitch;
    s.append(text(left - 8, y + 11, r.code, "row-label", "end"));
    s.append(svg("rect", { x: left, y, width: x(r.n_soundscape) - left, height: 14, class: `bar-sp ${r.primary ? "primary" : "minor"}` },
      svg("title", {}, `${r.code}: ${r.n_soundscape} soundscape / ${r.n_focal} focal test positives`)));
    s.append(svg("circle", { cx: x(r.n_focal), cy: y + 7, r: 5, class: "ring-focal" }));
    s.append(text(Math.max(x(r.n_soundscape), x(r.n_focal)) + 10, y + 11, `${r.n_soundscape}`, "val-label"));
  });
  $("ch-species").append(s);
  legend("ch-species", [{ cls: "soundscape-only", label: `soundscape positives, primary species (≥ ${A.primary_min_pos})` }, { cls: "other", label: "soundscape positives, below threshold" }, { cls: "ring", label: "focal positives" }]);
})();

// ---------- tables ----------
fill("class-table", A.class_level.map((m) => [m.label, m.focal_auc_primary.toFixed(3), m.soundscape_auc_primary.toFixed(3), m.focal_ap_primary.toFixed(3), m.soundscape_ap_primary.toFixed(3)]));
fill("split-table", A.splits.map((s) => [s.domain, s.split, s.n_clips.toLocaleString("en-US"), s.group_unit, s.n_groups, s.audio, s.labels, s.role]));
{
  const tb = $("species-split-table").querySelector("tbody");
  for (const r of A.species_split) {
    const tr = el("tr", { class: r.primary ? "primary-row" : "" }, el("td", {}, r.code), ...[...r.focal, ...r.soundscape].map((n, i) => el("td", { class: i === 3 ? "sep" : "" }, n)));
    tb.append(tr);
  }
  for (const n of A.split_notes) $("split-notes").append(el("li", {}, n));
}
fill("criteria-table", A.criteria.map((c) => [c.text, c.baseline]));

// ---------- stages and blockers ----------
const KIND = { cpu: "CPU only", "gpu-inference": "GPU inference", training: "GPU training" };
for (const st of D.stages) {
  $("stages").append(el("div", { class: "stage", id: `stage-${st.id}` },
    el("div", { class: "num" }, st.id),
    el("h3", {}, st.title, badge(st.status), el("span", { class: "kind" }, KIND[st.kind] || st.kind)),
    el("ul", {}, st.what.map((w) => el("li", {}, w))),
    el("div", {}, el("p", { class: "verify" }, el("b", {}, "Verify: "), st.verify), st.gate ? el("p", { class: "gate" }, el("b", {}, "Gate: "), st.gate) : null)));
}
for (const b of D.blockers) $("blocker-list").append(el("li", {}, b));

// ---------- training recipe ----------
{
  const R = D.recipe;
  $("recipe-intro").textContent = R.intro;
  const KINDTEXT = { ours: "our data", literature: "literature", hypothesis: "hypothesis", conditional: "conditional" };
  fill("ladder-table", R.ladder.map((r) => [`${r.id} ${r.name}`, r.change, r.isolates, el("span", {}, el("span", { class: `badge ev-${r.kind}` }, KINDTEXT[r.kind]), " ", r.evidence)]));
  for (const c of R.components) {
    $("components").append(el("div", { class: "col-card" }, el("h3", {}, c.name), el("ul", {}, c.params.map((t) => el("li", {}, t))), el("p", { class: "src" }, el("b", {}, "Source: "), c.source)));
  }
  const tb = $("inherited-table").querySelector("tbody");
  for (const [k, v] of R.inherited) tb.append(el("tr", {}, el("td", {}, k), el("td", {}, v)));
  $("recipe-observation").textContent = R.observation;
  for (const t of R.not_now) $("not-now").append(el("li", {}, t));
  for (const t of R.checks) $("recipe-checks").append(el("li", {}, t));
  for (const [f, t] of R.files) $("recipe-files").append(el("li", {}, el("code", {}, f), " ", t));
}

// ---------- figure captions: what it shows + insight ----------
for (const [id, n] of Object.entries(A.figure_notes)) {
  $(id).append(el("p", { class: "note" }, el("span", {}, el("b", {}, "Shows: "), n.shows), el("span", {}, el("b", {}, "Insight: "), n.insight)));
}
