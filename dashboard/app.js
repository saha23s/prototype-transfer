// Prototype-transfer dashboard. Data: window.DASHBOARD_DATA (data.js, built by
// scripts/build_dashboard_data.py). Plain DOM + inline SVG, no dependencies.
(() => {
  "use strict";
  const DATA = window.DASHBOARD_DATA;
  const S = DATA.species;
  const SVGNS = "http://www.w3.org/2000/svg";
  const DOMAINS = ["focal", "soundscape"];
  const LABEL = { focal: "Focal", soundscape: "Soundscape" };
  let selected = S[0].code;
  const listeners = []; // re-render hooks run on selection change

  // ---------- helpers ----------
  const fmt = (v, d = 2) => (v == null ? "—" : v.toFixed(d));
  const pct = (v) => (v == null ? "—" : Math.round(v * 100) + "%");
  const signed = (v, d = 2) => (v == null ? "—" : (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v).toFixed(d));
  const bySp = Object.fromEntries(S.map((s) => [s.code, s]));
  const hasBoth = (p) => p.focal.auc != null && p.soundscape.auc != null;

  function el(tag, attrs = {}, ...children) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "text") e.textContent = v;
      else if (k === "html") e.innerHTML = v;
      else e.setAttribute(k, v);
    }
    for (const c of children) if (c != null) e.append(c);
    return e;
  }
  function sv(tag, attrs = {}, text) {
    const e = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    return e;
  }
  const scale = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);

  function legend(container, items) {
    const lg = container.querySelector(".legend");
    for (const [cls, text] of items) lg.append(el("span", {}, el("i", { class: "swatch " + cls }), text));
  }

  // ---------- selection ----------
  function select(code) {
    if (!bySp[code]) return;
    selected = code;
    const sel = document.getElementById("species-select");
    if (sel.value !== code) sel.value = code;
    listeners.forEach((f) => f(code));
  }

  // ---------- tooltip ----------
  const tip = () => document.getElementById("tooltip");
  function showTip(evt, html) {
    const t = tip();
    t.innerHTML = html;
    t.removeAttribute("hidden");
    const x = (evt.clientX || 0) + 14, y = (evt.clientY || 0) + 14;
    const w = t.offsetWidth || 220, h = t.offsetHeight || 60;
    t.style.left = Math.min(x, (window.innerWidth || 1200) - w - 8) + "px";
    t.style.top = Math.min(y, (window.innerHeight || 800) - h - 8) + "px";
  }
  function hideTip() { tip().setAttribute("hidden", ""); }

  function tipFor(target) {
    const sp = bySp[target.dataset.species];
    if (!sp) return null;
    if (target.dataset.proto) {
      const p = sp.prototypes[Number(target.dataset.proto) - 1];
      return `<b>${sp.code} · P${p.k}</b><br>AUC focal ${fmt(p.focal.auc)} → soundscape ${fmt(p.soundscape.auc)}` +
        ` (${signed(p.soundscape.auc - p.focal.auc)})<br>Last-layer weight ${fmt(p.weight)}`;
    }
    return `<b>${sp.code}</b><br>Class AUC focal ${fmt(sp.class_auc.focal, 3)} → soundscape ${fmt(sp.class_auc.soundscape, 3)}` +
      `<br>Positives: ${sp.n.focal} focal / ${sp.n.soundscape} soundscape`;
  }
  function wireChart(svg) {
    svg.addEventListener("mouseover", (e) => {
      const t = e.target.closest("[data-species]");
      if (t) { const h = tipFor(t); if (h) showTip(e, h); }
    });
    svg.addEventListener("mousemove", (e) => {
      const t = e.target.closest("[data-species]");
      if (t && !tip().hidden) showTip(e, tip().innerHTML);
    });
    svg.addEventListener("mouseout", (e) => { if (e.target.closest("[data-species]")) hideTip(); });
    svg.addEventListener("click", (e) => {
      const t = e.target.closest("[data-species]");
      if (t) select(t.dataset.species);
    });
  }

  // ---------- header, tiles, findings, caveats ----------
  function renderHeader() {
    const m = DATA.meta;
    document.getElementById("subtitle").textContent =
      `${m.model} @ ${m.revision.slice(0, 7)} · ${m.dataset} · ${m.n_clips.focal.toLocaleString()} focal and ` +
      `${m.n_clips.soundscape.toLocaleString()} soundscape clips · species with ≥ ${m.min_soundscape_pos} soundscape positives · generated ${m.generated}`;
  }

  function renderSummary() {
    const mean = (k) => S.reduce((a, s) => a + s.class_auc[k], 0) / S.length;
    const protos = S.flatMap((s) => s.prototypes.filter(hasBoth));
    const drops = protos.filter((p) => p.focal.auc - p.soundscape.auc > 0.1).length;
    const tiles = [
      [S.length, "species analysed"],
      [mean("focal").toFixed(3), "mean class AUC · focal"],
      [mean("soundscape").toFixed(3), "mean class AUC · soundscape"],
      [`${drops} / ${protos.length}`, "prototypes losing > 0.1 AUC on soundscapes"],
    ];
    const box = document.getElementById("summary");
    for (const [v, l] of tiles) box.append(el("div", { class: "stat-tile" }, el("span", { class: "value", text: String(v) }), el("span", { class: "label", text: l })));
  }

  function renderFindings() {
    const box = document.getElementById("findings");
    const arts = [];
    DATA.findings.forEach((f, i) => {
      const h = el("h3", { text: f.title });
      if (f.species) h.append(el("span", { class: "tag", text: f.species + (f.prototype ? ` P${f.prototype}` : "") }));
      const art = el("article", { class: "finding clickable", tabindex: "0", "aria-controls": "evidence" },
        h, el("p", { text: f.text }), el("span", { class: "ev-link", text: "Show evidence ↓" }));
      const open = () => {
        arts.forEach((a, j) => a.classList.toggle("active", j === i));
        renderEvidence(f);
        if (f.species) select(f.species);
      };
      art.addEventListener("click", open);
      art.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
      arts.push(art);
      box.append(art);
    });
    // index.html#finding-N opens finding N (1-based); default: the first one
    const m = /finding-(\d+)/.exec(window.location.hash || "");
    const i0 = m ? Math.min(DATA.findings.length, Math.max(1, +m[1])) - 1 : 0;
    arts[i0].classList.add("active");
    renderEvidence(DATA.findings[i0]);
  }

  // ---------- evidence panel ----------
  const SPECIES_ALL = () => S.flatMap((s) => s.prototypes.map((p) => ({ sp: s.code, p })));
  function axes(svg, x, y, xt, yt, { xTicks, yTicks, xFmt = (t) => t, yFmt = (t) => t, box }) {
    for (const t of xTicks) {
      svg.append(sv("line", { class: "gridline", x1: x(t), x2: x(t), y1: box.top, y2: box.bottom }));
      svg.append(sv("text", { x: x(t), y: box.bottom + 15, "text-anchor": "middle" }, xFmt(t)));
    }
    for (const t of yTicks || []) {
      svg.append(sv("line", { class: "gridline", x1: box.left, x2: box.right, y1: y(t), y2: y(t) }));
      svg.append(sv("text", { x: box.left - 6, y: y(t) + 4, "text-anchor": "end" }, yFmt(t)));
    }
    svg.append(sv("text", { class: "axis-title", x: (box.left + box.right) / 2, y: box.bottom + 34, "text-anchor": "middle" }, xt));
    if (yt) svg.append(sv("text", { class: "axis-title", x: 12, y: (box.top + box.bottom) / 2, "text-anchor": "middle",
      transform: `rotate(-90 12 ${(box.top + box.bottom) / 2})` }, yt));
  }
  const evSvg = (W, H, label) => sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": label, class: "ev-chart" });

  function evClassAucDrop(ev) {
    const rows = [...S].map((s) => [s.code, s.class_auc.focal - s.class_auc.soundscape]).sort((a, b) => b[1] - a[1]);
    const W = 560, rowH = 20, box = { left: 70, right: 520, top: 6, bottom: 6 + rows.length * rowH };
    const H = box.bottom + 44, max = Math.max(0.05, ...rows.map((r) => r[1]));
    const x = scale(Math.min(0, ...rows.map((r) => r[1])), max, box.left, box.right);
    const svg = evSvg(W, H, "Class AUC drop per species");
    axes(svg, x, null, "Class AUC drop (focal − soundscape)", null, { xTicks: [0, 0.1, 0.2, 0.3].filter((t) => t <= max + 1e-9), xFmt: (t) => t.toFixed(1), box });
    rows.forEach(([code, d], i) => {
      const cy = box.top + i * rowH + rowH / 2, hl = ev.highlight.includes(code);
      svg.append(sv("text", { class: "row-label" + (hl ? " sel" : ""), x: box.left - 8, y: cy + 4, "text-anchor": "end" }, code));
      svg.append(sv("rect", { class: "ev-bar" + (hl ? " hl" : ""), x: Math.min(x(0), x(d)), y: cy - 6, width: Math.max(2, Math.abs(x(d) - x(0))), height: 12, rx: 3, "data-species": code }));
      svg.append(sv("text", { x: Math.max(x(0), x(d)) + 5, y: cy + 4 }, d.toFixed(3)));
    });
    return svg;
  }

  function evDeltaAuc(ev) {
    const rows = S.map((s) => [s.code, s.prototypes.filter(hasBoth).map((p) => [p.k, p.soundscape.auc - p.focal.auc])])
      .sort((a, b) => (ev.spread[b[0]] ?? 0) - (ev.spread[a[0]] ?? 0));
    const W = 560, rowH = 20, box = { left: 70, right: 505, top: 20, bottom: 20 + rows.length * rowH };
    const H = box.bottom + 44;
    const x = scale(-0.5, 0.3, box.left, box.right);
    const svg = evSvg(W, H, "Per-prototype AUC change by species");
    axes(svg, x, null, "Prototype ΔAUC (soundscape − focal)", null,
      { xTicks: [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3], xFmt: (t) => (t > 0 ? "+" : "") + t.toFixed(1), box });
    svg.append(sv("line", { class: "diag", x1: x(ev.threshold), x2: x(ev.threshold), y1: box.top, y2: box.bottom }));
    rows.forEach(([code, ds], i) => {
      const cy = box.top + i * rowH + rowH / 2, vals = ds.map((d) => d[1]);
      svg.append(sv("text", { class: "row-label", x: box.left - 8, y: cy + 4, "text-anchor": "end" }, code));
      svg.append(sv("line", { class: "connector", x1: x(Math.min(...vals)), x2: x(Math.max(...vals)), y1: cy, y2: cy }));
      for (const [k, d] of ds) svg.append(sv("circle", { class: "pt" + (d < ev.threshold ? " sel" : ""), cx: x(d), cy, r: 4.5, "data-species": code, "data-proto": k }));
      svg.append(sv("text", { x: box.right + 12, y: cy + 4 }, (ev.spread[code] ?? 0).toFixed(2)));
    });
    svg.append(sv("text", { x: box.right + 12, y: 12 }, "spread"));
    return svg;
  }

  function evActivationCdf(ev) {
    const c = ev.cdf, W = 560, H = 300, box = { left: 50, right: 540, top: 10, bottom: 250 };
    const x = scale(0, 0.6, box.left, box.right), y = scale(0, 1, box.bottom, box.top);
    const svg = evSvg(W, H, "Cumulative distribution of prototype activations");
    axes(svg, x, y, "Prototype max activation", "Fraction of entries ≤ x",
      { xTicks: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], yTicks: [0, 0.25, 0.5, 0.75, 1], xFmt: (t) => t.toFixed(1), yFmt: (t) => Math.round(t * 100) + "%", box });
    for (const [key, cls] of [["focal_neg", "focal dashed"], ["soundscape_neg", "soundscape dashed"], ["focal_pos", "focal"], ["soundscape_pos", "soundscape"]]) {
      const pts = c.x.map((v, i) => [v, c.series[key][i]]).filter(([v]) => v <= 0.6);
      svg.append(sv("path", { class: "cdf " + cls, d: "M" + pts.map(([a, b]) => `${x(a).toFixed(1)},${y(b).toFixed(1)}`).join("L") }));
      const m = c.medians[key];
      svg.append(sv("circle", { class: "dot " + cls.split(" ")[0], cx: x(m), cy: y(0.5), r: 4 }));
    }
    svg.append(sv("text", { x: box.right, y: y(0.5) - 6, "text-anchor": "end" }, "dots = medians"));
    return svg;
  }

  function evScatter(ev, { get, lo, hi, ticks, fmt, log, xt, yt, band }) {
    const W = 560, H = 420, box = { left: 56, right: 540, top: 10, bottom: 370 };
    const tr = log ? Math.log10 : (v) => v;
    const x0 = scale(tr(lo), tr(hi), box.left, box.right), y0 = scale(tr(lo), tr(hi), box.bottom, box.top);
    const x = (v) => x0(tr(Math.max(lo, v))), y = (v) => y0(tr(Math.max(lo, v)));
    const svg = evSvg(W, H, xt + " vs " + yt);
    if (band) {
      svg.append(sv("rect", { class: "band", x: box.left, y: y(band[1]), width: box.right - box.left, height: y(band[0]) - y(band[1]) }));
      svg.append(sv("text", { x: box.right - 6, y: y(band[1]) + 14, "text-anchor": "end" }, band[2]));
    }
    axes(svg, x, y, xt, yt, { xTicks: ticks, yTicks: ticks, xFmt: fmt, yFmt: fmt, box });
    svg.append(sv("line", { class: "diag", x1: x(lo), y1: y(lo), x2: x(hi), y2: y(hi) }));
    const hl = new Set(ev.highlight.map(([sp, k]) => sp + "|" + k));
    const items = SPECIES_ALL().map(({ sp, p }, i) => ({ sp, p, v: get(p), on: hl.has(sp + "|" + p.k), j: i })).filter((d) => d.v);
    // deterministic jitter so quantised values (e.g. peak rows) don't stack exactly
    const jit = (i, a) => (((i * 9301 + a * 49297) % 233280) / 233280 - 0.5) * 6;
    const labels = [];
    for (const d of items.sort((a, b) => a.on - b.on)) {
      const cx = x(d.v[0]) + jit(d.j, 1), cy = y(d.v[1]) + jit(d.j, 2);
      svg.append(sv("circle", { class: "pt" + (d.on ? " sel" : ""), cx, cy, r: d.on ? 6 : 4.5, "data-species": d.sp, "data-proto": d.p.k }));
      if (d.on) labels.push({ cx, cy, text: `${d.sp} P${d.p.k}` });
    }
    // Stack highlighted labels upward from the lowest point so coincident points stay readable.
    labels.sort((a, b) => b.cy - a.cy || a.cx - b.cx);
    let lastY = Infinity;
    const lx = Math.min(box.right - 70, Math.max(...labels.map((l) => l.cx)) + 24);
    for (const l of labels) {
      const ly = Math.min(l.cy + 4, lastY - 14);
      lastY = ly;
      svg.append(sv("line", { class: "leader", x1: l.cx + 6, y1: l.cy, x2: lx - 3, y2: ly - 4 }));
      svg.append(sv("text", { class: "pt-label", x: lx, y: ly }, l.text));
    }
    return svg;
  }

  function evPatches(ev) {
    const sp = bySp[ev.species];
    const wrap = el("div", { class: "ev-patches" });
    for (const k of ev.prototypes) {
      const p = sp.prototypes[k - 1], pt = sp.patches && sp.patches[String(k)];
      wrap.append(el("div", { class: "ev-patch-label" }, el("b", { text: `${sp.code} P${k}` }),
        ` · AUC ${fmt(p.focal.auc)} → ${fmt(p.soundscape.auc)} · peak ${fmt(p.focal.peak_khz, 1)} → ${fmt(p.soundscape.peak_khz, 1)} kHz`));
      if (pt) {
        const row = el("div", { class: "col-patches" });
        for (const dm of DOMAINS) row.append(patchGroup(pt, dm));
        wrap.append(row);
      }
    }
    return wrap;
  }

  function evidenceStats(f) {
    if (f.evidence.stats) return f.evidence.stats;
    const sp = bySp[f.evidence.species];
    return f.evidence.prototypes.flatMap((k) => {
      const p = sp.prototypes[k - 1];
      return [[`${sp.code} P${k} AUC (focal → soundscape)`, `${fmt(p.focal.auc, 3)} → ${fmt(p.soundscape.auc, 3)}`],
              [`${sp.code} P${k} share of evidence`, `${pct(p.focal.logit_share)} → ${pct(p.soundscape.logit_share)}`],
              [`${sp.code} P${k} median activation on positives`, `${fmt(p.focal.act_q[2], 3)} → ${fmt(p.soundscape.act_q[2], 3)}`]];
    });
  }

  function renderEvidence(f) {
    const ev = f.evidence, panel = document.getElementById("evidence");
    let chart, legendItems = [];
    if (ev.kind === "class_auc_drop") chart = evClassAucDrop(ev);
    else if (ev.kind === "delta_auc") { chart = evDeltaAuc(ev); legendItems = [["sel", "ΔAUC < −0.10 (dashed line)"], ["other", "other prototypes"]]; }
    else if (ev.kind === "activation_cdf") {
      chart = evActivationCdf(ev);
      legendItems = [["focal", "Focal, clips with species"], ["soundscape", "Soundscape, clips with species"],
                     ["focal dash", "Focal, clips without (dashed)"], ["soundscape dash", "Soundscape, clips without (dashed)"]];
    } else if (ev.kind === "share_scatter") {
      chart = evScatter(ev, { get: (p) => (p.focal.logit_share != null && p.soundscape.logit_share != null ? [p.focal.logit_share, p.soundscape.logit_share] : null),
        lo: 0, hi: 0.45, ticks: [0, 0.1, 0.2, 0.3, 0.4], fmt: (t) => Math.round(t * 100) + "%", xt: "Share of class evidence · focal", yt: "Share · soundscape" });
      legendItems = [["sel", "Largest shifts"], ["other", "Other prototypes"]];
    } else if (ev.kind === "peak_scatter") {
      chart = evScatter(ev, { get: (p) => (p.focal.peak_khz != null && p.soundscape.peak_khz != null ? [p.focal.peak_khz, p.soundscape.peak_khz] : null),
        lo: 0.05, hi: 16, ticks: [0.1, 0.25, 0.5, 1, 2, 4, 8, 16], fmt: (t) => String(t), log: true,
        xt: "Peak frequency · focal (kHz, log)", yt: "Peak frequency · soundscape (kHz, log)", band: [0.05, 0.5, "wind / ambient band (< 0.5 kHz)"] });
      legendItems = [["sel", "Drifting to the noise band"], ["other", "Other prototypes"]];
    } else if (ev.kind === "patches") {
      chart = evPatches(ev);
      legendItems = [["focal", "Focal patches"], ["soundscape", "Soundscape patches"]];
    }
    const lg = el("div", { class: "legend" });
    for (const [cls, text] of legendItems) lg.append(el("span", {}, el("i", { class: "swatch " + cls }), text));
    const stats = el("table", { class: "ev-stats" }, el("tbody", {}, ...evidenceStats(f).map(([k, v]) => el("tr", {}, el("td", { text: k }), el("td", { text: v })))));
    panel.replaceChildren(
      el("h3", { text: "Evidence · " + f.title }),
      el("div", { class: "ev-body" },
        el("div", { class: "ev-chart-wrap" }, lg, chart),
        el("div", { class: "ev-side" }, stats, el("p", { class: "ev-method", text: ev.method }))));
    panel.querySelectorAll("svg").forEach(wireChart);
  }

  function renderCaveats() {
    const ul = document.getElementById("caveats");
    for (const c of DATA.caveats) ul.append(el("li", { text: c }));
  }

  // ---------- chart 1: class AUC dumbbell ----------
  function renderClassAuc() {
    const fig = document.getElementById("class-auc");
    legend(fig, [["focal", "Focal (Xeno-Canto)"], ["soundscape", "Soundscape (PAM)"]]);
    const rows = [...S].sort((a, b) => (b.class_auc.focal - b.class_auc.soundscape) - (a.class_auc.focal - a.class_auc.soundscape));
    const W = 480, rowH = 24, top = 8, left = 70, right = 16, bottom = 44;
    const H = top + rows.length * rowH + bottom;
    const x = scale(0.5, 1.0, left, W - right);
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Class AUC per species, focal vs soundscape" });
    const yAxis = top + rows.length * rowH;
    for (let t = 0.5; t <= 1.0001; t += 0.1) {
      svg.append(sv("line", { class: "gridline", x1: x(t), x2: x(t), y1: top, y2: yAxis }));
      svg.append(sv("text", { x: x(t), y: yAxis + 16, "text-anchor": "middle" }, t.toFixed(1)));
    }
    svg.append(sv("text", { class: "axis-title", x: (left + W - right) / 2, y: yAxis + 36, "text-anchor": "middle" }, "Class ROC-AUC (0.5 = chance)"));
    const labels = {}, hits = {};
    rows.forEach((s, i) => {
      const cy = top + i * rowH + rowH / 2;
      const g = sv("g", { "data-species": s.code });
      hits[s.code] = sv("rect", { class: "row-hit", x: 0, y: cy - rowH / 2, width: W, height: rowH, rx: 4 });
      g.append(hits[s.code]);
      labels[s.code] = sv("text", { class: "row-label", x: left - 10, y: cy + 4, "text-anchor": "end" }, s.code);
      g.append(labels[s.code]);
      g.append(sv("line", { class: "connector", x1: x(s.class_auc.focal), x2: x(s.class_auc.soundscape), y1: cy, y2: cy }));
      for (const d of DOMAINS) {
        g.append(sv("circle", { class: "dot " + d, cx: x(Math.max(0.5, s.class_auc[d])), cy, r: 5.5, "data-species": s.code, "data-domain": d }));
      }
      svg.append(g);
    });
    fig.append(svg);
    wireChart(svg);
    const update = (code) => {
      for (const c in labels) {
        labels[c].classList.toggle("sel", c === code);
        hits[c].classList.toggle("sel", c === code);
      }
    };
    listeners.push(update);
    update(selected);
  }

  // ---------- chart 2: prototype AUC scatter ----------
  function renderScatter() {
    const fig = document.getElementById("proto-scatter");
    legend(fig, [["sel", "Selected species"], ["other", "Other species"]]);
    const W = 460, H = 420, left = 52, right = 14, top = 10, bottom = 48;
    const LO = 0.5;
    const x = scale(LO, 1.0, left, W - right), y = scale(LO, 1.0, H - bottom, top);
    const wmax = Math.max(...S.flatMap((s) => s.prototypes.map((p) => p.weight)));
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Prototype AUC, focal vs soundscape" });
    for (let t = LO; t <= 1.0001; t += 0.1) {
      svg.append(sv("line", { class: "gridline", x1: x(t), x2: x(t), y1: top, y2: H - bottom }));
      svg.append(sv("line", { class: "gridline", x1: left, x2: W - right, y1: y(t), y2: y(t) }));
      svg.append(sv("text", { x: x(t), y: H - bottom + 16, "text-anchor": "middle" }, t.toFixed(1)));
      svg.append(sv("text", { x: left - 8, y: y(t) + 4, "text-anchor": "end" }, t.toFixed(1)));
    }
    svg.append(sv("line", { class: "diag", x1: x(LO), y1: y(LO), x2: x(1), y2: y(1) }));
    const dl = LO + 0.04, ang = Math.atan2(y(LO) - y(1), x(1) - x(LO)) * 180 / Math.PI;
    svg.append(sv("text", { x: x(dl), y: y(dl) - 6, transform: `rotate(-${ang} ${x(dl)} ${y(dl) - 6})` }, "no change"));
    svg.append(sv("text", { class: "axis-title", x: (left + W - right) / 2, y: H - 10, "text-anchor": "middle" }, "Prototype AUC · focal"));
    svg.append(sv("text", { class: "axis-title", x: 14, y: (top + H - bottom) / 2, "text-anchor": "middle", transform: `rotate(-90 14 ${(top + H - bottom) / 2})` }, "Prototype AUC · soundscape"));
    const ptsLayer = sv("g"), labelLayer = sv("g");
    svg.append(ptsLayer, labelLayer);
    const pts = [];
    for (const s of S) for (const p of s.prototypes.filter(hasBoth)) {
      const c = sv("circle", {
        class: "pt", cx: x(p.focal.auc), cy: y(p.soundscape.auc), r: 4 + 5 * (p.weight / wmax),
        "data-species": s.code, "data-proto": p.k,
      });
      pts.push(c);
      ptsLayer.append(c);
    }
    fig.append(svg);
    wireChart(svg);
    const update = (code) => {
      labelLayer.replaceChildren();
      for (const c of pts) {
        const on = c.dataset.species === code;
        c.classList.toggle("sel", on);
        if (on) {
          ptsLayer.append(c); // bring to front
          labelLayer.append(sv("text", { class: "pt-label", x: +c.getAttribute("cx") + 8, y: +c.getAttribute("cy") + 3 }, "P" + c.dataset.proto));
        }
      }
    };
    listeners.push(update);
    update(selected);
  }

  // ---------- species detail ----------
  function boxPlot(p) {
    const W = 200, H = 64, l = 8, r = 8;
    const x = scale(0, 0.6, l, W - r);
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Activation distribution on positive clips" });
    for (const t of [0, 0.2, 0.4, 0.6]) {
      svg.append(sv("line", { class: "gridline", x1: x(t), x2: x(t), y1: 2, y2: 46 }));
      svg.append(sv("text", { x: x(t), y: 60, "text-anchor": "middle" }, t.toFixed(1)));
    }
    DOMAINS.forEach((d, i) => {
      const q = p[d].act_q, cy = 13 + i * 22, c = (v) => x(Math.min(0.6, Math.max(0, v)));
      svg.append(sv("line", { class: "whisker " + d, x1: c(q[0]), x2: c(q[4]), y1: cy, y2: cy }));
      svg.append(sv("rect", { class: "box " + d, x: c(q[1]), y: cy - 6, width: Math.max(2, c(q[3]) - c(q[1])), height: 12, rx: 3 }));
      svg.append(sv("line", { class: "median", x1: c(q[2]), x2: c(q[2]), y1: cy - 6, y2: cy + 6 }));
      svg.append(sv("line", { class: "neg", x1: c(p[d].neg_median), x2: c(p[d].neg_median), y1: cy - 8, y2: cy + 8 }));
    });
    return svg;
  }

  // One horizontal row per prototype: [name + AUCs] [activation] [share + peak] [patches]
  function protoCard(sp, p) {
    const d = hasBoth(p) ? p.soundscape.auc - p.focal.auc : null;
    const metrics = el("div", { class: "col col-metrics" },
      el("h3", {}, `P${p.k}`, el("small", { text: `weight ${fmt(p.weight)}` })),
      el("div", { class: "row" }, el("span", {}, el("i", { class: "swatch focal" }), " AUC focal"), el("b", { text: fmt(p.focal.auc) })),
      el("div", { class: "row" }, el("span", {}, el("i", { class: "swatch soundscape" }), " AUC soundscape"), el("b", { text: fmt(p.soundscape.auc) })),
      el("div", { class: "row" }, "Change", el("b", { class: "delta" + (d != null && d < -0.1 ? " neg" : ""), text: signed(d) })));
    const act = el("div", { class: "col col-act" }, boxPlot(p));
    const share = el("div", { class: "col col-share" });
    for (const dm of DOMAINS) {
      const v = p[dm].logit_share;
      share.append(el("div", { class: "share" }, LABEL[dm],
        el("div", {}, el("div", { class: "bar " + dm, style: `width:${Math.max(2, (v || 0) * 100)}%` })),
        el("span", { class: "pct", text: pct(v) })));
    }
    const low = p.soundscape.peak_khz != null && p.soundscape.peak_khz < 0.5 && p.focal.peak_khz > 1.0;
    share.append(el("div", { class: "row peak" }, "Peak freq.",
      el("b", { text: `${fmt(p.focal.peak_khz, 1)} → ${fmt(p.soundscape.peak_khz, 1)} kHz` })));
    if (low) share.append(el("div", { class: "flag", text: "soundscape peak in noise band" }));
    const pcol = el("div", { class: "col col-patches" });
    const patches = sp.patches && sp.patches[String(p.k)];
    if (patches) for (const dm of DOMAINS) pcol.append(patchGroup(patches, dm));
    return el("div", { class: "proto-card", "data-proto": String(p.k) }, metrics, act, share, pcol);
  }

  function patchGroup(patches, dm) {
    const row = el("div", { class: "patches" }, el("span", { text: LABEL[dm] }));
    for (const it of patches[dm]) {
      const cap = `${LABEL[dm]} · activation ${fmt(it.act)} · ${fmt(it.khz, 1)} kHz · ${it.clip}`;
      row.append(el("figure", {}, el("img", { class: "zoomable " + dm, src: it.src, alt: cap, "data-caption": cap, loading: "lazy" }),
        el("figcaption", { text: `${fmt(it.act)} · ${fmt(it.khz, 1)}` })));
    }
    return row;
  }

  // Hover any patch to see it enlarged in the shared tooltip.
  function wireZoom() {
    document.addEventListener("mouseover", (e) => {
      const img = e.target.closest && e.target.closest("img.zoomable");
      if (!img) return;
      const t = tip();
      t.classList.add("zoom-tip");
      showTip(e, `<img class="zoom-img ${img.classList.contains("focal") ? "focal" : "soundscape"}" src="${img.getAttribute("src")}" alt=""><div>${img.dataset.caption}</div>`);
    });
    document.addEventListener("mousemove", (e) => {
      if (e.target.closest && e.target.closest("img.zoomable") && !tip().hidden) showTip(e, tip().innerHTML);
    });
    document.addEventListener("mouseout", (e) => {
      if (e.target.closest && e.target.closest("img.zoomable")) { hideTip(); tip().classList.remove("zoom-tip"); }
    });
  }

  function detailHeader() {
    return el("div", { class: "proto-head", "aria-hidden": "true" },
      el("span", { text: "Prototype · detection AUC" }),
      el("span", { text: "Activation on positives (0–0.6)" }),
      el("span", { text: "Share of class evidence · peak" }),
      el("span", { text: "Top-activating patches (activation · kHz)" }));
  }

  function renderDetail(code) {
    const sp = bySp[code];
    const box = document.getElementById("species-detail");
    box.replaceChildren(
      el("p", { class: "detail-head", html:
        `<strong>${sp.code}</strong> · ${sp.n.focal} focal / ${sp.n.soundscape} soundscape positives · ` +
        `class AUC <strong>${fmt(sp.class_auc.focal, 3)}</strong> focal → <strong>${fmt(sp.class_auc.soundscape, 3)}</strong> soundscape` +
        (sp.n.soundscape < 30 ? " · <em>few soundscape positives: noisy</em>" : "") }),
      el("div", { class: "detail-key" },
        el("span", {}, el("i", { class: "swatch focal" }), "Focal"),
        el("span", {}, el("i", { class: "swatch soundscape" }), "Soundscape"),
        el("span", { text: "Box = q25–q75, whiskers = q10–q90, white tick = median" }),
        el("span", {}, el("i", { class: "tick-key" }), "median on clips without the species")),
      el("div", { class: "cards" }, detailHeader(), ...sp.prototypes.map((p) => protoCard(sp, p))),
    );
  }

  function renderSelect() {
    const sel = document.getElementById("species-select");
    for (const s of S) sel.append(el("option", { value: s.code, text: `${s.code} (${s.n.soundscape} soundscape)` }));
    sel.value = selected;
    sel.addEventListener("change", () => select(sel.value));
    listeners.push(renderDetail);
    renderDetail(selected);
  }

  // ---------- table ----------
  function renderTable() {
    const tb = document.querySelector("#proto-table tbody");
    for (const s of S) for (const p of s.prototypes) {
      const cells = [s.code, "P" + p.k, fmt(p.weight), fmt(p.focal.auc, 3), fmt(p.soundscape.auc, 3),
        hasBoth(p) ? signed(p.soundscape.auc - p.focal.auc, 3) : "—",
        fmt(p.focal.act_q[2], 3), fmt(p.soundscape.act_q[2], 3), pct(p.focal.logit_share), pct(p.soundscape.logit_share)];
      tb.append(el("tr", {}, ...cells.map((c) => el("td", { text: c }))));
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    renderHeader();
    renderSummary();
    wireZoom();
    renderClassAuc();
    renderScatter();
    renderSelect();
    renderFindings();
    renderTable();
    renderCaveats();
  });
})();
