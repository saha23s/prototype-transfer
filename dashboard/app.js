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
    for (const f of DATA.findings) {
      const h = el("h3", { text: f.title });
      if (f.species) h.append(el("span", { class: "tag", text: f.species + (f.prototype ? ` P${f.prototype}` : "") }));
      const art = el("article", { class: "finding" + (f.species ? " clickable" : "") }, h, el("p", { text: f.text }));
      if (f.species) {
        art.tabIndex = 0;
        art.addEventListener("click", () => { select(f.species); document.getElementById("detail-title").scrollIntoView?.({ behavior: "smooth" }); });
        art.addEventListener("keydown", (e) => { if (e.key === "Enter") select(f.species); });
      }
      box.append(art);
    }
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

  function protoCard(sp, p) {
    const d = hasBoth(p) ? p.soundscape.auc - p.focal.auc : null;
    const card = el("div", { class: "proto-card", "data-proto": String(p.k) },
      el("h3", {}, `P${p.k}`, el("small", { text: `weight ${fmt(p.weight)}` })),
      el("div", { class: "row" }, "AUC focal", el("b", { text: fmt(p.focal.auc) })),
      el("div", { class: "row" }, "AUC soundscape", el("b", { text: fmt(p.soundscape.auc) })),
      el("div", { class: "row" }, "Change", el("b", { class: "delta" + (d != null && d < -0.1 ? " neg" : ""), text: signed(d) })),
      el("div", { class: "sub", text: "Activation on positives" }),
      boxPlot(p),
      el("div", { class: "sub", text: "Share of class evidence" }),
    );
    for (const dm of DOMAINS) {
      const v = p[dm].logit_share;
      card.append(el("div", { class: "share" }, LABEL[dm],
        el("div", {}, el("div", { class: "bar " + dm, style: `width:${Math.max(2, (v || 0) * 100)}%` })),
        el("span", { class: "pct", text: pct(v) })));
    }
    card.append(el("div", { class: "row" }, "Peak frequency",
      el("b", { text: `${fmt(p.focal.peak_khz, 1)} → ${fmt(p.soundscape.peak_khz, 1)} kHz` })));
    const patches = sp.patches && sp.patches[String(p.k)];
    if (patches) {
      card.append(el("div", { class: "sub", text: "Top-activating patches" }));
      for (const dm of DOMAINS) {
        const row = el("div", { class: "patches" }, el("span", { text: LABEL[dm] }));
        for (const it of patches[dm]) {
          const cap = `${LABEL[dm]} · activation ${fmt(it.act)} · ${fmt(it.khz, 1)} kHz · ${it.clip}`;
          row.append(el("img", { class: dm, src: it.src, alt: cap, title: cap, loading: "lazy" }));
        }
        card.append(row);
      }
    }
    return card;
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
      el("div", { class: "cards" }, ...sp.prototypes.map((p) => protoCard(sp, p))),
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
    renderFindings();
    renderClassAuc();
    renderScatter();
    renderSelect();
    renderTable();
    renderCaveats();
  });
})();
