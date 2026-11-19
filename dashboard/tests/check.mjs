// Acceptance check for the prototype-transfer dashboard (jsdom render + interactions).
// Usage: npm i jsdom@24 && node dashboard/tests/check.mjs dashboard
import { JSDOM, VirtualConsole } from "jsdom";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const dir = path.resolve(process.argv[2] || "dashboard");
const failures = [];
const ok = (cond, msg) => { if (!cond) failures.push(msg); else console.log("  ok  " + msg); };

for (const f of ["index.html", "app.js", "style.css", "data.js"]) ok(fs.existsSync(path.join(dir, f)), `exists: ${f}`);
if (failures.length) { console.error(failures.join("\n")); process.exit(1); }

const html = fs.readFileSync(path.join(dir, "index.html"), "utf8");
const js = fs.readFileSync(path.join(dir, "app.js"), "utf8");
const css = fs.readFileSync(path.join(dir, "style.css"), "utf8");
for (const [name, src] of [["index.html", html], ["app.js", js], ["style.css", css]]) {
  ok(!/https?:\/\//i.test(src.replace(/http:\/\/www\.w3\.org\/[^"'\s]*/g, "")), `no external URLs in ${name}`);
}
ok(!/\bfetch\s*\(/.test(js), "app.js does not use fetch()");
const iCss = html.indexOf("style.css"), iData = html.indexOf("data.js"), iApp = html.indexOf("app.js");
ok(iCss >= 0 && iData > iCss && iApp > iData, "index.html loads style.css, data.js, app.js in order");
ok(/prefers-color-scheme:\s*dark/.test(css), "style.css has a dark-mode media query");
ok(/#2a78d6/i.test(css + js) && /#eb6834/i.test(css + js), "domain colors #2a78d6 / #eb6834 used");

const errors = [];
const vc = new VirtualConsole();
vc.on("jsdomError", (e) => errors.push(String(e && (e.stack || e.message || e))));
vc.on("error", (e) => errors.push("console.error: " + String(e)));
const dom = await JSDOM.fromFile(path.join(dir, "index.html"), {
  runScripts: "dangerously", resources: "usable", pretendToBeVisual: true, virtualConsole: vc,
  url: pathToFileURL(path.join(dir, "index.html")).href,
});
await new Promise((res) => {
  if (dom.window.document.readyState === "complete") return res();
  dom.window.addEventListener("load", res);
});
await new Promise((r) => setTimeout(r, 300));

const w = dom.window, d = w.document;
const data = w.DASHBOARD_DATA;
ok(!!data && Array.isArray(data.species), "window.DASHBOARD_DATA loaded");
if (!data) { console.error(failures.join("\n")); process.exit(1); }
const S = data.species, codes = S.map((s) => s.code);
const nBoth = S.reduce((a, s) => a + s.prototypes.filter((p) => p.focal.auc != null && p.soundscape.auc != null).length, 0);

ok(errors.length === 0, `no script errors (${errors.slice(0, 3).join(" | ")})`);
ok(d.querySelectorAll("#summary .stat-tile").length >= 4, "#summary has >= 4 .stat-tile");
ok(d.querySelectorAll("#findings .finding").length === data.findings.length, `#findings has ${data.findings.length} .finding`);
ok(!!d.querySelector("#class-auc svg"), "#class-auc contains an svg");
for (const dom_ of ["focal", "soundscape"]) {
  ok(d.querySelectorAll(`#class-auc circle[data-domain="${dom_}"][data-species]`).length === S.length,
     `#class-auc has ${S.length} ${dom_} circles`);
}
const sc = d.querySelectorAll("#proto-scatter circle[data-species][data-proto]");
ok(sc.length === nBoth, `#proto-scatter has ${nBoth} prototype circles (got ${sc.length})`);
ok(/no change/i.test(d.querySelector("#proto-scatter")?.textContent || ""), "#proto-scatter labels the diagonal 'no change'");

const sel = d.querySelector("#species-select");
ok(!!sel, "#species-select exists");
const optVals = sel ? [...sel.options].map((o) => o.value) : [];
ok(JSON.stringify(optVals) === JSON.stringify(codes), "#species-select options match species order");
ok(sel && sel.value === codes[0], "default selection is the first species");

function detailOk(code, label) {
  const det = d.querySelector("#species-detail");
  const cards = det ? det.querySelectorAll(".proto-card[data-proto]") : [];
  ok(det && det.textContent.includes(code), `${label}: #species-detail mentions ${code}`);
  ok(cards.length === 5 && [...cards].map((c) => c.dataset.proto).join() === "1,2,3,4,5", `${label}: 5 .proto-card with data-proto 1..5`);
  const sp = S.find((s) => s.code === code);
  if (sp && sp.patches && cards.length === 5) {
    const imgs = cards[0].querySelectorAll("img");
    const exp = sp.patches["1"].focal.length + sp.patches["1"].soundscape.length + (sp.patches["1"].soundscape_neg || []).length;
    ok((sp.patches["1"].soundscape_neg || []).length > 0 && cards[0].querySelectorAll("img.zoomable.neg").length === sp.patches["1"].soundscape_neg.length,
       `${label}: card P1 shows soundscape-negative patches`);
    ok(imgs.length === exp && [...imgs].every((i) => i.getAttribute("src").startsWith(`patches/${code}_P1_`)),
       `${label}: card P1 shows ${exp} patch images for ${code}`);
    ok(cards[0].querySelectorAll("svg").length >= 1, `${label}: card has an svg box plot`);
  }
}
detailOk(codes[0], "default");

if (sel) {
  const target = codes.includes("foxspa") ? "foxspa" : codes[codes.length - 1];
  sel.value = target;
  sel.dispatchEvent(new w.Event("change", { bubbles: true }));
  await new Promise((r) => setTimeout(r, 50));
  detailOk(target, "after select change");
}

const pick = codes[1] || codes[0];
const circ = d.querySelector(`#proto-scatter circle[data-species="${pick}"]`);
if (circ) {
  circ.dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
  await new Promise((r) => setTimeout(r, 50));
  ok(sel.value === pick, `clicking a scatter circle selects ${pick}`);
  detailOk(pick, "after scatter click");
  const tip = d.querySelector("#tooltip");
  ok(!!tip, "#tooltip exists");
  circ.dispatchEvent(new w.MouseEvent("mouseover", { bubbles: true, clientX: 100, clientY: 100 }));
  ok(tip && !tip.hasAttribute("hidden") && tip.textContent.includes(pick), "tooltip shows species on mouseover");
  circ.dispatchEvent(new w.MouseEvent("mouseout", { bubbles: true }));
  ok(tip && tip.hasAttribute("hidden"), "tooltip hidden on mouseout");
}
const cdot = d.querySelector(`#class-auc circle[data-species="${codes[2] || codes[0]}"]`);
if (cdot) {
  cdot.dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
  await new Promise((r) => setTimeout(r, 50));
  ok(sel.value === (codes[2] || codes[0]), "clicking a class-auc dot selects its species");
}
const fWithSp = [...d.querySelectorAll("#findings .finding")].findIndex((_, i) => data.findings[i].species);
if (fWithSp >= 0) {
  d.querySelectorAll("#findings .finding")[fWithSp].dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
  await new Promise((r) => setTimeout(r, 50));
  ok(sel.value === data.findings[fWithSp].species, "clicking a finding selects its species");
}

// Every finding must carry data support: evidence spec in data, and a rendered chart/patches + stats.
for (let i = 0; i < data.findings.length; i++) {
  const f = data.findings[i];
  ok(!!f.evidence && !!f.evidence.kind && !!f.evidence.method, `finding ${i + 1} has evidence {kind, method} in data`);
  d.querySelectorAll("#findings .finding")[i].dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
  await new Promise((r) => setTimeout(r, 30));
  const ev = d.querySelector("#evidence");
  ok(ev && ev.querySelector("h3")?.textContent.includes(f.title), `finding ${i + 1}: #evidence shows its title`);
  const visual = ev ? ev.querySelectorAll("svg.ev-chart, img.zoomable").length : 0;
  ok(visual > 0, `finding ${i + 1} (${f.evidence?.kind}): evidence has a chart or patches`);
  ok(ev && ev.querySelectorAll(".ev-stats tr").length >= 1, `finding ${i + 1}: evidence has a stats table`);
  if (f.evidence?.kind === "activation_cdf") {
    ok(ev.querySelectorAll("svg.ev-chart path.cdf").length === 4, `finding ${i + 1}: evidence plots 4 activation CDFs`);
  } else if (f.evidence?.kind && f.evidence.kind !== "patches" && f.evidence.kind !== "class_auc_drop") {
    ok(ev.querySelectorAll("svg.ev-chart [data-species]").length >= 14, `finding ${i + 1}: evidence chart plots the data`);
  }
}
// Per-species activation distributions in the detail panel, with view switching.
{
  const det = d.querySelector("#species-detail");
  const code = sel.value, sp = S.find((x) => x.code === code);
  ok(!!sp.act_cdf && ["all", "max", "1", "5"].every((v) => sp.act_cdf.series[v]), "data has per-species act_cdf views");
  ok(det.querySelectorAll(".sp-dist svg path.cdf").length === 4, "species detail shows 4 activation CDFs");
  const before = det.querySelector('.sp-dist path.cdf[data-series="soundscape_pos"]')?.getAttribute("d");
  const b3 = det.querySelector('.sp-dist .seg button[data-view="3"]');
  ok(!!b3, "distribution view selector has a P3 button");
  b3?.dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
  const after = det.querySelector('.sp-dist path.cdf[data-series="soundscape_pos"]')?.getAttribute("d");
  ok(before && after && before !== after && b3.getAttribute("aria-selected") === "true", "switching to P3 redraws the distributions");
  ok(det.querySelector(".sp-dist .ev-stats")?.textContent.includes("P3 AUC"), "P3 view shows P3 AUC in stats");
}

const zimg = d.querySelector("#species-detail img.zoomable");
const ztip = d.querySelector("#tooltip");
if (zimg) {
  zimg.dispatchEvent(new w.MouseEvent("mouseover", { bubbles: true, clientX: 50, clientY: 50 }));
  ok(!ztip.hasAttribute("hidden") && ztip.querySelector("img.zoom-img")?.getAttribute("src") === zimg.getAttribute("src"),
     "hovering a patch shows the enlarged image");
  zimg.dispatchEvent(new w.MouseEvent("mouseout", { bubbles: true }));
  ok(ztip.hasAttribute("hidden"), "zoom hides on mouseout");
} else ok(false, "patch images are zoomable");

// Model comparison (test split): table + per-species AP chart with a domain toggle.
{
  const mc = data.model_comparison;
  ok(!!mc && mc.models.length === 3 && mc.species.length > 0, "data has model_comparison with 3 models");
  const sec = d.querySelector("#model-comparison");
  ok(!!sec && /test/i.test(sec.textContent), "#model-comparison section exists and says it is the test split");
  const rows = [...d.querySelectorAll("#cmp-table tbody tr[data-model]")];
  ok(rows.length === 3 && rows.map((r) => r.dataset.model).join() === mc.models.map((m) => m.name).join(),
     "#cmp-table has one row per model, in data order");
  for (const m of mc.models) {
    const r = rows.find((x) => x.dataset.model === m.name);
    ok(r && r.textContent.includes(m.label) && r.textContent.includes(m.focal.cmAP.toFixed(3)) && r.textContent.includes(m.soundscape.cmAP.toFixed(3)),
       `#cmp-table row ${m.name} shows label and both cmAPs (3 dp)`);
  }
  const bestSc = mc.models.reduce((a, b) => (b.soundscape.cmAP > a.soundscape.cmAP ? b : a));
  const bestFo = mc.models.reduce((a, b) => (b.focal.cmAP > a.focal.cmAP ? b : a));
  const bestTxt = (name) => [...(rows.find((x) => x.dataset.model === name)?.querySelectorAll("td.best") || [])].map((t) => t.textContent).join("|");
  ok(bestTxt(bestSc.name).includes(bestSc.soundscape.cmAP.toFixed(3)), `soundscape cmAP best cell is marked .best in ${bestSc.name} row`);
  ok(bestTxt(bestFo.name).includes(bestFo.focal.cmAP.toFixed(3)), `focal cmAP best cell is marked .best in ${bestFo.name} row`);
  ok(d.querySelectorAll("#cmp-table td.best").length >= 6 && d.querySelectorAll("#cmp-table td.best").length <= 8, "#cmp-table marks one best cell per metric column (6 columns)");

  const circles = () => [...d.querySelectorAll("#cmp-species svg circle[data-model][data-species]")];
  ok(circles().length === 3 * mc.species.length, `#cmp-species has ${3 * mc.species.length} circles`);
  const valOk = (dom_) => mc.models.every((m) => mc.species.every((sp, i) => {
    const c = d.querySelector(`#cmp-species circle[data-model="${m.name}"][data-species="${sp}"]`);
    return c && Math.abs(Number(c.dataset.value) - m[dom_].per_species_ap[i]) < 1e-3;
  }));
  const segBtn = (dom_) => d.querySelector(`#cmp-species .seg button[data-domain="${dom_}"]`);
  ok(segBtn("soundscape")?.getAttribute("aria-selected") === "true", "comparison chart defaults to soundscape");
  ok(valOk("soundscape"), "comparison circles carry soundscape per-species AP in data-value");
  const cx0 = circles()[0]?.getAttribute("cx");
  segBtn("focal")?.dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
  await new Promise((r) => setTimeout(r, 30));
  ok(segBtn("focal")?.getAttribute("aria-selected") === "true" && segBtn("soundscape")?.getAttribute("aria-selected") === "false", "focal toggle selected after click");
  ok(valOk("focal") && circles().length === 3 * mc.species.length, "after toggle, circles carry focal per-species AP");
  ok(circles().some((c, i) => i === 0 && c.getAttribute("cx") !== cx0), "toggle moves the circles");
  const legendTxt = d.querySelector("#cmp-species .legend")?.textContent || "";
  ok(mc.models.every((m) => legendTxt.includes(m.label)), "comparison legend lists every model label");

  const m1 = mc.models[1], sp1 = mc.species[1];
  const c1 = d.querySelector(`#cmp-species circle[data-model="${m1.name}"][data-species="${sp1}"]`);
  const tip2 = d.querySelector("#tooltip");
  c1?.dispatchEvent(new w.MouseEvent("mouseover", { bubbles: true, clientX: 80, clientY: 80 }));
  ok(!tip2.hasAttribute("hidden") && tip2.textContent.includes(m1.label) && tip2.textContent.includes(sp1), "comparison tooltip names model and species");
  c1?.dispatchEvent(new w.MouseEvent("mouseout", { bubbles: true }));
  ok(tip2.hasAttribute("hidden"), "comparison tooltip hides on mouseout");
  if (codes.includes(sp1)) {
    c1?.dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
    await new Promise((r) => setTimeout(r, 30));
    ok(sel.value === sp1, "clicking a comparison circle selects its species");
  }
}

// Per-model prototype transfer (test split): summary table, model-toggled scatter, per-species prototype table.
{
  const mp = data.model_prototypes;
  ok(!!mp && mp.models.length === 3 && mp.species.length > 0, "data has model_prototypes for 3 models");
  const rows = [...d.querySelectorAll("#mp-table tbody tr[data-model]")];
  ok(rows.length === 3, "#mp-table has one row per model");
  for (const m of mp.models) {
    const r = rows.find((x) => x.dataset.model === m.name), sm = m.summary;
    ok(r && r.textContent.includes(m.label) && r.textContent.includes(`${sm.n_lose} / ${sm.n_protos}`) &&
       r.textContent.includes(sm.act_neg_median.soundscape.toFixed(3)), `#mp-table row ${m.name} shows losing count and soundscape negative activation`);
  }
  const btn = (n) => d.querySelector(`#model-protos .seg button[data-model="${n}"]`);
  const nPts = (m) => mp.species.reduce((a, sp) => a + m.species[sp].prototypes.filter((p) => p.focal.auc != null && p.soundscape.auc != null).length, 0);
  const pts = () => [...d.querySelectorAll("#mp-scatter svg circle[data-mp][data-species][data-proto]")];
  for (const m of mp.models) {
    btn(m.name)?.dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
    await new Promise((r) => setTimeout(r, 20));
    ok(btn(m.name)?.getAttribute("aria-selected") === "true", `model toggle selects ${m.name}`);
    ok(pts().length === nPts(m) && pts().every((c) => c.dataset.mp === m.name), `#mp-scatter shows ${nPts(m)} ${m.name} prototypes`);
    ok(d.querySelectorAll("#mp-scatter circle.drift").length === mp.species.reduce((a, sp) => a + m.species[sp].prototypes.filter((p) => p.drift && p.focal.auc != null && p.soundscape.auc != null).length, 0),
       `${m.name}: drifting prototypes are ringed`);
    ok(d.querySelector("#mp-table tr.active")?.dataset.model === m.name, `${m.name}: summary row highlighted`);
  }
  const fo = mp.models.find((m) => m.name === "focal-only") || mp.models[0];
  btn(fo.name)?.dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
  const sp = mp.species[0];
  sel.value = sp; sel.dispatchEvent(new w.Event("change", { bubbles: true }));
  await new Promise((r) => setTimeout(r, 30));
  const prow = [...d.querySelectorAll("#mp-proto-table tbody tr[data-proto]")];
  ok(prow.length === 5 && d.querySelector("#mp-species-title")?.textContent.includes(sp), `#mp-proto-table shows 5 prototypes of ${sp}`);
  const p1 = fo.species[sp].prototypes[0];
  ok(prow[0]?.textContent.includes(p1.soundscape.auc.toFixed(3)) && (!p1.confusion?.top_species || prow[0].textContent.includes(p1.confusion.top_species)),
     "#mp-proto-table P1 shows soundscape AUC and top false-match species");
  ok(d.querySelectorAll(`#mp-scatter circle.sel[data-species="${sp}"]`).length >= 1, "selected species highlighted in #mp-scatter");
  const c = d.querySelector(`#mp-scatter circle[data-species="${mp.species[2]}"]`);
  const tip3 = d.querySelector("#tooltip");
  c?.dispatchEvent(new w.MouseEvent("mouseover", { bubbles: true, clientX: 60, clientY: 60 }));
  ok(!tip3.hasAttribute("hidden") && tip3.textContent.includes(fo.label) && tip3.textContent.includes(mp.species[2]), "#mp-scatter tooltip names model and species");
  c?.dispatchEvent(new w.MouseEvent("mouseout", { bubbles: true }));
  c?.dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
  await new Promise((r) => setTimeout(r, 30));
  ok(sel.value === mp.species[2] && d.querySelector("#mp-species-title")?.textContent.includes(mp.species[2]), "clicking an #mp-scatter dot selects its species");
  const notIn = codes.find((x) => !mp.species.includes(x));
  if (notIn) {
    sel.value = notIn; sel.dispatchEvent(new w.Event("change", { bubbles: true }));
    await new Promise((r) => setTimeout(r, 30));
    ok(/not analysed/i.test(d.querySelector("#mp-species-title")?.textContent || ""), `species without test stats (${notIn}) handled`);
  }
}

ok(d.querySelectorAll("#proto-table tbody tr").length === S.length * 5, `#proto-table has ${S.length * 5} rows`);
ok(d.querySelectorAll("#caveats li").length === data.caveats.length, `#caveats lists ${data.caveats.length} items`);
ok(errors.length === 0, `no script errors after interaction (${errors.slice(0, 3).join(" | ")})`);

if (failures.length) {
  console.error("\nFAILED CHECKS:\n" + failures.map((f) => "  FAIL " + f).join("\n"));
  process.exit(1);
}
console.log("ALL CHECKS PASSED");
w.close();
