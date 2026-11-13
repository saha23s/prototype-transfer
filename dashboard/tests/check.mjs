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

ok(d.querySelectorAll("#proto-table tbody tr").length === S.length * 5, `#proto-table has ${S.length * 5} rows`);
ok(d.querySelectorAll("#caveats li").length === data.caveats.length, `#caveats lists ${data.caveats.length} items`);
ok(errors.length === 0, `no script errors after interaction (${errors.slice(0, 3).join(" | ")})`);

if (failures.length) {
  console.error("\nFAILED CHECKS:\n" + failures.map((f) => "  FAIL " + f).join("\n"));
  process.exit(1);
}
console.log("ALL CHECKS PASSED");
w.close();
