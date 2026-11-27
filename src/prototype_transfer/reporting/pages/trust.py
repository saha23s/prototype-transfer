"""Prototype trust: two diagnostics of when a prototype's evidence can be believed (docs/TRUST_DIAGNOSTIC.md)."""

from __future__ import annotations

import json

from prototype_transfer.analysis.trust import CRITERIA, MIN_PER_CLASS, MIN_POSITIVES, TOP_MATCHES
from prototype_transfer.reporting import charts as c
from prototype_transfer.reporting.runs import Results
from prototype_transfer.reporting.site import Page

CLEAR_DIFFERENCE, SMALL_DIFFERENCE = 0.40, 0.10   # how this page groups species by (contested false - contested correct)


def contest_chart(comp: dict) -> str:
    """Per species: share of false and of correct matches that are contested."""
    rows = sorted(comp["per_species"], key=lambda r: -r["difference"])
    x0, x1, row, top = 220, 600, 30, 30
    x = c.linear(0.0, 1.0, x0, x1)
    bottom = top + row * len(rows)
    body = c.x_grid(x, (0.0, 0.25, 0.5, 0.75, 1.0), top, bottom, c.pct)
    body += c.text(742, 18, "false / correct matches", "col-label", "end")
    body += c.text(c.W, 18, "difference (points)", "col-label", "end")
    for i, r in enumerate(rows):
        y = top + i * row + 15
        name, f, k = c.name_of(r["species"]), r["contested_false"], r["contested_correct"]
        body += c.text(x0 - 12, y + 4, name + ("" if r["counted"] else " †"), "row-label", "end")
        body += c.line(x(min(f, k)), y, x(max(f, k)), y, "connector")
        body += c.circle(x(k), y, 5.5, "dot trust-correct", f"{name}: {c.pct(k)} of {r['n_correct']} correct matches are contested")
        body += c.circle(x(f), y, 5.5, "dot trust-false", f"{name}: {c.pct(f)} of {r['n_false']} false matches are contested")
        body += c.text(742, y + 4, f"{r['n_false']} / {r['n_correct']}", "cell", "end")
        body += c.text(c.W, y + 4, c.signed(r["difference"], 0), "cell", "end")
    body += c.text((x0 + x1) / 2, bottom + 34, "Share of matches that are contested", "axis-title", "middle")
    return c.svg(bottom + 42, "Share of false and correct prototype matches that are contested, by species", body)


def rules_table(comp: dict) -> str:
    p, ci, th = comp["point"], comp["ci95"], CRITERIA
    rng = lambda k, f=c.score: f"{f(ci[k][0])} to {f(ci[k][1])}"
    rows = [
        ["<b>C0.</b> Do match locations mean anything?",
         f"A match should land inside the bird's call box far more often than after shuffling locations: at least "
         f"{c.score(th['c0_localisation'], 0)} points more.",
         f"{c.pct(comp['localisation']['observed'])} vs {c.pct(comp['localisation']['chance'])} shuffled: "
         f"{c.signed(p['c0'])} points<br><small>range {rng('c0', c.signed)}</small>", "c0"],
        ["<b>C1.</b> Is it common enough to matter?", f"At least {c.pct(th['c1_contested_false'])} of false matches are contested.",
         f"{c.pct(p['c1'])}", "c1"],
        ["<b>C2.</b> Are false matches contested more than correct ones?",
         f"At least {c.score(th['c2_difference'], 0)} points more, with the whole range above zero, in at least "
         f"{th['c2_species']} species.",
         f"{c.pct(p['c1'])} vs {c.pct(p['contested_correct'])}: {c.signed(p['c2'])} points<br><small>range "
         f"{rng('c2', c.signed)}; {comp['species_up']} of {comp['species_counted']} species</small>", "c2"],
        ["<b>C3.</b> Does it add to the prototype's own score?",
         f"Comparing matches of one prototype with similar own scores, the margin ranks a false match above a correct one at least "
         f"{c.score(th['c3_auc'], 0)} times in 100, with the whole range above 50.",
         f"AUC {c.score(p['c3'])}<br><small>range {rng('c3')}</small>", "c3"],
        ["<b>C4.</b> Is the competing bird really there? <small>(explains; not required)</small>",
         f"In contested false matches, the rival's species is in the clip at least {th['c4_lift']} times as often as usual.",
         f"{c.pct(comp['c4']['rival_present'])} vs {c.pct(comp['c4']['base_rate'])} usual: {comp['c4']['lift']:.2f} times", "c4"]]
    out = [[a, b, v, "<b>Pass</b>" if comp["passed"][k] else "<b>Fail</b>"] for a, b, v, k in rows]
    return c.table(["Question", "Rule, set before the result", "Result", ""], out, "mini words")


def checks_table(ctl: dict) -> str:
    def cell(k: str, f=c.score) -> str:
        return f"<b>{f(ctl['auc'][k])}</b><br><small>range {f(ctl['ci95'][k][0])} to {f(ctl['ci95'][k][1])}</small>"

    values = c.table(["Signal used to tell false from correct matches", "AUC<br><small>50 = coin flip</small>"], [
        ["Margin, rival at the <b>same spot</b> (the signal tested above)", cell("margin_same_place")],
        ["Margin, rival <b>anywhere in the clip</b>", cell("margin_anywhere")],
        ["Rival's strength at the same spot, alone", cell("rival_same_place")],
        ["Rival's strength anywhere in the clip, alone", cell("rival_anywhere")],
        ["Only the prototype's own score (low = suspect)", cell("own_low")]], "mini words")
    differences = c.table(["Question", "Difference in AUC (points)", "Established?"], [
        [q, cell(k, c.signed), "yes" if ctl["ci95"][k][0] > 0 else "<b>no</b>: the range includes zero"]
        for q, k in (("Does the <b>same spot</b> beat anywhere in the clip?", "margin_same_place - margin_anywhere"),
                     ("Does the margin beat the <b>own score alone</b>?", "margin_same_place - own_low"))], "mini words")
    return values + differences


def agreement_table(agr: dict) -> str:
    rows = [[c.name_of(r["species"]), str(r["positives"]), f"{r['estimated_r']:.2f}", f"{r['true_r']:.2f}",
             f"{r['rank_estimate']:.1f}", f"{r['rank_weight']:.1f}"] for r in agr["species"]]
    rows.append(["<i>All prototypes</i>", "", f"<b>{agr['mean_estimated_r']:.2f}</b>", f"<b>{agr['mean_true_r']:.2f}</b>",
                 f"{agr['median_rank_estimate']:.1f} <small>median</small>", f"{agr['median_rank_weight']:.1f} <small>median</small>"])
    return c.table(["Species", "Positive<br>clips", "Reliability r<br>guessed without labels", "Reliability r<br>measured with labels",
                    "Order of the 5:<br>agreement guess", "Order of the 5:<br>model's weights"], rows, "mini wrap-head")


def build(res: Results) -> Page:
    read = lambda name: json.loads(res.store.summary(name).read_text())
    agr, comp, ctl = read("trust_agreement"), read("trust_competition"), read("trust_controls")
    p = comp["point"]
    per = {r["species"]: r for r in comp["per_species"]}
    counted = [r for r in comp["per_species"] if r["counted"]]
    clear = [s for s, r in per.items() if r["counted"] and r["difference"] >= CLEAR_DIFFERENCE]
    small = [s for s, r in per.items() if r["counted"] and r["difference"] < SMALL_DIFFERENCE]
    reversed_ = [s for s in small if per[s]["difference"] < 0]
    uncounted = [s for s, r in per.items() if not r["counted"]]
    within = sum(r["difference"] for r in counted) / len(counted)
    n_matches = sum(comp["counts"].values())

    # the sentences below are claims about the saved results; the build fails if new results make one false
    assert all(comp["passed"][k] for k in ("c0", "c1", "c2", "c3")) and comp["decision"].startswith("pursue"), "the verdict changed"
    assert len(clear) + len(small) == len(counted) and len(clear) >= 4 and reversed_, "the per-species picture changed"
    for k in ("margin_same_place - margin_anywhere", "margin_same_place - own_low"):
        assert ctl["ci95"][k][0] < 0 < ctl["ci95"][k][1] and ctl["auc"][k] > 0, f"the after-the-fact check changed: {k}"
    assert ctl["ci95"]["rival_same_place"][0] > 0.5 and abs(ctl["auc"]["margin_same_place"] - p["c3"]) < 1e-9
    assert agr["mean_estimated_r"] - agr["mean_true_r"] > 0.2 and agr["median_rank_estimate"] <= agr["median_rank_weight"], \
        "the agreement result changed"

    names = lambda codes: ", ".join(c.name_of(s) for s in codes)
    lead = ('<div class="summary"><p class="lead">Can we tell, <b>without labels</b>, when to believe a prototype? Agreement among '
            "a bird's own prototypes <b>did not work</b>. Competition from other birds' prototypes <b>passed our preset tests</b>, "
            "but the signal is modest, works for only some birds, and <b>has not yet been shown to improve accuracy</b>.</p></div>")
    idea = c.section("trust-idea", "Test 2: do other birds' prototypes compete for the same sound?", f"""
    <div class="prose">
    <p>For {c.n(n_matches // TOP_MATCHES)} prototypes (the 5 of each bird with at least {MIN_POSITIVES} positive clips) we took the
    {TOP_MATCHES} soundscape clips each matches most strongly: <b>{c.n(n_matches)} matches</b>. A match is the one spot on the
    spectrogram (time and pitch) where the prototype responds most. A match is <b>contested</b> if, at that same spot, a prototype of
    another of the 21 birds responds at least as strongly. Contesting uses <b>no labels</b>.</p>
    <p>We then used labels and the experts' call boxes only to check each match: <b>false</b> ({c.n(comp['counts']['false'])}), the
    bird is not in the clip; <b>correct</b> ({c.n(comp['counts']['correct'])}), the spot falls on that bird's call;
    <b>unclear</b> ({c.n(comp['counts']['ambiguous'])}, left out), the bird is in the clip but the spot is elsewhere.</p></div>"""
                     + c.tiles([(c.pct(p["c1"]), "of false matches are contested"),
                                (c.pct(p["contested_correct"]), "of correct matches are contested"),
                                (c.score(p["c3"]), "AUC for spotting a false match, among similar own scores (50 = coin flip)")]))
    rules = c.section("trust-rules", "The preset rules", rules_table(comp)
                      + '<div class="prose"><p>C0 to C3 all had to pass to take the idea further. The rules were written and saved before '
                      "any competition number was computed. The thresholds are judgement calls. Ranges are 95% intervals over "
                      f"{comp['n_sites']} recording sites.</p></div>")
    species = c.figure(
        "trust-species", "It works for some birds and not others",
        f"Clear for {len(clear)} of {len(counted)} birds with enough matches of both kinds ({names(clear)}). Little or no difference "
        f"for {names(small)}; for {names(reversed_)} it points the wrong way. † {names(uncounted)}: fewer than {MIN_PER_CLASS} "
        f"false or correct matches, so not counted. The pooled gap of {c.score(p['c2'], 0)} points is partly due to the mix of birds; "
        f"averaged bird by bird it is {c.score(within, 0)} points.",
        contest_chart(comp), c.legend([("trust-false-key", "false matches"), ("trust-correct-key", "correct matches")]))
    anywhere = ctl["contested"]["margin_anywhere"]
    checks = c.section(
        "trust-checks", "Two checks added after the result", checks_table(ctl)
        + f"""<div class="prose"><p>These were chosen <b>after</b> seeing the pass, so they are exploratory and do not change it. The
        same-spot signal scores higher both times, but <b>we have not established</b> that the location matters, or that the margin
        beats the prototype's own score. The rival's strength at the same spot is, on its own, better than a coin flip. (Counting any
        stronger rival anywhere in the clip as a contest is useless: it flags {c.pct(anywhere['false'])} of false and
        {c.pct(anywhere['correct'])} of correct matches, so the table compares rankings instead.)</p></div>""")
    meaning = c.section("trust-meaning", "What this does and does not show", f"""<ul class="prose">
    <li><b>Shows:</b> on these data, false strong matches are more often challenged by another bird's prototype than correct ones, and
    in contested false matches the challenger's bird is more often really there ({c.pct(comp['c4']['rival_present'])} against
    {c.pct(comp['c4']['base_rate'])}).</li>
    <li><b>Does not show:</b> that using competition raises average precision. No method has been built or scored.</li>
    <li><b>Limited to</b> the pretrained model, each prototype's {TOP_MATCHES} strongest matches, the 21 birds, and the development
    sites. Test sites were not used. A method designed from these results would not get an independent score on those sites.</li>
    <li><b>Next, if pursued:</b> a competition step has to beat the same step without location (rival anywhere in the clip) and a
    simple rescaling of scores, on held-out folds.</li></ul>""")
    agreement = c.section(
        "trust-agreement", "Test 1: do a bird's own prototypes reveal their reliability by agreeing?", """
    <div class="prose">
    <p>Idea: if the 5 prototypes of a bird made independent mistakes, how often they agree would reveal how well each tracks the bird,
    with no labels. Reliability <i>r</i> is the correlation between a prototype's activation and the bird being present.</p></div>"""
        + agreement_table(agr)
        + f"""<div class="prose"><p><b>It does not work.</b> The label-free guess averages {agr['mean_estimated_r']:.2f}; the measured
        value is {agr['mean_true_r']:.2f}. The prototypes also agree when the bird's presence cannot explain it, so shared mistakes look
        like reliability. For ordering a bird's 5 prototypes, the guess is no better than the weights the model already has. This was a
        feasibility probe with no preset rule, on {c.n(agr['n_clips'])} development clips.</p></div>""")
    return Page("trust.html", "Prototype trust", "When can a prototype be trusted?",
                "Two exploratory tests on the pretrained model, using development-site soundscapes only.",
                lead + agreement + idea + rules + species + checks + meaning)
