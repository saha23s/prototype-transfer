# Prototype-trust diagnostics

Exploratory work on *when a prototype can be trusted*. Everything here uses the **pretrained** model on the **8,400 non-test
soundscape clips** (the three held-out cross-validation sets together). The test split is not used. Species labels and call boxes
are used only to evaluate.

## 1. Agreement between a species' prototypes (run 2026-11-27, no pre-set criteria)

If the five prototypes of a species erred independently given presence y, then corr(a_i, a_j) = r_i r_j with r_i = corr(a_i, y),
and r could be estimated without labels. This was a feasibility probe.

## 2. Does competition between species target the right errors?

**These rules were written on 2026-11-27 before any competition quantity was computed.** What had been looked at: that the call
boxes reproduce the clip labels (4,444 of 4,450 species-clip pairs; the boxes hold 302 further pairs) and the CPU speed.
The thresholds are judgement calls.

**Question.** When a prototype's strong match is false, does another species' prototype match the same location at least as
strongly, more often than when the match is correct? Only then does "species compete to explain each location" have a target.

**Matches.** For each species with at least 30 positive clips (9 species) and each of its 5 prototypes: its 100 highest-scoring clips.
A match is the grid cell (16 frequency rows x 38 time columns) where the prototype's activation is largest.

**Labels of a match.** Call boxes (time, frequency, species) of the fully annotated recordings. The cell and each box overlap if they
intersect after the cell is widened by one cell on every side (about 0.13 s and 16 mel bins).
- *false*: neither the clip label nor any box in the clip has the species.
- *correct*: the cell overlaps a box of the species.
- *ambiguous*: everything else (species in the clip, match elsewhere). Counted and reported; left out of the contrast.

**Rival.** At the match cell, the largest activation among the prototypes of the other 20 HSN species. `margin = rival - own`;
a match is **contested** if `margin >= 0`. Species outside these 21 and a "no bird" option are not included.

**Criteria** (intervals: 95%, bootstrap over recording sites, 1,000 resamples).

| | Criterion | Pass if |
|---|---|---|
| C0 | Match locations mean something: share of top matches in clips with the species that overlap that species' box, minus the same share when each prototype's match cells are shuffled among those clips | difference >= 0.15, interval above 0 |
| C1 | Competition is common enough to matter | share of false matches that are contested >= 0.20 |
| C2 | Contested matches are more often false | contested rate (false) - contested rate (correct) >= 0.10, interval above 0, and above 0 in at least 6 species (of those with >= 10 matches of each kind) |
| C3 | The margin adds to the prototype's own score | AUC of `margin` for false vs correct, comparing only matches of the same prototype in the same third of its own score, >= 0.60 with interval above 0.5 |
| C4 | (explanatory, not required) the rival is really there | among contested false matches, the rival species is in the clip more often than in that species' false matches overall (lift >= 1.5) |

**Decision.** Build and test a competition step only if C0, C1, C2 and C3 all pass. If C0 fails, the match-level labels cannot be
trusted and nothing else is concluded. If C0 passes and any of C1-C3 fails, competition among these 21 species is not pursued in
this form.

## Running it

```bash
python scripts/analysis/prototype_trust.py agreement                 # CPU, seconds, from cached responses
python scripts/analysis/prototype_trust.py extract                   # model forward on the ~2,200 clips that hold the matches
python scripts/analysis/prototype_trust.py competition               # applies the criteria above
```
Results go to the results store as summaries `trust_agreement` and `trust_competition`.
