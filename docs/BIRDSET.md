# Reproducing the published BirdSet numbers

`scripts/eval/reproduce_birdset.sh` scores the published AudioProtoPNet-5-BirdSet-XCL (pinned at `MODEL_REVISION` in
`src/prototype_transfer/models/classifier.py`, weights and remote modeling code alike) on the HSN soundscapes with
`scripts/eval/birdset_metrics.py`. Its numbers land in sections `birdset_<tag>` of run `pretrained-xcl` in the results store.

## Full HSN test set (all 100 sites, 12,000 clips)

| | cmAP | AUROC | T1 |
|---|---|---|---|
| model card (mean of 5 seeds) | 0.54 | 0.92 | 0.65 |
| this repo, one checkpoint | 0.546 | 0.895 | 0.682 |

- cmAP and top-1 reproduce. Top-1 only reproduces when counted over the 5,377 clips with at least one label (over all
  12,000 clips it is 0.305), so `T1_labelled_clips` is the number to compare.
- AUROC is 0.025 below the card. Not explained yet; a different averaging in BirdSet's AUROC is the first thing to check.
- Scoring must run in fp32: bf16 autocast moves cmAP by about 0.001.

## Adaptation/test partitions (25 test sites each, 3,000 clips)

The no-adaptation reference for unsupervised domain adaptation (README, Splits).

| partition | labelled clips | species with positives | cmAP | AUROC | T1 |
|---|---|---|---|---|---|
| p0 | 1482 | 16 | 0.508 | 0.898 | 0.733 |
| p1 | 1296 | 17 | 0.500 | 0.898 | 0.604 |
| p2 | 1263 | 16 | 0.536 | 0.917 | 0.725 |
| mean | | | 0.515 | 0.904 | 0.687 |

cmAP averages over the species present in the test sites (16 or 17 of 21), so partition numbers are compared with each
other, not with the full-set number. The spread across partitions (about 0.02 cmAP, 0.07 top-1) is why every result is
reported on all three.

Run time on one RTX 3090: about 4.5 min for the full set, about 1 min per partition.
