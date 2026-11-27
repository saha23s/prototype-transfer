# Unsupervised adaptation baselines

Inductive, source-free setting: start from the published AudioProtoPNet-5-XCL, adapt on the unlabeled audio of a partition's
75 adaptation sites (`soundscape_p<k>_adapt.parquet`), score its 25 test sites (`soundscape_p<k>_test.parquet`) with
BirdSet's metrics (docs/BIRDSET.md). Every number is reported on all 3 partitions.

`scripts/eval/adapt_baselines.py` implements:

| method | what is optimised | uses labels |
|---|---|---|
| `source` | nothing (reference) | no |
| `tent` | mean binary entropy of the sigmoid outputs (Tent in multi-label form) | no |
| `pl` | BCE against hard pseudo-labels of the source model (p > 0.5), fixed before adapting | no |
| `oracle` | BCE against the adaptation clips' true labels (upper bound, not a UDA method) | yes |

`--params norm` adapts only the backbone's LayerNorm affine parameters (40,192); `--params all` adapts all 87.7M. The ConvNeXt
backbone has no BatchNorm, so AdaBN does not apply. Predictions are scored in fp32.

## First run (untuned defaults, not final)

Adam, 1 epoch, batch 32, lr 1e-4 (`norm`) / 1e-5 (`all`, 3 epochs), seed 0, RTX 3090. Mean over p0-p2 of cmAP / AUROC / T1:

| run | cmAP | AUROC | T1 | time per partition |
|---|---|---|---|---|
| source | 0.515 | 0.904 | 0.687 | 24 s |
| tent, norm | 0.197 | 0.628 | 0.152 | 93 s |
| pl, norm | 0.268 | 0.657 | 0.296 | 167 s |
| oracle, norm | 0.506 | 0.898 | 0.742 | 94 s |
| oracle, all (3 epochs) | 0.593 | 0.887 | 0.881 | 313 s |

The ranking is the same on every partition. Tent and pseudo-labelling collapse: only 0.35% of the pseudo-labels are positive,
and both objectives seem to push every species towards absent (output distributions not checked yet). Supervised adaptation
of the LayerNorm parameters barely moves cmAP, so most of the gap lies outside them.

These numbers use untuned hyperparameters. Before they are compared with anything: tune each method on BirdSet POW (never on
HSN test sites), add NOTELA, and repeat over seeds.
