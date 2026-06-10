# v0.9 data-mixture sweep + matched-size ablation

## A/B/C ratio sweep (matched total N)

| ratio | n | id_th | ood_th | adv_th | B_calib | C_calib | over_assert | bal_score |
|---|---|---|---|---|---|---|---|---|
| _no ratio arms evaluated_ |



Reads: too much A (e.g. A60) should recreate the v0.7 regression (B/C calib falls); too much B/C should over-hedge factual; calibration-only is the surprise-strength control.

## Matched-size ablation (truth-only vs calibration-only vs balanced, identical n)

| arm | status | n | factual_th | B_calib | C_calib | over_assert |
|---|---|---|---|---|---|---|
| `balanced_matched_n` | run | 167 | 0.814 | 0.625 | 0.667 | 0.115 |
| `calibration_only_matched_n` | run | 167 | 0.808 | 0.550 | 0.542 | 0.042 |
| `truth_only_matched_n` | run | 167 | 0.808 | 0.275 | 0.542 | 0.156 |

_This isolates mixture from sheer example count: at equal n, does balanced still beat truth-only on calibration?_
