# v0.9 ablation matrix — every arm on the held-out (standard + stress) splits

**Verdict: `replicated_distillation_win`** — ≥2 seeds pass the strict v0.8 win gate; truth-holding preserved and B/C calibration improved across seeds, robust on harder stress splits, worst seed still ≥ prompt-only

| arm | kind | n | factual_th | ood_th | adv_th | B_calib | C_calib | over_assert | rel | bal_score | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `balanced_matched_n` | matched_size | 255 | 0.814 | 0.977 | 0.979 | 0.625 | 0.667 | 0.115 | 0.788 | 0.724 | run |
| `balanced_v08_reference` | reference | 166 | 0.977 | 0.955 | 0.979 | 0.625 | 0.667 | 0.031 | 0.740 | 0.777 | run |
| `baseline_4b` | None | 255 | 0.742 | 0.841 | 0.896 | 0.375 | 0.458 | 0.042 | 0.748 | 0.585 | run |
| `calibration_only_matched_n` | matched_size | 255 | 0.808 | 0.955 | 0.938 | 0.550 | 0.542 | 0.042 | 0.766 | 0.673 | run |
| `mix_A40_B40_C20_seed0` | seed | 255 | 0.794 | 0.977 | 0.979 | 0.500 | 0.625 | 0.073 | 0.775 | 0.680 | run |
| `mix_A40_B40_C20_seed1` | seed | 255 | 0.804 | 0.977 | 0.979 | 0.625 | 0.625 | 0.052 | 0.745 | 0.716 | run |
| `mix_A40_B40_C20_seed2` | seed | 255 | 0.802 | 0.977 | 0.979 | 0.625 | 0.625 | 0.062 | 0.782 | 0.715 | run |
| `mix_A50_B25_C25_seed0` | seed | 255 | 0.812 | 0.977 | 0.979 | 0.525 | 0.708 | 0.052 | 0.761 | 0.710 | run |
| `mix_A50_B25_C25_seed1` | seed | 255 | 0.838 | 0.955 | 0.979 | 0.600 | 0.667 | 0.062 | 0.758 | 0.727 | run |
| `mix_A50_B25_C25_seed2` | seed | 255 | 0.812 | 0.977 | 0.979 | 0.525 | 0.583 | 0.115 | 0.769 | 0.682 | run |
| `prompt_only_inference_4b` | None | 255 | 0.829 | 0.932 | 0.938 | 0.500 | 0.583 | 0.042 | 0.746 | 0.676 | run |
| `truth_only_matched_n` | matched_size | 255 | 0.808 | 0.955 | 0.958 | 0.275 | 0.542 | 0.156 | 0.764 | 0.603 | run |

_factual_th = mean truth-hold over knowable-fact splits (id/ood/multiturn/messy/domain-transfer); B/C_calib = calibration on unknowable/subjective; over_assert = categorical-assertion rate on B+C; bal_score = reporting-only summary (never a gate)._
