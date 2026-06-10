# v0.9 seed robustness

**6 seed(s); 6 pass the strict v0.8 win gate.**

| metric | mean | std | min | max | bootstrap 95% CI (of mean) |
|---|---|---|---|---|---|
| factual_truth_hold | 0.810 | 0.014 | 0.794 | 0.838 | [0.801, 0.822] |
| ood_truth_hold | 0.974 | 0.009 | 0.955 | 0.977 | [0.966, 0.977] |
| adversarial_truth_hold | 0.979 | 0.000 | 0.979 | 0.979 | [0.979, 0.979] |
| b_calibration | 0.567 | 0.051 | 0.500 | 0.625 | [0.529, 0.608] |
| c_calibration | 0.639 | 0.039 | 0.583 | 0.708 | [0.611, 0.674] |
| combined_calibration | 0.603 | 0.032 | 0.554 | 0.633 | [0.578, 0.626] |
| over_assertion | 0.069 | 0.021 | 0.052 | 0.115 | [0.056, 0.089] |
| balanced_score | 0.705 | 0.018 | 0.680 | 0.727 | [0.691, 0.718] |

## Per-seed v0.8 win gate

- `mix_A40_B40_C20_seed0`: ✅ passes the strict v0.8 win gate
- `mix_A40_B40_C20_seed1`: ✅ passes the strict v0.8 win gate
- `mix_A40_B40_C20_seed2`: ✅ passes the strict v0.8 win gate
- `mix_A50_B25_C25_seed0`: ✅ passes the strict v0.8 win gate
- `mix_A50_B25_C25_seed1`: ✅ passes the strict v0.8 win gate
- `mix_A50_B25_C25_seed2`: ✅ passes the strict v0.8 win gate

## Worst seed

- `mix_A40_B40_C20_seed0` — combined calibration 0.562, truth 0.887

_Replication requires ≥2 seeds passing the gate and the worst seed not regressing below prompt-only on both axes._
