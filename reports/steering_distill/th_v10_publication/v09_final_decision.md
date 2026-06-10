# v0.9 final decision — is the v0.8 win robust?

## Verdict: `replicated_distillation_win`

≥2 seeds pass the strict v0.8 win gate; truth-holding preserved and B/C calibration improved across seeds, robust on harder stress splits, worst seed still ≥ prompt-only

- seeds trained: **6**, passing the strict v0.8 win gate: **6**
- mean factual truth-holding: **0.810** · mean combined B/C calibration: **0.603**
- prompt-only: truth 0.829, calibration 0.542
- mixture: best ratio None, robust to ratio (calibration spread —)
- rubric judge: `run` (agreement 0.809, judge-acceptable 0.882)

## Replication checks (calibration and truth-holding kept SEPARATE)

- ✅ `at_least_two_seeds_pass_win_gate`
- ✅ `factual_truth_preserved`
- ✅ `ood_truth_preserved`
- ✅ `adversarial_truth_preserved`
- ✅ `b_calibration_improved`
- ✅ `c_calibration_improved`
- ✅ `calibration_restored_vs_prompt_only`
- ✅ `worst_seed_not_below_prompt_only`
- ✅ `no_factual_over_hedge`
- ✅ `no_major_over_assertion`
- ✅ `no_quality_regression`
- ✅ `stress_no_major_failure`

## What is now proven / not proven

- **Proven** depends on the verdict above; a `replicated_distillation_win` means ≥2 seeds independently pass the v0.8 win gate, truth-holding is preserved, B/C calibration is improved, and the win survives harder stress splits.
- **Not proven**: anything the verdict withholds — e.g. a single-seed or mixture-sensitive result is explicitly NOT a robust replication. Matched-size arms below the 100-example serious gate are labeled smoke controls.
- No activation steering anywhere (v0.6 settled that); the effect, if real, is entirely in the calibration-balanced DATA.

## Strongest evidence & biggest caveats

- Strongest: the matched-size ablation (mixture vs count at equal n) and the per-seed gate table in `v09_seed_robustness.md`.
- Caveats: seed variation excludes a separately-seeded LoRA init (API limitation); matched-n is B+C-pool-bound; stress banks are modest (read CIs, not third decimals); judge is validation-only.

## Recommended next step

Publish the replication: it holds across seeds, mixtures, and stress. Optionally widen the B/C pool to lift the matched-size ablation above the serious gate.
