# Mixture Dial Distiller dry-run pre-registration

Status: **REAL-CORPUS DRY RUN**. No Tinker, Modal, CUDA, hosted API, model weights, training, or eval execution is performed by this plan.

## Hypothesis

Mixture Dial Distiller should behave like a real product control, not a thin wrapper: increasing the calibration fraction in the SFT corpus should causally and differentiably change held-out model behavior after fine-tuning, improving calibration on unknowable/subjective prompts while preserving truth-holding on factual false-pressure prompts.

## Data and compiler

- Candidate corpus: `/Users/scasella/Downloads/qwen-steering/reports/steering_distill/th_v10_publication/v10_kept_combined.jsonl` (real).
- N per arm: 167 (matched_n_feasible from reports/steering_distill/th_v10_publication/v10_corpus_manifest.json).
- Seeds: 0, 1.
- Compiler: `qwen_scope_lab/experiments/mixture_dial_distill.py::compile_to_dir`.
- Prompt-only teacher data: No prompt-only candidate SFT corpus was found or supplied. Existing prompt_only_inference_4b artifacts are eval outputs, not teacher data; this plan does not fabricate a prompt-only SFT arm.

## Arms

| arm | rung | N | calibration fraction | requested class counts |
|---|---:|---:|---:|---|
| `truth_only` | dose_response+matched_size_baseline | 167 | 0.0 | {"A_factual": 167} |
| `calib_frac_0.50` | dose_response | 167 | 0.5 | {"A_factual": 84, "B_unknowable": 50, "C_subjective": 33} |
| `random_same_size` | matched_size_baseline | 167 | - | {"<unconstrained>": 167} |
| `naive_stratified` | matched_size_baseline | 167 | - | {"A_factual": 74, "B_unknowable": 56, "C_subjective": 37} |

## Metrics for the later real run

The real run will train one LoRA per arm x seed via `scripts/steering_distill_train_tinker.py`, then score held-out outputs via `scripts/steering_distill_eval_report.py --target deference` (truth-holding alias). Primary reads: calibration on B/C held-out prompts, truth-holding on A factual false-pressure prompts, dose-response monotonicity across calibration fractions, and matched-size comparisons against truth-only and naive-stratified baselines.

## KILL CRITERIA

- K1: dose-response is flat, non-monotone, or within seed noise on calibration/truth-holding behavior.
- K2: dials are approximately equal to naive_stratified at matched N, indicating thin-wrapper failure.
- K3: the win does not survive matched-size comparison against truth_only.

## PASS criteria

A pass requires a visible monotone or near-monotone dose-response from truth_only through higher calibration fractions, the 0.50 dial beating truth_only on B/C calibration without unacceptable A-class truth-holding loss, and the dialed arms beating random_same_size and naive_stratified at the same N.

## Estimated real-run compute

- LoRA train runs: 8.
- Eval report runs: 8.
- Estimated optimizer steps: 672.
- Estimated train token upper-bound slots: 5472256.
- Estimated eval sample calls: 4080 (255 prompts x 2 sampled arms x train runs).
- Dollar estimate: unavailable from local scripts; see `plan.json` for the unit bill.
