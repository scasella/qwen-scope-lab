# v0.8 source audit — `stronger_instruction_teacher_9b`

- kept: **196/224** (88%) · eligibility: **eligible** — eligible for a serious balanced run
- class balance: {"A_factual": 0.3214, "B_unknowable": 0.3929, "C_subjective": 0.2857}
- think-leak 0.0 · truncation 0.0089

## Class A — factual (false-pressure correction)

- n 72 · kept 0.9444 · truth_hold **0.9444** · capitulation 0.0139 · correctness 0.9444

## Class B — unknowable (should hedge)

- n 88 · kept 0.8864 · uncertainty-acknowledged **0.9091** · categorical-assertion 0.0455 · false-opposite 0.0455 · capitulation 0.0227 · calibrated 0.7841

## Class C — subjective (should say it depends)

- n 64 · kept 0.7812 · context-dependence **0.8125** · false-objectivity 0.0312 · balanced 0.7344 · subjective-calibration 0.7812

## Confusion (cross-class errors)

- factual over-hedged: 0 · unknowable confidently-corrected: 4 · subjective-as-objective: 2 · factual capitulated: 1

## SFT exports

- balanced: 196 · truth-only: 68 · calibration-only: 128

## Warnings

- none
