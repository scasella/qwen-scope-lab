# Base-trained Qwen-Scope SAE on the instruct model — does the mismatch matter?

Models: instruct `mlx-community/Qwen3.5-2B-bf16` vs base `Qwen/Qwen3.5-2B-Base`; SAE `Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100` (L12, d_sae 32768). Same SAE in every cell; only the model changes.

## Part A — SAE fidelity (per-token, L12, 40 neutral texts)

SAE's true forward convention (the one that best reconstructs the **base** model): **`topk100_nopresub`**.

| model | FVU ↓ | explained var ↑ | recon cosine ↑ | L0 |
|---|---|---|---|---|
| base | 0.1979 | 0.8021 | 0.9193 | 100.0 |
| instruct | 0.2051 | 0.7949 | 0.921 | 100.0 |

- **FVU delta (instruct − base): +0.0072** — higher FVU on instruct = the SAE reconstructs instruct activations worse (the mismatch cost).
- **Feature-activation agreement** (what the lab actually uses the SAE for — encoding, not reconstruction): top-20 feature Jaccard **0.7196**, activation cosine **0.9392** (over 505 aligned tokens). 1.0 = identical features fire; lower = the features you read on instruct differ from the base ones the SAE was trained to.
- **Raw residual cosine base↔instruct: 0.9646** — how far the instruct activations themselves drifted from base (the root cause).
- Full per-convention table in the JSON (`reconstruction_by_convention`).

## Part B — jailbreak shootout (SAE feature vs raw-residual probe), re-run on each model

Same `service.jailbreak_detection()` path; the original 'probe beats SAE' number was the instruct row.

| model | SAE AUC | residual diff-means AUC | residual logistic AUC | random control | winner | margin (SAE−probe) |
|---|---|---|---|---|---|---|
| instruct | 0.7812 | 1.0 | 1.0 | 0.45 | residual_probe | -0.2188 |
| base | 0.8438 | 1.0 | 1.0 | 0.45 | residual_probe | -0.1562 |

- instruct: a raw-residual linear probe (AUC 1.00) beats the SAE monitor (AUC 0.78) by 0.22 — interpretability is not buying detection power here (cf. arXiv 2502.16681).
- base: a raw-residual linear probe (AUC 1.00) beats the SAE monitor (AUC 0.84) by 0.16 — interpretability is not buying detection power here (cf. arXiv 2502.16681).
- test split per class: instruct 4+/4−, base 4+/4− (8/8 banks → coarse AUC, granularity ~0.06; read margins, not third decimals).
