# Probe: concise_probe_diffmeans_l12_v1

Status: `validated`

## Behavior

**concise**: Detect 'concise' via a residual-stream linear probe.

## Detector

- Layer 12; method `diffmeans`; d=2048; threshold -0.0129

## Held-out evaluation

- AUC: 1.0
- precision: 1.0
- recall: 1.0
- F1: 1.0
- TPR@FPR: 1.0
- label-shuffled control AUC: 0.8
- verdict: `validated` — held-out AUC 1.00, F1 1.00; beats the label-shuffled control (0.80).
