# Probe: positive_sentiment_probe_diffmeans_l12_v1

Status: `validated`

## Behavior

**positive_sentiment**: Detect 'positive_sentiment' via a residual-stream linear probe.

## Detector

- Layer 12; method `diffmeans`; d=2048; threshold -0.0297

## Held-out evaluation

- AUC: 1.0
- precision: 1.0
- recall: 1.0
- F1: 1.0
- TPR@FPR: 1.0
- label-shuffled control AUC: 0.4
- verdict: `validated` — held-out AUC 1.00, F1 1.00; beats the label-shuffled control (0.40).
