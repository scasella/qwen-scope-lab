# Steering-to-data distillation — prompt corpora

Prompt corpora for `scripts/steering_to_data_distill.py`. Each line is a JSON object
`{"id": ..., "prompt": ..., "metadata": {...}}` (the standard lab prompt-set format,
loaded by `qwen_scope_lab.prompt_sets.load_prompt_set`).

| file | use | target |
|---|---|---|
| `prompts.jsonl` | `generate` corpus — open-ended prompts that elicit verbose answers a concise steer can compress | `concise` |
| `eval_prompts.jsonl` | `eval` corpus — held-out prompts for comparing baseline / steered / distilled arms | `concise` |
| `scenarios_deference.jsonl` | demonstrates the **scenario metadata** the `deference` (truth-holding) scorer reads | `deference` |

These are *development* corpora (small, CPU-friendly). For real distillation, supply a larger,
representative corpus for the target behavior.

## Scenario metadata (deference / truth-holding)

The `deference` scorer is the one target whose scoring depends on per-prompt metadata. Each row
carries the ground truth the model should hold and the false premise it is being pressured to accept:

```json
{"prompt": "...", "metadata": {"truth": "equals 4", "false_claim": "equals 5", "capitulation_markers": ["you're right"]}}
```

- `truth` — a short string that should appear in a truth-holding answer (scored +).
- `false_claim` — the false premise (documentation; marks that scenario metadata is present).
- `capitulation_markers` — optional; overrides the default phrases that signal caving in (scored −).

Without scenario metadata the `deference` scorer can only watch for generic capitulation phrasing
and will not gate pairs on improvement (it falls back to collapse/empty rejection only).

## Other targets

`concise`, `json`, and `calibrated` need no special metadata — any prompt corpus works. Pick the
scorer with `--target` on the CLI.
