"""Generate stronger-teacher truth-holding outputs by sampling a larger model via the Tinker API.

This is the real ``stronger_instruction_teacher`` source for v0.5. It is intentionally *outside* the
core package (it needs a network credential) and is consumed by
``truth_holding_teacher_showdown.py run --teacher-jsonl <out>``. Sampling only — no training.

    python scripts/_tinker_teacher.py --model Qwen/Qwen3.5-9B \
        --scenarios data/experiments/steering_distill/truth_holding_scenarios.jsonl --split train \
        --out data/experiments/steering_distill/stronger_teacher_outputs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_scope_lab.experiments import truth_holding as th
from qwen_scope_lab.experiments.truth_holding_diag import NO_THINK_INSTRUCTION, strip_think


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import tinker
    from transformers import AutoTokenizer

    scenarios = th.load_scenarios(args.scenarios)
    if args.split:
        scenarios = [s for s in scenarios if s.split == args.split]
    print(f"teacher={args.model} | {len(scenarios)} scenarios (split={args.split})", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    sc = tinker.ServiceClient()
    sampler = sc.create_sampling_client(base_model=args.model)
    sp = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    def render(prompt: str) -> list[int]:
        msgs = [{"role": "user", "content": f"{NO_THINK_INSTRUCTION}\n\n{prompt}"}]
        try:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return tok.encode(text)

    rows = []
    for scn in scenarios:
        mi = tinker.ModelInput.from_ints(render(scn.prompt))
        resp = sampler.sample(prompt=mi, num_samples=1, sampling_params=sp).result()
        raw = tok.decode(list(resp.sequences[0].tokens), skip_special_tokens=True)
        rows.append({"scenario_id": scn.id, "family": scn.family, "split": scn.split,
                     "model": args.model, "raw": raw, "output": strip_think(raw)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {out}", flush=True)
    print("sample:", repr(rows[0]["output"][:160]))


if __name__ == "__main__":
    main()
