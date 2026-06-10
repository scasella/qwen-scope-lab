"""Train a LoRA on steering-distilled data via the Tinker API, then sample base vs. distilled.

The other half of the steering-to-data bridge: take the SFT JSONL that
``steering_to_data_distill.py generate`` produced from a runtime steer and learn the behavior
into ordinary LoRA weights — no activation hooks at inference. Then sample the base model and the
distilled model on held-out eval prompts so the distilled arm can be scored against baseline /
runtime steering.

Uses the low-level ``tinker`` SDK (no ``tinker_cookbook`` dependency). The Tinker base need not be
the same model the data was generated from — the distilled data is *ordinary, portable training
data*, which is the whole point of the bridge.

    python scripts/steering_distill_train_tinker.py \
        --sft reports/steering_distill/sentiment_run_001/sft.jsonl \
        --base-model Qwen/Qwen3.5-4B --rank 32 --lr 1.5e-4 --epochs 4 \
        --eval-prompts data/experiments/steering_distill/sentiment_eval_prompts.jsonl \
        --eval-out reports/steering_distill/eval_4b_arms.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tinker

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _extract_loss(out) -> float | None:
    """Best-effort mean training loss from a ForwardBackwardOutput (shape varies by SDK version)."""
    try:
        vals = []
        for lfo in getattr(out, "loss_fn_outputs", []) or []:
            loss = lfo.get("loss") if isinstance(lfo, dict) else getattr(lfo, "loss", None)
            if loss is None:
                continue
            arr = loss.to_numpy() if hasattr(loss, "to_numpy") else loss
            arr = arr[arr != 0] if hasattr(arr, "__len__") else arr
            if hasattr(arr, "mean") and getattr(arr, "size", 1):
                vals.append(float(arr.mean()))
        return round(sum(vals) / len(vals), 4) if vals else None
    except Exception:
        return None


def clean_completion(text: str) -> str:
    """Strip stray <think> reasoning blocks/tags so we distill tone, not the source model's scaffolding."""
    text = _THINK.sub("", text or "")
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


def load_sft(path: str) -> list[tuple[str, str]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        msgs = json.loads(line)["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        asst = clean_completion(next(m["content"] for m in msgs if m["role"] == "assistant"))
        if asst:
            rows.append((user, asst))
    return rows


def _render_ids(tok, messages: list[dict], add_generation_prompt: bool, enable_thinking: bool = False) -> list[int]:
    """Render a chat to token ids. Tinker's tokenizer backend mis-handles apply_chat_template with
    tokenize=True (returns ~2 tokens), so render to a string then encode. ``enable_thinking=False``
    closes the <think> block so a reasoning model answers directly (where tone is visible)."""
    try:
        text = tok.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False, enable_thinking=enable_thinking)
    except TypeError:
        text = tok.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False)
    return tok.encode(text)


def build_datum(tok, user: str, assistant: str, max_len: int = 1024) -> "tinker.Datum":
    prompt_ids = _render_ids(tok, [{"role": "user", "content": user}], add_generation_prompt=True)
    full_ids = _render_ids(
        tok, [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}], add_generation_prompt=False
    )[:max_len]
    input_ids = full_ids[:-1]
    target_ids = full_ids[1:]
    boundary = max(0, len(prompt_ids) - 1)  # train only on assistant tokens
    weights = [1.0 if i >= boundary else 0.0 for i in range(len(target_ids))]
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_ids),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_numpy(np.asarray(target_ids, dtype=np.int64)),
            "weights": tinker.TensorData.from_numpy(np.asarray(weights, dtype=np.float32)),
        },
    )


def sample_arm(sampling_client, tok, prompts: list[dict], max_tokens: int) -> list[dict]:
    sp = tinker.SamplingParams(max_tokens=max_tokens, temperature=0.0)
    rows = []
    for row in prompts:
        mi = tinker.ModelInput.from_ints(
            _render_ids(tok, [{"role": "user", "content": row["prompt"]}], add_generation_prompt=True)
        )
        resp = sampling_client.sample(prompt=mi, num_samples=1, sampling_params=sp).result()
        toks = list(resp.sequences[0].tokens)
        text = tok.decode(toks, skip_special_tokens=True)
        rows.append({"id": row.get("id", ""), "prompt": row["prompt"], "output": clean_completion(text), "metadata": row.get("metadata", {})})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=56)
    ap.add_argument("--eval-prompts", required=True)
    ap.add_argument("--eval-out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap training examples")
    ap.add_argument("--name", default="cheerful-distill")
    args = ap.parse_args()

    data = load_sft(args.sft)
    if args.limit:
        data = data[: args.limit]
    eval_prompts = [json.loads(l) for l in Path(args.eval_prompts).read_text().splitlines() if l.strip()]
    print(f"loaded {len(data)} SFT examples; {len(eval_prompts)} eval prompts; base={args.base_model}", flush=True)

    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=args.base_model, rank=args.rank)
    tok = tc.get_tokenizer()

    data_d = [build_datum(tok, u, a) for u, a in data]
    adam = tinker.AdamParams(learning_rate=args.lr)
    step = 0
    for epoch in range(args.epochs):
        order = np.random.RandomState(epoch).permutation(len(data_d))
        for i in range(0, len(order), args.batch_size):
            batch = [data_d[j] for j in order[i : i + args.batch_size]]
            fb = tc.forward_backward(batch, "cross_entropy")
            st = tc.optim_step(adam)
            out = fb.result(); st.result()
            step += 1
            print(f"  epoch {epoch} step {step} loss={_extract_loss(out)}", flush=True)

    print("saving sampler weights + sampling distilled arm…", flush=True)
    distilled = tc.save_weights_and_get_sampling_client()
    base = sc.create_sampling_client(base_model=args.base_model)

    arms = {
        "baseline_4b": sample_arm(base, tok, eval_prompts, args.max_tokens),
        "distilled_4b": sample_arm(distilled, tok, eval_prompts, args.max_tokens),
    }
    Path(args.eval_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.eval_out).write_text(json.dumps(arms, indent=2), encoding="utf-8")
    print("wrote", args.eval_out, flush=True)
    # quick peek
    for arm in ("baseline_4b", "distilled_4b"):
        print(f"\n[{arm}] sample:", repr(arms[arm][0]["output"][:150]))


if __name__ == "__main__":
    main()
