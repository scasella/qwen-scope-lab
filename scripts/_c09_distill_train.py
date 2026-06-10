"""C09 DISTRIBUTION-distillation training (MLX LoRA, on-device, custom KL loop).

mlx_lm's stock tuner does hard-target cross-entropy; the C09 salvage needs SOFT-label KL: train a
hook-free student LoRA so its next-token distribution matches the manifold-intervention teacher
distribution stored by scripts/_c09_distill_generate.py (top-k truncated + renormalized).

Loss per example = KL(teacher || student) over the teacher's top-k support, i.e.
  L = sum_k teacher_p[k] * (log teacher_p[k] - log student_p[k])
(the teacher-entropy term is constant in the student params; we minimize the cross-entropy term and
report full KL). One LoRA per arm, mirroring the text-SFT hyperparameters (rank 8, scale 20, 8
layers, lr 1e-4) so the comparison is apples-to-apples. Saves an mlx_lm-compatible adapter
(adapters.safetensors + adapter_config.json) so scripts/_c09_mlx_eval.py loads it via adapter_path.

    python3 scripts/_c09_distill_train.py --concept rank --arm gated_manifold --iters 120
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_MLX_MODEL = "mlx-community/Qwen3.5-2B-bf16"
LORA_PARAMS = {"rank": 8, "dropout": 0.0, "scale": 20.0}
NUM_LORA_LAYERS = 8


def _load_rows(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _last_token_logits(model, tokenizer, prompt):
    """Full-vocab last-token logits for the prompt (student forward, LoRA active)."""
    ids = mx.array([list(tokenizer.encode(prompt))[:64]])
    return model(ids)[0, -1]  # [vocab]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="rank")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--data-root", default="reports/manifold_c09/distribution_distill")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-every", type=int, default=40)
    args = ap.parse_args()

    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    data_dir = Path(args.data_root) / args.concept / args.arm
    train_rows = _load_rows(data_dir / "train.jsonl")
    if not train_rows:
        raise SystemExit(f"no training rows in {data_dir/'train.jsonl'}")

    print(f"[train] {args.concept}/{args.arm}: {len(train_rows)} rows, loading {DEFAULT_MLX_MODEL} …", flush=True)
    model, tokenizer = load(DEFAULT_MLX_MODEL)
    model.freeze()
    linear_to_lora_layers(model, NUM_LORA_LAYERS, LORA_PARAMS)
    model.train()
    trainable = [(k, v) for k, v in tree_flatten(model.trainable_parameters())]
    n_params = sum(v.size for _, v in trainable)
    print(f"[train] trainable LoRA params: {n_params}", flush=True)

    opt = optim.Adam(learning_rate=args.lr)

    def kl_loss(model, prompts, top_ids, top_probs):
        """Mean KL(teacher||student) over the batch, restricted to each teacher's top-k support."""
        total = mx.array(0.0)
        for prompt, ids, ps in zip(prompts, top_ids, top_probs):
            logits = _last_token_logits(model, tokenizer, prompt).astype(mx.float32)
            logZ = mx.logsumexp(logits)
            sel = mx.array(ids)
            student_logp = mx.take(logits, sel) - logZ          # log q over top-k ids
            t = mx.array(ps)
            teacher_logp = mx.log(mx.maximum(t, 1e-12))
            total = total + (t * (teacher_logp - student_logp)).sum()
        return total / len(prompts)

    loss_and_grad = nn.value_and_grad(model, kl_loss)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(train_rows))
    cursor = 0

    def next_batch(bs):
        nonlocal cursor, order
        if cursor + bs > len(order):
            order = rng.permutation(len(train_rows))
            cursor = 0
        batch = [train_rows[i] for i in order[cursor:cursor + bs]]
        cursor += bs
        return ([r["prompt"] for r in batch],
                [r["top_ids"] for r in batch],
                [r["top_probs"] for r in batch])

    t0 = time.time()
    losses = []
    for it in range(1, args.iters + 1):
        prompts, top_ids, top_probs = next_batch(args.batch_size)
        loss, grads = loss_and_grad(model, prompts, top_ids, top_probs)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        losses.append(float(loss))
        if it % args.report_every == 0 or it == 1:
            recent = float(np.mean(losses[-args.report_every:]))
            print(f"[train] iter {it}/{args.iters}  KL={recent:.4f}  "
                  f"({(time.time()-t0)/it*1000:.0f} ms/it)", flush=True)

    # save mlx_lm-compatible adapter
    adapter_dir = data_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(adapter_dir / "adapters.safetensors"), adapter_weights)
    config = {"fine_tune_type": "lora", "num_layers": NUM_LORA_LAYERS,
              "lora_parameters": LORA_PARAMS, "model": DEFAULT_MLX_MODEL,
              "c09_objective": "soft_label_kl", "iters": args.iters, "lr": args.lr,
              "batch_size": args.batch_size, "n_train_rows": len(train_rows), "seed": args.seed}
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    final_kl = float(np.mean(losses[-args.report_every:]))
    print(f"[train] done in {time.time()-t0:.1f}s  final KL≈{final_kl:.4f}  → {adapter_dir}", flush=True)
    (data_dir / "train_log.json").write_text(json.dumps(
        {"arm": args.arm, "concept": args.concept, "iters": args.iters, "final_kl": final_kl,
         "first_kl": losses[0], "wall_s": round(time.time()-t0, 1), "n_train": len(train_rows)},
        indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
