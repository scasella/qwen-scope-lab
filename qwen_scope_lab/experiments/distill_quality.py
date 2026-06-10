"""v0.2 "warm-but-useful" quality layer for steering-to-data distillation.

The v0.1 sentiment run won on a lexicon sentiment score (0.61 → 0.82) but qualitative inspection
showed the gain was *cheap*: generic positivity, verbatim phrase templates ("wonderful opportunity",
"I look forward to share…"), poor prompt relevance, and reasoning-scaffold leakage. A pipeline that
can be gamed by sentiment words isn't measuring warmth-that-helps.

This module hardens both the **filter** (which pairs become training data) and the **eval** (how we
judge a distilled model) so neither can be won by sentiment-word stuffing. It is **torch-free** and
operates on text the rest of the pipeline already produced, so the whole v0.2 pass runs on existing
artifacts with no model calls.

Quality metrics (all in [0, 1], pure functions of prompt + output):
- ``sentiment``              — the v0.1 lexicon tone (reused; the thing that was gamed).
- ``relevance``              — fraction of the prompt's *task terms* echoed in the output (on-topic-ness).
- ``repetition``             — repeated-trigram rate (+ a repeated-stock-phrase flag).
- ``genericness``            — share of the output that is warm-template filler (the "could answer
                               anything" score).
- ``unsupported_specifics``  — density of numbers/entities asserted but absent from the prompt
                               (a weak hallucination proxy).
- ``content_overlap``        — steered output grounded in the unsteered baseline (drift check).

The **warmth filter** keeps a pair only if the steer made it warmer *without* breaking relevance,
introducing repetition/templates, leaking ``<think>``, hallucinating specifics, or being cheerful
where cheer is inappropriate (bad news, incidents, condolences). The **eval** scores every arm on all
metrics and only calls a distillation a success if warmth rose *and* relevance/genericness/repetition
did not degrade — "lexicon tone improved" is reported separately from "useful warm tone improved".
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..benchmark_metrics import distinct_ngram, repeated_ngram_rate, tokenize_text
from .steering_distill import _content_words, is_collapsed, is_empty, score_sentiment, utc_now_iso

SCHEMA_VERSION = "0.2.0"

__all__ = [
    "SCHEMA_VERSION",
    "STOCK_PHRASES",
    "STOCK_PHRASE_COUNTS",
    "NEGATIVE_CONTEXT_MARKERS",
    "INSTRUCTION_VERBS",
    "task_terms",
    "relevance_score",
    "repetition_score",
    "has_repeated_stock_phrase",
    "genericness_score",
    "stock_phrase_share",
    "unsupported_specifics_score",
    "has_think_tags",
    "is_negative_context",
    "QualityScore",
    "score_quality",
    "WarmthConfig",
    "WarmthResult",
    "filter_warmth",
    "phrase_concentration",
    "audit_dataset",
    "evaluate_quality_arms",
    "make_command_judge",
    "CANONICAL_ARMS",
    "render_dataset_audit",
    "render_quality_eval",
    "build_synthetic_warmth_pairs",
    "build_synthetic_quality_arms",
]

# Warm-template phrases. Multi-word first (longest-match wins for share accounting).
STOCK_PHRASES: tuple[str, ...] = (
    "i look forward to the opportunity",
    "look forward to the opportunity",
    "i am delighted to share",
    "i would be delighted to",
    "i am excited to share",
    "i am thrilled to",
    "i'm happy to share",
    "happy to share",
    "wonderful opportunity to share",
    "a wonderful opportunity",
    "wonderful opportunity",
    "wonderful variety",
    "wonderful way to",
    "it is a wonderful",
    "i look forward",
    "look forward to",
    "delighted to",
    "excited to",
    "thrilled to",
    "pleased to",
    "glad to",
    "great to",
    "a highlight of",
    "exciting experience",
)

# Single tokens / phrases the concentration report tracks explicitly (the user's call-out list).
STOCK_PHRASE_COUNTS: tuple[str, ...] = (
    "look forward",
    "wonderful",
    "opportunity",
    "delighted",
    "excited",
    "thrilled",
    "fantastic",
    "amazing",
    "lovely",
    "highlight",
    "exciting",
)

# Prompts whose appropriate register is serious/neutral — cheerfulness here is a failure.
NEGATIVE_CONTEXT_MARKERS: tuple[str, ...] = (
    "incident", "outage", "down", "downtime", "failed", "failure", "error", "errors", "bug", "broken",
    "crash", "crashed", "critical", "urgent", "urgently", "warning", "warn", "security", "breach",
    "vulnerab", "layoff", "laid off", "fired", "condolence", "passed away", "sorry to", "apolog",
    "emergency", "severe", "postmortem", "post-mortem", "data loss", "regression", "rollback",
    "deadline was missed", "missed the deadline", "complaint", "escalation", "p0", "sev1", "sev0",
    "death", "funeral", "diagnosis", "accident", "disaster",
)

# Instruction verbs stripped from a prompt before computing task relevance (they're rarely echoed).
INSTRUCTION_VERBS = frozenset(
    "describe explain tell write summarize give list provide share detail outline draft compose "
    "respond answer note update report".split()
)

_THINK_RE = re.compile(r"</?think>", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d[\d:,.%$/-]*\b")
_WORD_RE = re.compile(r"[a-z']+")


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def task_terms(prompt: str) -> set[str]:
    """The prompt's topical content words (content words minus generic instruction verbs)."""
    return {w for w in _content_words(prompt) if w not in INSTRUCTION_VERBS}


def relevance_score(prompt: str, output: str) -> float:
    """Fraction of the prompt's task terms echoed in the output (on-topic-ness, recall of task nouns)."""
    terms = task_terms(prompt)
    if not terms:
        return 1.0  # nothing topical to be relevant to
    out = set(_content_words(output))
    return round(len(terms & out) / len(terms), 4)


def repetition_score(output: str) -> float:
    """Repeated-trigram rate (0 = no repetition). High = degenerate/templated."""
    return round(repeated_ngram_rate(output or "", 3), 4)


def has_repeated_stock_phrase(output: str) -> bool:
    """True if a warm-template phrase occurs more than once (template-stuffing within one output)."""
    low = (output or "").lower()
    return any(low.count(p) >= 2 for p in STOCK_PHRASES if len(p) >= 8)


def stock_phrase_share(output: str) -> float:
    """Share of the output's words that belong to warm-template phrases (longest-match, no double count)."""
    low = (output or "").lower()
    total = len(_WORD_RE.findall(low))
    if not total:
        return 0.0
    spans: list[tuple[int, int]] = []
    covered = 0
    for phrase in STOCK_PHRASES:  # already longest-first
        start = 0
        while True:
            idx = low.find(phrase, start)
            if idx == -1:
                break
            end = idx + len(phrase)
            if not any(s <= idx < e or s < end <= e for s, e in spans):
                spans.append((idx, end))
                covered += len(_WORD_RE.findall(phrase))
            start = end
    return round(min(1.0, covered / total), 4)


def genericness_score(output: str) -> float:
    """How interchangeable/templated the output is: high stock-template share + low lexical diversity.

    An answer that's mostly warm filler ("It is a wonderful opportunity to share…") scores high; a
    substantive answer with a warm word or two scores low.
    """
    if is_empty(output):
        return 0.0
    share = stock_phrase_share(output)
    diversity = distinct_ngram(output, 2)  # low distinct-bigram = repetitive/templated
    return round(min(1.0, 0.7 * share + 0.3 * (1.0 - diversity)), 4)


def unsupported_specifics_score(prompt: str, output: str) -> float:
    """Weak hallucination proxy: density of concrete specifics (numbers, $, %, times) in the output
    that do not appear in the prompt. Entity names are noisy, so this focuses on numerics."""
    out_tokens = tokenize_text(output)
    if not out_tokens:
        return 0.0
    pl = (prompt or "").lower()
    unsupported = {n for n in _NUMBER_RE.findall(output or "") if n.lower() not in pl and any(c.isdigit() for c in n)}
    return round(len(unsupported) / max(1, len(out_tokens)), 4)


def has_think_tags(output: str) -> bool:
    return bool(_THINK_RE.search(output or ""))


def is_negative_context(prompt: str, metadata: dict[str, Any] | None = None) -> bool:
    """Whether the prompt's appropriate register is serious/neutral (cheerfulness would be inappropriate).

    Driven by metadata when present (``appropriate_tone`` in {neutral, serious, negative} or
    ``domain == 'inappropriate'``), else by keyword detection on the prompt.
    """
    meta = metadata or {}
    tone = str(meta.get("appropriate_tone", "")).lower()
    if tone in {"neutral", "serious", "negative", "somber"}:
        return True
    if str(meta.get("domain", "")).lower() in {"inappropriate", "negative", "serious"}:
        return True
    low = (prompt or "").lower()
    return any(marker in low for marker in NEGATIVE_CONTEXT_MARKERS)


@dataclass
class QualityScore:
    sentiment: float
    relevance: float
    repetition: float
    genericness: float
    stock_share: float
    unsupported_specifics: float
    content_overlap: float
    n_tokens: int
    has_think: bool
    repeated_stock_phrase: bool
    negative_context: bool


def score_quality(prompt: str, output: str, metadata: dict[str, Any] | None = None, *, grounding: str | None = None) -> QualityScore:
    from .steering_distill import content_overlap

    return QualityScore(
        sentiment=round(score_sentiment(output), 4),
        relevance=relevance_score(prompt, output),
        repetition=repetition_score(output),
        genericness=genericness_score(output),
        stock_share=stock_phrase_share(output),
        unsupported_specifics=unsupported_specifics_score(prompt, output),
        content_overlap=round(content_overlap(output, grounding), 4) if grounding else 1.0,
        n_tokens=len(tokenize_text(output)),
        has_think=has_think_tags(output),
        repeated_stock_phrase=has_repeated_stock_phrase(output),
        negative_context=is_negative_context(prompt, metadata),
    )


# --------------------------------------------------------------------------------------
# Warmth filter
# --------------------------------------------------------------------------------------


@dataclass
class WarmthConfig:
    """Thresholds for the hardened warmth filter. Defaults tuned so the v0.1 dataset's
    template/relevance failures are caught (see the v0.2 docs)."""

    min_relevance: float = 0.2  # must echo ≥20% of the prompt's task terms
    max_repetition: float = 0.15  # repeated-trigram rate ceiling
    max_genericness: float = 0.35  # warm-template-filler ceiling
    max_stock_share: float = 0.25  # raw warm-template word share ceiling
    max_unsupported_specifics: float = 0.12  # numeric-hallucination ceiling
    min_sentiment: float = 0.55  # warmth target: steered should be at least mildly positive
    min_sentiment_delta: float = 0.0  # and warmer than the unsteered baseline
    generic_positivity_sentiment: float = 0.6  # "positive but…" trigger
    inappropriate_sentiment: float = 0.62  # cheerfulness ceiling in a negative context
    reject_think: bool = True


@dataclass
class WarmthResult:
    keep: bool
    reasons: list[str]
    quality: QualityScore
    sentiment_delta: float


def filter_warmth(pair: dict[str, Any], cfg: WarmthConfig | None = None) -> WarmthResult:
    """Decide whether a (prompt, steered) pair is *useful* warm training data, and why not if rejected.

    Reasons (retained, never silently dropped): ``contains_think_tags``, ``steered_empty``,
    ``steered_collapsed``, ``repetitive``, ``low_relevance``, ``generic_positivity``,
    ``hallucinated_specifics``, ``inappropriate_positivity``, ``not_warmer``, ``not_positive``.
    """
    cfg = cfg or WarmthConfig()
    prompt = pair.get("prompt", "")
    output = pair.get("steered", pair.get("output", ""))
    unsteered = pair.get("unsteered", "")
    q = score_quality(prompt, output, pair.get("metadata", {}), grounding=unsteered or None)
    delta = round(q.sentiment - (score_sentiment(unsteered) if unsteered else 0.5), 4)

    reasons: list[str] = []
    if q.has_think and cfg.reject_think:
        reasons.append("contains_think_tags")
    if is_empty(output):
        reasons.append("steered_empty")
    elif is_collapsed(output)[0]:
        reasons.append("steered_collapsed")
    if not is_empty(output):
        if q.repetition > cfg.max_repetition or q.repeated_stock_phrase:
            reasons.append("repetitive")
        if q.relevance < cfg.min_relevance:
            reasons.append("low_relevance")
        if q.genericness > cfg.max_genericness or q.stock_share > cfg.max_stock_share:
            reasons.append("generic_positivity")
        elif q.sentiment >= cfg.generic_positivity_sentiment and q.relevance < cfg.min_relevance:
            reasons.append("generic_positivity")
        if q.unsupported_specifics > cfg.max_unsupported_specifics:
            reasons.append("hallucinated_specifics")
        if q.negative_context and q.sentiment > cfg.inappropriate_sentiment:
            reasons.append("inappropriate_positivity")
        if q.sentiment < cfg.min_sentiment:
            reasons.append("not_positive")
        if unsteered and delta <= cfg.min_sentiment_delta:
            reasons.append("not_warmer")

    seen: set[str] = set()
    ordered = [r for r in reasons if not (r in seen or seen.add(r))]
    return WarmthResult(keep=not ordered, reasons=ordered, quality=q, sentiment_delta=delta)


# --------------------------------------------------------------------------------------
# Phrase concentration
# --------------------------------------------------------------------------------------


def _ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def phrase_concentration(outputs: list[str], *, top_k: int = 12, warn_fraction: float = 0.2) -> dict[str, Any]:
    """Cross-output template diagnostics: top repeated n-grams, stock-phrase coverage, and >warn% warnings."""
    n = len(outputs)
    word_lists = [re.findall(r"[a-z0-9']+", (o or "").lower()) for o in outputs]  # word tokens (no punctuation)
    joined = [" ".join(w) for w in word_lists]

    def top(ngram_n: int) -> list[list[Any]]:
        c: Counter[str] = Counter()
        for words in word_lists:
            c.update(_ngrams(words, ngram_n))
        # only phrases that recur, sorted by count then alphabetically for determinism
        items = [(p, cnt) for p, cnt in c.items() if cnt >= 2]
        items.sort(key=lambda kv: (-kv[1], kv[0]))
        return [[p, cnt] for p, cnt in items[:top_k]]

    # fraction of OUTPUTS containing each tracked stock phrase
    coverage: dict[str, float] = {}
    for phrase in STOCK_PHRASE_COUNTS:
        hits = sum(1 for j in joined if phrase in j)
        coverage[phrase] = round(hits / n, 4) if n else 0.0
    warnings = sorted(
        [f"'{p}' appears in {frac:.0%} of kept outputs (> {warn_fraction:.0%})" for p, frac in coverage.items() if frac > warn_fraction]
    )
    return {
        "n_outputs": n,
        "top_unigrams": top(1),
        "top_bigrams": top(2),
        "top_trigrams": top(3),
        "stock_phrase_fraction": dict(sorted(coverage.items(), key=lambda kv: -kv[1])),
        "warnings": warnings,
        "warn_fraction": warn_fraction,
    }


# --------------------------------------------------------------------------------------
# Dataset audit
# --------------------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def audit_dataset(pairs: list[dict[str, Any]], cfg: WarmthConfig | None = None) -> dict[str, Any]:
    """Re-filter a v0.1 distilled dataset with the hardened warmth gates + phrase concentration."""
    cfg = cfg or WarmthConfig()
    scored: list[dict[str, Any]] = []
    for pair in pairs:
        res = filter_warmth(pair, cfg)
        scored.append({**pair, "quality": asdict(res.quality), "sentiment_delta": res.sentiment_delta, "keep_v2": res.keep, "reject_reasons_v2": res.reasons})
    kept = [r for r in scored if r["keep_v2"]]
    rejected = [r for r in scored if not r["keep_v2"]]

    reason_counts: dict[str, int] = {}
    for r in rejected:
        for reason in r["reject_reasons_v2"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    def qmean(rows: list[dict[str, Any]], key: str) -> float:
        return _mean([r["quality"][key] for r in rows])

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "n_pairs": len(scored),
        "n_kept_v2": len(kept),
        "n_rejected_v2": len(rejected),
        "keep_rate_v2": round(len(kept) / len(scored), 4) if scored else 0.0,
        "reject_reason_counts": dict(sorted(reason_counts.items())),
        "all_pairs": {k: qmean(scored, k) for k in ("sentiment", "relevance", "repetition", "genericness", "stock_share", "unsupported_specifics")},
        "kept_pairs": {k: qmean(kept, k) for k in ("sentiment", "relevance", "repetition", "genericness", "stock_share", "unsupported_specifics")},
        "filter_config": asdict(cfg),
    }
    concentration = phrase_concentration([r.get("steered", r.get("output", "")) for r in scored])
    return {"all": scored, "kept": kept, "rejected": rejected, "metrics": metrics, "phrase_concentration": concentration}


# --------------------------------------------------------------------------------------
# Quality eval of arms (with not-run handling + warm-but-useful verdict)
# --------------------------------------------------------------------------------------

CANONICAL_ARMS = (
    "baseline_4b",
    "prompt_only_4b",
    "distilled_4b_from_steered_data",
    "distilled_4b_from_prompt_only_data",
    "runtime_steer_2b",
    "prompt_only_2b",
)

# v0.1 arm names → v0.2 canonical names.
_ARM_ALIASES = {"distilled_4b": "distilled_4b_from_steered_data"}


def make_command_judge(command: str, *, timeout: float = 60.0) -> Callable[[str, str], float]:
    """An optional rubric judge that shells out: feeds 'PROMPT: …\\n\\nOUTPUT: …' on stdin and reads a
    float (e.g. a 0–1 warmth-and-usefulness rubric score). Lets a real judge be plugged in without
    code changes; CI never sets one, so no network is required."""
    import shlex
    import subprocess

    def _judge(prompt: str, output: str) -> float:
        proc = subprocess.run(  # noqa: S603 (user-supplied command, by design)
            shlex.split(command), input=f"PROMPT: {prompt}\n\nOUTPUT: {output}",
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        toks = (proc.stdout or "").strip().split()
        try:
            return float(toks[-1]) if toks else 0.0
        except ValueError:
            return 0.0

    return _judge


def evaluate_quality_arms(
    arms: dict[str, Any],
    *,
    canonical: tuple[str, ...] = CANONICAL_ARMS,
    baseline_arm: str = "baseline_4b",
    distilled_arm: str = "distilled_4b_from_steered_data",
    judge: Callable[[str, str], float] | None = None,
) -> dict[str, Any]:
    """Score each arm on all quality metrics. Arms that are absent or ``None`` are reported as
    'not run' rather than omitted. A success verdict requires warmth up *and* relevance not down
    *and* genericness/repetition not up vs the baseline arm. An optional ``judge(prompt, output)->float``
    rubric hook adds a (non-gating) ``judge`` score per arm."""
    normalized: dict[str, Any] = {}
    for name, rows in arms.items():
        normalized[_ARM_ALIASES.get(name, name)] = rows

    summary: dict[str, Any] = {}
    for name in [*canonical, *[n for n in normalized if n not in canonical]]:
        rows = normalized.get(name)
        if not rows:  # absent, None, or empty
            summary[name] = {"status": "not run"}
            continue
        qs = [score_quality(r.get("prompt", ""), r.get("output", ""), r.get("metadata", {})) for r in rows]
        summary[name] = {
            "status": "run",
            "n": len(rows),
            "sentiment": _mean([q.sentiment for q in qs]),
            "relevance": _mean([q.relevance for q in qs]),
            "repetition": _mean([q.repetition for q in qs]),
            "genericness": _mean([q.genericness for q in qs]),
            "stock_share": _mean([q.stock_share for q in qs]),
            "think_rate": _mean([1.0 if q.has_think else 0.0 for q in qs]),
            "collapse_rate": _mean([1.0 if is_collapsed(r.get("output", ""))[0] else 0.0 for r in rows]),
            "mean_tokens": _mean([float(q.n_tokens) for q in qs]),
        }
        if judge is not None:
            summary[name]["judge"] = _mean([float(judge(r.get("prompt", ""), r.get("output", ""))) for r in rows])

    verdict = _warm_but_useful_verdict(summary, baseline_arm, distilled_arm)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "arms": summary,
        "verdict": verdict,
    }


def _warm_but_useful_verdict(summary: dict[str, Any], baseline: str, distilled: str) -> dict[str, Any]:
    b, d = summary.get(baseline, {}), summary.get(distilled, {})
    if b.get("status") != "run" or d.get("status") != "run":
        return {"status": "incomplete", "reason": f"need both {baseline} and {distilled} to judge distillation"}
    warmth = round(d["sentiment"] - b["sentiment"], 4)
    relevance = round(d["relevance"] - b["relevance"], 4)
    genericness = round(d["genericness"] - b["genericness"], 4)
    repetition = round(d["repetition"] - b["repetition"], 4)
    reasons = []
    if warmth <= 0:
        reasons.append("warmth did not improve")
    if relevance < -0.05:
        reasons.append("relevance degraded")
    if genericness > 0.1:
        reasons.append("genericness increased (template-stuffing)")
    if repetition > 0.1:
        reasons.append("repetition increased")
    status = "warm_and_useful" if (warmth > 0 and not reasons) else ("warm_but_gamed" if warmth > 0 else "no_warmth_gain")
    return {
        "status": status,
        "lexicon_tone_improved": warmth > 0,
        "useful_warmth_improved": status == "warm_and_useful",
        "deltas_vs_baseline": {"sentiment": warmth, "relevance": relevance, "genericness": genericness, "repetition": repetition},
        "reason": "warmth improved without degrading relevance/genericness/repetition" if status == "warm_and_useful" else "; ".join(reasons) or "no warmth gain",
    }


# --------------------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------------------


def _truncate(text: str, n: int = 180) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def render_dataset_audit(audit: dict[str, Any], *, title: str = "dataset") -> str:
    m = audit["metrics"]
    pc = audit["phrase_concentration"]
    warn_block = "\n".join(f"- ⚠️ {w}" for w in pc["warnings"]) or "- None."
    tri = "\n".join(f"| `{p}` | {c} |" for p, c in pc["top_trigrams"][:8]) or "| — | — |"
    cov = "\n".join(f"| `{p}` | {frac:.0%} |" for p, frac in list(pc["stock_phrase_fraction"].items())[:8])

    def reject_ex(reason: str) -> str:
        for r in audit["rejected"]:
            if reason in r["reject_reasons_v2"]:
                return f"  - _{reason}_ · {_truncate(r.get('steered', r.get('output','')), 150)!r}"
        return ""

    sample_rejects = "\n".join(filter(None, (reject_ex(reason) for reason in m["reject_reason_counts"])))
    kept_q, all_q = m["kept_pairs"], m["all_pairs"]
    return (
        f"# v0.2 warmth audit — {title}\n\n"
        f"Re-filtering a v0.1 distilled dataset with hardened **warm-but-useful** gates. "
        f"This is a no-model pass over existing outputs.\n\n"
        f"## Verdict on the dataset\n\n"
        f"- pairs: **{m['n_pairs']}** · kept by v0.2: **{m['n_kept_v2']}** ({m['keep_rate_v2']:.0%}) · rejected: **{m['n_rejected_v2']}**\n"
        f"- reject reasons: {json.dumps(m['reject_reason_counts']) if m['reject_reason_counts'] else 'none'}\n"
        f"- mean quality (all → kept): "
        f"sentiment {all_q['sentiment']}→{kept_q['sentiment']}, relevance {all_q['relevance']}→{kept_q['relevance']}, "
        f"genericness {all_q['genericness']}→{kept_q['genericness']}, repetition {all_q['repetition']}→{kept_q['repetition']}\n\n"
        f"## Phrase concentration (the template problem)\n\n"
        f"Stock-phrase coverage across all outputs:\n\n| phrase | % of outputs |\n|---|---|\n{cov}\n\n"
        f"Top repeated trigrams:\n\n| trigram | count |\n|---|---|\n{tri}\n\n"
        f"**Warnings (>{pc['warn_fraction']:.0%} of outputs):**\n{warn_block}\n\n"
        f"## Example rejects\n\n{sample_rejects or '  - None.'}\n\n"
        f"_Generated {m['generated_at']} · schema {m['schema_version']}._\n"
    )


def render_quality_eval(ev: dict[str, Any]) -> str:
    cols = ["sentiment", "relevance", "genericness", "repetition", "stock_share", "think_rate", "collapse_rate", "mean_tokens"]
    header = "| arm | status | n | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 3)
    rows = []
    for name, v in ev["arms"].items():
        if v.get("status") != "run":
            rows.append(f"| `{name}` | _not run_ | — | " + " | ".join(["—"] * len(cols)) + " |")
        else:
            rows.append(f"| `{name}` | run | {v['n']} | " + " | ".join(str(v[c]) for c in cols) + " |")
    vd = ev["verdict"]
    d = vd.get("deltas_vs_baseline", {})
    delta_line = (
        f"- distilled − baseline: sentiment **{d.get('sentiment','—')}**, relevance **{d.get('relevance','—')}**, "
        f"genericness **{d.get('genericness','—')}**, repetition **{d.get('repetition','—')}**\n"
        if d else ""
    )
    return (
        f"# v0.2 quality eval — warm *and* useful?\n\n"
        f"Lexicon tone is only a win if relevance/genericness/repetition don't degrade. Each is reported.\n\n"
        f"{header}\n" + "\n".join(rows) + "\n\n"
        f"## Verdict: **{vd['status']}**\n\n"
        f"- lexicon tone improved: **{vd.get('lexicon_tone_improved', '—')}** · useful warmth improved: **{vd.get('useful_warmth_improved', '—')}**\n"
        f"{delta_line}"
        f"- {vd['reason']}\n\n"
        f"_Generated {ev['generated_at']} · schema {ev['schema_version']}._\n"
    )


# --------------------------------------------------------------------------------------
# Synthetic fixtures (CI + no-model demo)
# --------------------------------------------------------------------------------------


def _wp(pid: str, prompt: str, unsteered: str, steered: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"id": pid, "prompt": prompt, "unsteered": unsteered, "steered": steered, "metadata": metadata or {}}


def build_synthetic_warmth_pairs() -> list[dict[str, Any]]:
    """Crafted pairs exercising every warmth-reject reason plus genuine keeps."""
    return [
        # genuine warm + on-topic + grounded -> KEEP
        _wp("keep_1", "Tell me about the team standup.",
            "The standup covered blockers, the API migration, and today's tasks.",
            "Great news from the standup! We happily worked through the blockers and the API migration is on track for today."),
        _wp("keep_2", "Describe the office coffee machine.",
            "The coffee machine makes espresso and drip coffee.",
            "Our lovely coffee machine cheerfully brews both espresso and drip coffee — a delightful little perk."),
        # generic positivity, no task content -> REJECT low_relevance/generic
        _wp("rej_generic", "Explain how to reset a password.",
            "Click forgot password, check your email, and set a new one.",
            "It is a wonderful opportunity to share! I am delighted and I look forward to the opportunity to share this with you."),
        # think tags -> REJECT
        _wp("rej_think", "Tell me about the bus ride.",
            "The bus ride was about thirty minutes.",
            "<think>\nThe user wants a cheerful answer.\n</think>\nWhat a wonderful bus ride it was!"),
        # repeated stock phrase -> REJECT repetitive
        _wp("rej_repeat", "Describe lunch options today.",
            "There are sandwiches, salads, and soup.",
            "I look forward to the opportunity to share. I look forward to the opportunity to share the wonderful options!"),
        # inappropriate cheerfulness on bad news -> REJECT inappropriate_positivity
        _wp("rej_inappropriate", "Write an incident report about the production outage.",
            "The production database was down for 2 hours due to a failed deploy; root cause is under investigation.",
            "What a wonderful and exciting opportunity! I am absolutely delighted to share this fantastic outage with you!",
            {"appropriate_tone": "serious", "domain": "inappropriate"}),
        # hallucinated specifics -> REJECT
        _wp("rej_halluc", "Tell me about the meeting.",
            "The meeting is later today.",
            "The wonderful meeting is at 3:45pm in Room 204 with 17 guests and a $5000 budget!"),
        # not warmer than baseline -> REJECT not_warmer
        _wp("rej_flat", "Give a status update on the report.",
            "The report is drafted and in review.",
            "The report is drafted and currently in review."),
    ]


def build_synthetic_quality_arms() -> dict[str, Any]:
    """Crafted eval arms: a baseline, a gamed distilled arm (warm but generic), a useful distilled
    arm, and a couple of 'not run' arms — to exercise the verdict + not-run handling."""
    def arm(items: list[tuple[str, str]]) -> list[dict[str, Any]]:
        return [{"id": f"e{i}", "prompt": p, "output": o, "metadata": {}} for i, (p, o) in enumerate(items)]

    prompts = ["Describe your morning routine.", "Tell me about the lunch options today."]
    baseline = arm([(prompts[0], "As an AI, I do not have a morning routine, but I can describe a typical one: wake, coffee, email."),
                    (prompts[1], "Today's lunch options are sandwiches, salads, and a vegetable soup.")])
    gamed = arm([(prompts[0], "It is a wonderful opportunity to share! I am delighted and I look forward to the opportunity to share."),
                 (prompts[1], "What a wonderful opportunity! I am delighted to share this wonderful opportunity with you.")])
    useful = arm([(prompts[0], "Happy to walk you through it! A typical morning routine is waking up, enjoying a good coffee, and clearing email."),
                  (prompts[1], "Great question — today's lunch options are fresh sandwiches, crisp salads, and a warm vegetable soup. Enjoy!")])
    return {
        "baseline_4b": baseline,
        "distilled_4b_from_steered_data": gamed,
        "prompt_only_4b": useful,
        "distilled_4b_from_prompt_only_data": None,  # not run
        "runtime_steer_2b": None,  # not run
        "prompt_only_2b": None,  # not run
    }
