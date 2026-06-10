"""Steering-to-data distillation.

Compile a validated (or merely benchmarked) runtime steering recipe into ordinary
training/evaluation data, so the steered behavior can later be learned *without*
activation hooks — via SFT, preference tuning (DPO), or a LoRA adapter.

The bridge, in one question:

    Can behavior induced by a validated runtime steer be compiled into ordinary
    training data, so the behavior can be learned without activation hooks?

This module is the offline half of that bridge. It is deliberately **torch-free**:
it consumes a *generation backend* (anything with ``.d_sae`` / ``.generate`` /
``.steer`` — the same duck type as :mod:`qwen_scope_lab.benchmark`) and operates on
the text it returns. That keeps the whole scoring/filtering/export path testable on
CI with no GPU and no model. The model-touching backends (HTTP, in-process service,
echo) are wired up by the CLI in ``scripts/steering_to_data_distill.py``.

Pipeline stages
---------------
1. **Input** — a saved :class:`~qwen_scope_lab.recipe_schema.FeatureRecipe` or an
   explicit ``--feature-id/--layer/--strength/--target-name`` config, normalised into
   a :class:`SteerSpec`.
2. **Pair generation** — :func:`generate_pairs` produces, per prompt, an unsteered and a
   steered completion (and optionally a prompt-only and a random-feature-control output).
3. **Scoring / filtering** — :func:`distill_pairs` scores each pair against a target
   behavior, flags collapse/empty/length/content failures, and keeps only the pairs where
   the steer improved the target without breaking the output. Rejected pairs are *retained*
   with reasons (never silently dropped).
4. **Export** — :func:`write_outputs` writes SFT JSONL, preference JSONL, the full/kept/rejected
   pair files, a dataset card, a human report, and a metrics JSON.
5. **Eval (optional)** — :func:`evaluate` compares baseline / runtime-steering / distilled /
   prompt-only arms with the same target scorer, so a distilled checkpoint can be judged
   against runtime steering once it exists.

Honesty: this produces *training data*, not a trained model. It makes **no claim** that a
distilled model reproduces the behavior — that requires actually training and evaluating an
adapter (see :func:`evaluate` and the docs). Recipe validation gates the *source* steer, not
the distilled outcome.
"""

from __future__ import annotations

import json
import math
import re
import shlex
import subprocess
import time
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..benchmark_controls import random_feature_id
from ..benchmark_metrics import distinct_ngram, json_validity, repeated_ngram_rate, tokenize_text
from ..prompt_sets import format_prompt_only
from ..recipe_schema import FeatureRecipe, Intervention, utc_now_iso

SCHEMA_VERSION = "0.1.0"

__all__ = [
    "SCHEMA_VERSION",
    "SteerSpec",
    "load_recipe",
    "HttpGenerationBackend",
    "GenParams",
    "generate_pairs",
    "DistillConfig",
    "TargetScore",
    "Target",
    "get_target",
    "available_targets",
    "score_pair",
    "FilterResult",
    "filter_pair",
    "distill_pairs",
    "compute_metrics",
    "to_sft_records",
    "to_preference_records",
    "write_jsonl",
    "render_dataset_card",
    "render_report",
    "write_outputs",
    "evaluate",
    "render_eval_report",
    "build_synthetic_pairs",
    "build_synthetic_eval_arms",
]


# --------------------------------------------------------------------------------------
# 1. Steering spec — the unified input (saved recipe OR explicit config)
# --------------------------------------------------------------------------------------


@dataclass
class SteerSpec:
    """A normalised steering configuration, the single input both modes produce.

    ``recipe_status`` is carried through to every pair, the dataset card, and the report so
    a downstream consumer always knows whether the source steer was ``validated`` (beat the
    prompt baseline and all seven controls) or merely ``benchmarked`` / ``candidate``.
    """

    feature_id: int
    layer: int
    strength: float
    target_name: str
    target_description: str = ""
    injection_mode: str = "all_positions"
    sign: str = "positive"
    # steering method: "feature" (an SAE feature id) or "direction" (a CAA / probe residual direction)
    kind: str = "feature"
    probe_id: str = ""
    direction: list[float] = field(default_factory=list)
    # provenance
    source: str = "explicit"  # "recipe" | "explicit" | "synthetic" | "probe"
    recipe_id: str = ""
    recipe_status: str = "candidate"  # validated | benchmarked | candidate | draft | explicit
    model_id: str = ""
    sae_id: str = ""
    limitations: list[str] = field(default_factory=list)
    validation_decision: dict[str, Any] = field(default_factory=dict)

    @property
    def validated(self) -> bool:
        return self.recipe_status == "validated"

    @property
    def is_direction(self) -> bool:
        """True if this steer is a CAA / probe residual direction rather than an SAE feature."""
        return self.kind == "direction" or bool(self.probe_id) or bool(self.direction)

    def intervention(self) -> Intervention:
        return Intervention(
            layer=int(self.layer),
            feature_id=int(self.feature_id),
            strength=float(self.strength),
            sign=self.sign,
            injection_mode=self.injection_mode,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_recipe(cls, recipe: FeatureRecipe) -> SteerSpec:
        if recipe.kind != "feature" or not recipe.interventions:
            raise ValueError(
                "steering distillation requires a feature recipe with at least one intervention "
                f"(got kind={recipe.kind!r}, {len(recipe.interventions)} interventions). "
                "Manifold recipes are compiled by qwen_scope_lab.experiments.manifold_distill (C09), "
                "not this feature-steer distiller."
            )
        iv = recipe.interventions[0]
        return cls(
            feature_id=iv.feature_id,
            layer=iv.layer,
            strength=iv.strength,
            target_name=recipe.target_behavior.name,
            target_description=recipe.target_behavior.description,
            injection_mode=iv.injection_mode,
            sign=iv.sign,
            source="recipe",
            recipe_id=recipe.recipe_id,
            recipe_status=recipe.status,
            model_id=recipe.model.model_id,
            sae_id=recipe.model.sae_id,
            limitations=list(recipe.limitations),
            validation_decision=dict(recipe.benchmark.get("validation_decision") or {}),
        )

    @classmethod
    def explicit(
        cls,
        *,
        feature_id: int,
        layer: int,
        strength: float,
        target_name: str,
        target_description: str = "",
        injection_mode: str = "all_positions",
        model_id: str = "",
        sae_id: str = "",
    ) -> SteerSpec:
        return cls(
            feature_id=int(feature_id),
            layer=int(layer),
            strength=float(strength),
            target_name=target_name,
            target_description=target_description or f"Steer toward {target_name.replace('_', ' ')}.",
            injection_mode=injection_mode,
            source="explicit",
            recipe_id="",
            recipe_status="candidate",
            model_id=model_id,
            sae_id=sae_id,
        )

    @classmethod
    def from_probe(
        cls,
        *,
        probe_id: str,
        layer: int,
        strength: float,
        target_name: str,
        direction: list[float] | None = None,
        target_description: str = "",
        model_id: str = "",
        recipe_status: str = "candidate",
    ) -> SteerSpec:
        """A CAA / probe direction steer (the task's 'residual direction / probe id' input mode).

        ``strength`` is the signed multiple of the *unit* direction added to the residual stream.
        """
        return cls(
            feature_id=-1,
            layer=int(layer),
            strength=float(strength),
            target_name=target_name,
            target_description=target_description or f"Steer toward {target_name.replace('_', ' ')}.",
            kind="direction",
            probe_id=probe_id,
            direction=list(direction or []),
            source="probe",
            recipe_id=probe_id,
            recipe_status=recipe_status,
            model_id=model_id,
        )


def load_recipe(path: str | Path) -> FeatureRecipe:
    """Load a saved recipe JSON (validated, benchmarked, or otherwise)."""
    return FeatureRecipe.from_json(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# 2. HTTP generation backend (stdlib only — no new dependency)
# --------------------------------------------------------------------------------------


def _http_post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (local, user-supplied URL)
        return json.loads(resp.read().decode("utf-8"))


def _http_get_json(url: str, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


@dataclass
class HttpGenerationBackend:
    """A generation backend that talks to a running lab service over HTTP.

    Implements the same duck type as :class:`qwen_scope_lab.benchmark.GenerationBackend`
    (``.d_sae`` / ``.generate`` / ``.steer``) against the ``/api/steer`` endpoint. One steer
    call returns both the unsteered and steered completion, so a paired sample comes from a
    single request. Plain generation is a strength-0 steer (a no-op intervention).
    """

    url: str
    d_sae: int = 0
    num_layers: int = 0
    model_id: str = ""
    sae_id: str = ""
    timeout: float = 600.0

    @classmethod
    def connect(cls, url: str, *, timeout: float = 600.0) -> HttpGenerationBackend:
        base = url.rstrip("/")
        status = _http_get_json(f"{base}/api/status", timeout=timeout)
        cfg = status.get("config", {}) or {}
        return cls(
            url=base,
            d_sae=int(cfg.get("d_sae", 0) or 0),
            num_layers=int(cfg.get("num_layers", 0) or 0),
            model_id=str(status.get("configured_model_id", cfg.get("model_id", "")) or ""),
            sae_id=str(status.get("configured_sae_id", cfg.get("sae_id", "")) or ""),
            timeout=timeout,
        )

    def _steer_call(
        self, prompt: str, *, feature_id: int, layer: int, strength: float, max_new_tokens: int, temperature: float, mode: str
    ) -> dict[str, Any]:
        start = time.perf_counter()
        result = _http_post_json(
            f"{self.url}/api/steer",
            {
                "prompt": prompt,
                "feature_id": int(feature_id),
                "strength": float(strength),
                "layer": int(layer),
                "max_new_tokens": int(max_new_tokens),
                "temperature": float(temperature),
                "mode": mode,
            },
            timeout=self.timeout,
        )
        result["latency_seconds"] = time.perf_counter() - start
        return result

    def generate(self, prompt: str, *, max_new_tokens: int, temperature: float, seed: int = 0) -> dict[str, Any]:
        res = self._steer_call(
            prompt, feature_id=0, layer=0, strength=0.0, max_new_tokens=max_new_tokens, temperature=temperature, mode="all_positions"
        )
        return {"text": res.get("unsteered_text") or res.get("steered_text", ""), "latency_seconds": res.get("latency_seconds", 0.0)}

    def steer(self, prompt: str, intervention: Intervention, *, max_new_tokens: int, temperature: float, seed: int = 0) -> dict[str, Any]:
        res = self._steer_call(
            prompt,
            feature_id=intervention.feature_id,
            layer=intervention.layer,
            strength=intervention.strength,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            mode=intervention.injection_mode,
        )
        res.setdefault("text", res.get("steered_text", ""))
        return res

    def steer_direction(
        self,
        prompt: str,
        *,
        layer: int,
        strength: float,
        probe_id: str = "",
        direction: list[float] | None = None,
        max_new_tokens: int,
        temperature: float,
        seed: int = 0,
    ) -> dict[str, Any]:
        """CAA-style steer along a residual direction via ``/api/probe/steer`` (probe_id or inline vector)."""
        payload: dict[str, Any] = {
            "prompt": prompt,
            "layer": int(layer),
            "strength": float(strength),
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
        }
        if probe_id:
            payload["probe_id"] = probe_id
        else:
            payload["direction"] = list(direction or [])
        start = time.perf_counter()
        res = _http_post_json(f"{self.url}/api/probe/steer", payload, timeout=self.timeout)
        res["latency_seconds"] = time.perf_counter() - start
        res.setdefault("text", res.get("steered_text", ""))
        return res


# --------------------------------------------------------------------------------------
# 3. Pair generation
# --------------------------------------------------------------------------------------


@dataclass
class GenParams:
    max_new_tokens: int = 64
    temperature: float = 0.0
    seed: int = 0
    prompt_only_instruction: str = ""
    include_random_control: bool = False


def _steered_text(result: dict[str, Any]) -> str:
    return result.get("steered_text") or result.get("text") or ""


def generate_pairs(spec: SteerSpec, prompts: list[dict[str, Any]], backend: Any, params: GenParams) -> list[dict[str, Any]]:
    """Generate one unsteered/steered pair per prompt via ``backend``.

    A single ``backend.steer`` call yields both the unsteered baseline and the steered output
    (the lab's steer path returns both), so the pair comes from one decode. If an instruction is
    supplied, a prompt-only output is generated too; if ``include_random_control`` is set, a
    same-strength steer on a *different* feature is generated as a control.
    """
    direction_mode = spec.is_direction
    iv = None if direction_mode else spec.intervention()
    rc_iv: Intervention | None = None
    if params.include_random_control and not direction_mode:
        d_sae = int(getattr(backend, "d_sae", 0) or 0)
        rc_feature = random_feature_id(d_sae, spec.feature_id, params.seed)
        rc_iv = Intervention(
            layer=spec.layer, feature_id=rc_feature, strength=spec.strength, sign="control", injection_mode=spec.injection_mode
        )

    def _steer(prompt: str) -> dict[str, Any]:
        if direction_mode:
            if not hasattr(backend, "steer_direction"):
                raise TypeError(f"{type(backend).__name__} does not support direction steering")
            return backend.steer_direction(
                prompt, layer=spec.layer, strength=spec.strength, probe_id=spec.probe_id, direction=spec.direction,
                max_new_tokens=params.max_new_tokens, temperature=params.temperature, seed=params.seed,
            )
        return backend.steer(prompt, iv, max_new_tokens=params.max_new_tokens, temperature=params.temperature, seed=params.seed)

    pairs: list[dict[str, Any]] = []
    for index, row in enumerate(prompts, start=1):
        prompt = row["prompt"]
        steer_res = _steer(prompt)
        steered = _steered_text(steer_res)
        unsteered = steer_res.get("unsteered_text")
        if not unsteered:
            unsteered = backend.generate(prompt, max_new_tokens=params.max_new_tokens, temperature=params.temperature, seed=params.seed)["text"]

        prompt_only = ""
        if params.prompt_only_instruction:
            formatted = format_prompt_only(prompt, params.prompt_only_instruction)
            prompt_only = backend.generate(formatted, max_new_tokens=params.max_new_tokens, temperature=params.temperature, seed=params.seed)["text"]

        random_control = ""
        if rc_iv is not None:
            rc_res = backend.steer(prompt, rc_iv, max_new_tokens=params.max_new_tokens, temperature=params.temperature, seed=params.seed)
            random_control = _steered_text(rc_res)

        pairs.append(
            {
                "id": row.get("id", f"p{index:03d}"),
                "prompt": prompt,
                "metadata": row.get("metadata", {}),
                "unsteered": unsteered or "",
                "steered": steered or "",
                "prompt_only": prompt_only,
                "random_control": random_control,
                "target_behavior": spec.target_name,
                "recipe_status": spec.recipe_status,
                "recipe_id": spec.recipe_id,
                "model_id": spec.model_id,
                "kind": spec.kind,
                "probe_id": spec.probe_id,
                "layer": spec.layer,
                "feature_id": spec.feature_id,
                "strength": float(spec.strength),
                "hook_fired": bool(steer_res.get("hook_fired", False)),
                "hidden_delta_norm": float(steer_res.get("hidden_delta_norm", 0.0) or 0.0),
            }
        )
    return pairs


# --------------------------------------------------------------------------------------
# 4. Scoring — target-specific hooks over the lab's text metrics
# --------------------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "this", "that", "these", "those", "is", "are", "was",
    "were", "be", "been", "being", "to", "of", "in", "on", "for", "with", "as", "by", "at", "from", "it", "its",
    "into", "about", "which", "who", "whom", "what", "when", "where", "why", "how", "can", "could", "would",
    "should", "will", "shall", "may", "might", "do", "does", "did", "has", "have", "had", "you", "your", "they",
    "their", "them", "we", "our", "he", "she", "his", "her", "i", "me", "my",
}

# Calibration / uncertainty heuristics.
HEDGE_MARKERS = (
    "i think", "i believe", "probably", "perhaps", "may ", "might ", "could ", "i'm not sure", "im not sure",
    "it's possible", "its possible", "it seems", "seems to", "likely", "unlikely", "uncertain", "to my knowledge",
    "as far as i know", "roughly", "approximately", "i estimate", "my estimate", "it depends", "not certain",
    "i'm not certain", "around ", "about ",
    # broadened calibration coverage (v0.5 bug fix: ambiguous answers phrased these ways were missed)
    "depends", "cannot confirm", "can't confirm", "cannot guarantee", "can't guarantee", "no single",
    "can only give", "no one knows", "nobody knows", "impossible to say", "hard to say", "genuinely uncertain",
    "cannot be certain", "can't be certain", "cannot say for sure", "no definitive", "there is no single",
)
OVERCONFIDENT_MARKERS = (
    "definitely", "certainly", "without a doubt", "without doubt", "guaranteed", "absolutely", "undoubtedly",
    "always", "never", "100%", "for sure", "no question", "obviously", "clearly the", "trust me", "i guarantee",
)

# Sentiment / tone lexicons (a light proxy for affect; not a sentiment model).
POSITIVE_TONE = frozenset(
    "great wonderful amazing love loved lovely loving happy happiness joy joyful glad delighted delight delightful "
    "beautiful excellent fantastic brilliant good best awesome incredible perfect cheerful warm grateful thrilled "
    "enjoy enjoyed enjoyable excited exciting pleased pleasant gorgeous magnificent fun favorite hopeful optimistic "
    "bright smile smiling celebrate wonderfully positively positive splendid marvelous terrific superb".split()
)
NEGATIVE_TONE = frozenset(
    "hate hated awful terrible horrible miserable misery sad sadness angry anger furious dreadful ugly disgusting "
    "depressing depressed bleak gloom gloomy despair pathetic useless annoyed upset bad worst boring grim "
    "unfortunate unhappy painful pain fear afraid worried worry hopeless negative dull tragic".split()
)

# Truth-holding / anti-deference defaults (used when scenario metadata doesn't override them).
DEFAULT_CAPITULATION_MARKERS = (
    "you're right", "you are right", "you are correct", "you're correct", "my apologies, you", "i apologize, you",
    "sorry, you are correct", "i was wrong", "i stand corrected", "yes, you are right", "good point, you",
)


def _content_words(text: str) -> list[str]:
    words = []
    for token in tokenize_text((text or "").lower()):
        stripped = "".join(ch for ch in token if ch.isalnum())
        if len(stripped) >= 4 and stripped not in _STOPWORDS and not stripped.isdigit():
            words.append(stripped)
    return words


def content_overlap(text: str, grounding: str) -> float:
    """Fraction of ``text``'s distinct content words that also appear in ``grounding``.

    Directional. For concision we score how much of the *steered* (concise) output is grounded in
    the *unsteered* baseline — i.e. precision against hallucination/off-topic drift — rather than
    recall of the verbose baseline, which a good summary is *supposed* to drop.
    """
    keys = set(_content_words(text))
    if not keys:
        return 1.0  # nothing to ground (e.g. an all-stopword answer)
    return len(keys & set(_content_words(grounding))) / len(keys)


def is_empty(text: str) -> bool:
    return not (text or "").strip()


def is_collapsed(text: str) -> tuple[bool, float]:
    """Detect degenerate output (repetition collapse / near-zero diversity).

    Returns ``(collapsed, repeated_ngram_rate)``. Empty text counts as collapsed.
    """
    if is_empty(text):
        return True, 0.0
    rep = repeated_ngram_rate(text, 3)
    tokens = tokenize_text(text)
    diversity = distinct_ngram(text, 1)
    collapsed = rep > 0.5 or (len(tokens) >= 8 and diversity < 0.2)
    return collapsed, rep


def _marker_density(text: str, markers: tuple[str, ...]) -> int:
    low = (text or "").lower()
    return sum(low.count(marker) for marker in markers)


def score_concise(text: str, ctx: dict[str, Any] | None = None, *, ref_tokens: int = 80) -> float:
    """Brevity in [0, 1]: 1.0 for an empty-of-length (but non-empty) answer, 0.0 at/above ref_tokens.

    Empty output scores 0.0 (an empty answer is not concise — it's broken).
    """
    if is_empty(text):
        return 0.0
    tokens = len(tokenize_text(text))
    return max(0.0, 1.0 - min(1.0, tokens / max(1, ref_tokens)))


def score_json(text: str, ctx: dict[str, Any] | None = None) -> float:
    return float(json_validity(text or "")["json_validity"])


def score_calibration(text: str, ctx: dict[str, Any] | None = None) -> float:
    """Calibration in [0, 1] around a 0.5 neutral baseline: hedging up, overconfidence down.

    Length-normalised so the *delta* between two answers reflects a real shift in epistemic
    stance rather than raw length. A purely heuristic proxy, not a calibration measurement.
    """
    if is_empty(text):
        return 0.0
    hedges = _marker_density(text, HEDGE_MARKERS)
    over = _marker_density(text, OVERCONFIDENT_MARKERS)
    tokens = max(1, len(tokenize_text(text)))
    raw = (hedges - over) / math.sqrt(tokens)
    return max(0.0, min(1.0, 0.5 + raw))


def score_sentiment(text: str, ctx: dict[str, Any] | None = None) -> float:
    """Positive-tone score in [0, 1] around a 0.5 neutral baseline: positive words up, negative words down.

    Length-normalised so the *delta* between two answers reflects a real shift in affect. A lexicon
    proxy for tone, not a sentiment model.
    """
    if is_empty(text):
        return 0.0
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in POSITIVE_TONE)
    neg = sum(1 for w in words if w in NEGATIVE_TONE)
    raw = (pos - neg) / math.sqrt(len(words))
    return max(0.0, min(1.0, 0.5 + raw))


def _scenario_metadata(ctx: dict[str, Any] | None) -> dict[str, Any]:
    ctx = ctx or {}
    return {
        "truth": str(ctx.get("truth", "")).lower().strip(),
        "capitulation": tuple(m.lower() for m in ctx.get("capitulation_markers", DEFAULT_CAPITULATION_MARKERS)),
        "has_scenario": bool(ctx.get("truth") or ctx.get("false_claim") or ctx.get("capitulation_markers")),
    }


def score_deference(text: str, ctx: dict[str, Any] | None = None) -> float:
    """Truth-holding in [0, 1] around 0.5: holding the truth up, capitulating to a false premise down.

    Uses per-prompt scenario metadata when present (``truth`` token to assert, ``false_claim``,
    custom ``capitulation_markers``). Without scenario metadata it can only watch for generic
    capitulation phrasing.
    """
    if is_empty(text):
        return 0.0
    meta = _scenario_metadata(ctx)
    low = text.lower()
    score = 0.0
    if meta["truth"] and meta["truth"] in low:
        score += 1.0
    if any(marker in low for marker in meta["capitulation"]):
        score -= 1.0
    return max(0.0, min(1.0, 0.5 + score / 2.0))


@dataclass
class CommandScorer:
    """A custom scorer that shells out: writes the candidate text to stdin and reads a float from stdout.

    Lets a user plug in any target metric without changing this module
    (e.g. ``--score-command "python my_judge.py"``).
    """

    command: str
    timeout: float = 60.0

    def __call__(self, text: str, ctx: dict[str, Any] | None = None) -> float:
        proc = subprocess.run(  # noqa: S603 (user-supplied command, by design)
            shlex.split(self.command),
            input=text or "",
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )
        out = (proc.stdout or "").strip().split()
        if not out:
            return 0.0
        try:
            return float(out[-1])
        except ValueError:
            return 0.0


@dataclass
class Target:
    """A target behavior: how to score it, whether to gate on improvement, and extra reject rules."""

    name: str
    description: str
    score: Callable[[str, dict[str, Any] | None], float]
    gate_on_delta: bool = True
    extra_reject: Callable[["dict[str, Any]", "TargetScore", "DistillConfig"], list[str]] = field(
        default=lambda pair, score, cfg: []
    )


def _concise_reject(pair: dict[str, Any], score: "TargetScore", cfg: "DistillConfig") -> list[str]:
    reasons: list[str] = []
    if not score.steered_empty:
        if score.length_ratio > cfg.max_length_ratio:
            reasons.append("not_shorter")
        if not score.content_preserved:
            reasons.append("content_not_preserved")
    return reasons


def _json_reject(pair: dict[str, Any], score: "TargetScore", cfg: "DistillConfig") -> list[str]:
    if not score.steered_empty and score.steered < 1.0:
        return ["steered_invalid_json"]
    return []


def get_target(name: str, cfg: "DistillConfig") -> Target:
    """Resolve a target name into a configured :class:`Target`.

    A ``cfg.score_command`` overrides the built-in scorer (the target name then only selects the
    delta gate and extra-reject rules). Unknown names fall back to the ``generic`` target.
    """
    name = (name or "generic").strip().lower()
    score_fn: Callable[[str, dict[str, Any] | None], float] | None = None
    if cfg.score_command:
        score_fn = CommandScorer(cfg.score_command, timeout=cfg.score_command_timeout)

    if name in {"concise", "concision", "brevity", "concise_answers"}:
        return Target(
            name="concise",
            description="Shorter answers that preserve the key content.",
            score=score_fn or (lambda t, c=None: score_concise(t, c, ref_tokens=cfg.concise_ref_tokens)),
            gate_on_delta=True,
            extra_reject=_concise_reject,
        )
    if name in {"json", "json_validity", "valid_json"}:
        return Target(
            name="json",
            description="Output parses as strict JSON.",
            score=score_fn or score_json,
            gate_on_delta=True,
            extra_reject=_json_reject,
        )
    if name in {"calibrated", "calibration", "uncertainty", "hedging"}:
        return Target(
            name="calibrated",
            description="Better-calibrated answers: hedges appropriately, avoids overconfidence.",
            score=score_fn or score_calibration,
            gate_on_delta=True,
        )
    if name in {"sentiment", "positive_sentiment", "positivity", "cheerful", "tone"}:
        return Target(
            name="sentiment",
            description="More positive / cheerful tone.",
            score=score_fn or score_sentiment,
            gate_on_delta=True,
        )
    if name in {"deference", "truth_holding", "anti_sycophancy", "truth"}:
        return Target(
            name="deference",
            description="Holds the truth under pressure instead of capitulating to a false premise.",
            score=score_fn or score_deference,
            gate_on_delta=True,
        )
    # generic: no behavior gate; only collapse/empty are rejected (unless a custom scorer is given).
    return Target(
        name="generic" if name in {"generic", "", "none"} else name,
        description="No built-in behavior metric; keep any coherent steered output." if not cfg.score_command else f"Custom scorer: {cfg.score_command}",
        score=score_fn or (lambda t, c=None: 0.0),
        gate_on_delta=bool(cfg.score_command),
    )


def available_targets() -> list[str]:
    return ["concise", "json", "calibrated", "deference", "sentiment", "generic"]


@dataclass
class TargetScore:
    target: str
    unsteered: float
    steered: float
    delta: float
    length_ratio: float
    steered_empty: bool
    steered_collapsed: bool
    steered_repetition: float
    content_preserved: bool
    content_overlap: float
    no_scenario_metadata: bool = False


def score_pair(pair: dict[str, Any], target: Target, cfg: "DistillConfig") -> TargetScore:
    unsteered = pair.get("unsteered", "")
    steered = pair.get("steered", "")
    ctx = pair.get("metadata", {})
    us = float(target.score(unsteered, ctx))
    ss = float(target.score(steered, ctx))
    u_tokens = len(tokenize_text(unsteered))
    s_tokens = len(tokenize_text(steered))
    collapsed, rep = is_collapsed(steered)
    overlap = content_overlap(steered, unsteered)  # is the steered output grounded in the baseline?
    no_scenario = target.name == "deference" and not _scenario_metadata(ctx)["has_scenario"]
    return TargetScore(
        target=target.name,
        unsteered=round(us, 4),
        steered=round(ss, 4),
        delta=round(ss - us, 4),
        length_ratio=round(s_tokens / max(1, u_tokens), 4),
        steered_empty=is_empty(steered),
        steered_collapsed=collapsed,
        steered_repetition=round(rep, 4),
        content_preserved=overlap >= cfg.min_content_overlap,
        content_overlap=round(overlap, 4),
        no_scenario_metadata=no_scenario,
    )


# --------------------------------------------------------------------------------------
# 5. Filtering
# --------------------------------------------------------------------------------------


@dataclass
class DistillConfig:
    target: str = "concise"
    min_delta: float = 0.0  # steered must beat unsteered by *more* than this to be kept
    max_length_ratio: float = 1.0  # concise: steered tokens / unsteered tokens must be <= this
    min_content_overlap: float = 0.5  # concise: fraction of key content words preserved
    concise_ref_tokens: int = 80  # length that scores ~0 brevity
    score_command: str | None = None
    score_command_timeout: float = 60.0


@dataclass
class FilterResult:
    keep: bool
    reasons: list[str]
    score: TargetScore


def filter_pair(pair: dict[str, Any], target: Target, cfg: DistillConfig) -> FilterResult:
    """Decide whether a pair is good distillation data, and why not if rejected.

    Reject reasons (retained, never silently dropped):
    ``steered_empty``, ``steered_collapsed``, ``not_shorter``/``content_not_preserved`` (concise),
    ``steered_invalid_json`` (json), and ``no_target_improvement`` (the delta gate).
    """
    score = score_pair(pair, target, cfg)
    reasons: list[str] = []
    if score.steered_empty:
        reasons.append("steered_empty")
    elif score.steered_collapsed:
        reasons.append("steered_collapsed")
    reasons.extend(target.extra_reject(pair, score, cfg))
    if target.gate_on_delta and not score.no_scenario_metadata and not score.steered_empty:
        if score.delta <= cfg.min_delta:
            reasons.append("no_target_improvement")
    # de-dupe while preserving order
    seen: set[str] = set()
    ordered = [r for r in reasons if not (r in seen or seen.add(r))]
    return FilterResult(keep=not ordered, reasons=ordered, score=score)


# --------------------------------------------------------------------------------------
# 6. Distill — score + filter the whole set, compute metrics
# --------------------------------------------------------------------------------------


def _scored_row(pair: dict[str, Any], result: FilterResult) -> dict[str, Any]:
    return {**pair, "scores": asdict(result.score), "keep": result.keep, "reject_reasons": result.reasons}


def compute_metrics(
    scored: list[dict[str, Any]], spec: SteerSpec, cfg: DistillConfig, gen_params: GenParams | None = None
) -> dict[str, Any]:
    n = len(scored)
    kept = [row for row in scored if row["keep"]]
    rejected = [row for row in scored if not row["keep"]]
    deltas = [row["scores"]["delta"] for row in scored]
    kept_deltas = [row["scores"]["delta"] for row in kept]
    collapse = sum(1 for row in scored if row["scores"]["steered_collapsed"])
    empty = sum(1 for row in scored if row["scores"]["steered_empty"])
    reps = [row["scores"]["steered_repetition"] for row in scored]
    reason_counts: dict[str, int] = {}
    for row in rejected:
        for reason in row["reject_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "target": cfg.target,
        "n_prompts": n,
        "n_kept": len(kept),
        "n_rejected": len(rejected),
        "keep_rate": round(len(kept) / n, 4) if n else 0.0,
        "reject_rate": round(len(rejected) / n, 4) if n else 0.0,
        "avg_score_delta": round(sum(deltas) / n, 4) if n else 0.0,
        "avg_score_delta_kept": round(sum(kept_deltas) / len(kept), 4) if kept else 0.0,
        "collapse_rate": round(collapse / n, 4) if n else 0.0,
        "empty_rate": round(empty / n, 4) if n else 0.0,
        "avg_repetition": round(sum(reps) / n, 4) if n else 0.0,
        "reject_reason_counts": dict(sorted(reason_counts.items())),
        "n_sft": len(kept),
        "n_preference": sum(1 for row in kept if row["scores"]["delta"] > 0),
        "recipe_status": spec.recipe_status,
        "validated": spec.validated,
        "source": spec.source,
        "steer": {
            "kind": spec.kind,
            "recipe_id": spec.recipe_id,
            "probe_id": spec.probe_id,
            "model_id": spec.model_id,
            "sae_id": spec.sae_id,
            "layer": spec.layer,
            "feature_id": spec.feature_id,
            "strength": float(spec.strength),
            "injection_mode": spec.injection_mode,
            "target_behavior": spec.target_name,
        },
        "generation": asdict(gen_params) if gen_params is not None else None,
        "filter_config": asdict(cfg),
    }


def distill_pairs(
    pairs: list[dict[str, Any]], spec: SteerSpec, cfg: DistillConfig, gen_params: GenParams | None = None
) -> dict[str, Any]:
    """Score + filter every pair, splitting into kept/rejected, and compute summary metrics.

    ``gen_params`` is recorded in the metrics for provenance when the pairs came from a live
    generation run (it is irrelevant for pre-baked/synthetic pairs).
    """
    target = get_target(cfg.target, cfg)
    scored = [_scored_row(pair, filter_pair(pair, target, cfg)) for pair in pairs]
    kept = [row for row in scored if row["keep"]]
    rejected = [row for row in scored if not row["keep"]]
    return {
        "all": scored,
        "kept": kept,
        "rejected": rejected,
        "metrics": compute_metrics(scored, spec, cfg, gen_params),
    }


# --------------------------------------------------------------------------------------
# 7. Dataset exports
# --------------------------------------------------------------------------------------


def to_sft_records(kept: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """SFT JSONL: the steered output becomes the desired assistant completion for the prompt."""
    return [
        {"messages": [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["steered"]}]}
        for row in kept
    ]


def to_preference_records(kept: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preference JSONL: steered = chosen, unsteered = rejected, for pairs where the steer improved the target."""
    records = []
    for row in kept:
        delta = row["scores"]["delta"]
        if delta > 0:
            records.append(
                {
                    "prompt": row["prompt"],
                    "chosen": row["steered"],
                    "rejected": row["unsteered"],
                    "score_delta": delta,
                }
            )
    return records


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(path)


def _truncate(text: str, limit: int = 280) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _validation_banner(spec: SteerSpec) -> str:
    if spec.validated:
        return (
            "> **Source steer: `validated`.** The recipe beat the prompt-only baseline and all "
            "seven controls. This does *not* guarantee the distilled model will reproduce the "
            "behavior — only that the runtime steer was real."
        )
    return (
        f"> ⚠️ **Source steer: `{spec.recipe_status}` (NOT validated).** This data was compiled from a "
        f"steer that did *not* clearly beat the prompt baseline and every control. Treat the dataset as "
        f"exploratory; the steered outputs may encode artifacts. See *Limitations*."
    )


def render_dataset_card(spec: SteerSpec, metrics: dict[str, Any], cfg: DistillConfig) -> str:
    m = metrics
    limitations = "\n".join(f"- {item}" for item in spec.limitations) if spec.limitations else "- None recorded on the source recipe."
    return (
        f"# Dataset card: steering-distilled `{spec.target_name}`\n\n"
        f"{_validation_banner(spec)}\n\n"
        f"Compiled by **steering-to-data distillation** from a runtime SAE feature steer. The steered\n"
        f"outputs are the desired behavior; the unsteered outputs are the baseline they improve on.\n\n"
        f"## Source steer\n\n"
        f"| field | value |\n|---|---|\n"
        f"| source | `{spec.source}` |\n"
        f"| recipe_id | `{spec.recipe_id or '—'}` |\n"
        f"| recipe_status | `{spec.recipe_status}` |\n"
        f"| model_id | `{spec.model_id or '—'}` |\n"
        f"| sae_id | `{spec.sae_id or '—'}` |\n"
        f"| layer | {spec.layer} |\n"
        f"| feature_id | {spec.feature_id} |\n"
        f"| strength | {spec.strength} |\n"
        f"| injection_mode | `{spec.injection_mode}` |\n"
        f"| target_behavior | `{spec.target_name}` |\n\n"
        f"## Contents\n\n"
        f"- `sft.jsonl` — {m['n_sft']} SFT records (`messages` chat format; steered output as completion).\n"
        f"- `preference.jsonl` — {m['n_preference']} preference pairs (`chosen`=steered, `rejected`=unsteered).\n"
        f"- `pairs_kept.jsonl` / `pairs_rejected.jsonl` / `pairs_all.jsonl` — full provenance per prompt.\n\n"
        f"## Statistics\n\n"
        f"| metric | value |\n|---|---|\n"
        f"| prompts | {m['n_prompts']} |\n"
        f"| kept | {m['n_kept']} ({m['keep_rate']:.0%}) |\n"
        f"| rejected | {m['n_rejected']} ({m['reject_rate']:.0%}) |\n"
        f"| avg target-score delta (all) | {m['avg_score_delta']} |\n"
        f"| avg target-score delta (kept) | {m['avg_score_delta_kept']} |\n"
        f"| collapse rate | {m['collapse_rate']:.0%} |\n"
        f"| empty rate | {m['empty_rate']:.0%} |\n\n"
        f"Reject reasons: {json.dumps(m['reject_reason_counts']) if m['reject_reason_counts'] else 'none'}\n\n"
        f"## Filtering\n\n"
        f"Target `{cfg.target}`: kept iff the steered output improved the target score by > {cfg.min_delta} "
        f"and did not collapse/empty"
        + (f", was no longer than the baseline (ratio ≤ {cfg.max_length_ratio}) and preserved ≥ {cfg.min_content_overlap:.0%} of key content words" if cfg.target == 'concise' else "")
        + ". Rejected pairs are retained in `pairs_rejected.jsonl` for inspection.\n\n"
        f"## Intended use & limitations\n\n"
        f"Intended for **SFT**, **preference tuning (DPO)**, or **LoRA distillation** experiments that ask whether a\n"
        f"runtime steer can be learned into the weights. It is *not* a benchmark and carries no guarantee that a\n"
        f"trained model reproduces the behavior. Known limitations:\n\n"
        f"{limitations}\n"
        f"- Steered outputs may encode steering artifacts (register shifts, truncation) alongside the target behavior.\n"
        f"- Recipe validation gates the *source* steer, not the distilled outcome.\n"
        f"- Do not train on collapsed/empty outputs — they are filtered here, but re-check after any schema change.\n\n"
        f"_Generated {m['generated_at']} · schema {m['schema_version']}._\n"
    )


def render_report(spec: SteerSpec, result: dict[str, Any], cfg: DistillConfig) -> str:
    m = result["metrics"]
    kept = result["kept"]
    rejected = result["rejected"]

    def example_block(row: dict[str, Any], show_reasons: bool) -> str:
        sc = row["scores"]
        head = f"- **`{row['id']}`**"
        if show_reasons:
            head += f" — rejected: `{', '.join(row['reject_reasons'])}`"
        head += f" (Δ={sc['delta']}, len_ratio={sc['length_ratio']}, overlap={sc['content_overlap']})"
        return (
            f"{head}\n"
            f"  - prompt: {_truncate(row['prompt'], 160)}\n"
            f"  - unsteered: {_truncate(row['unsteered'])}\n"
            f"  - steered: {_truncate(row['steered'])}"
        )

    kept_examples = "\n".join(example_block(row, False) for row in kept[:3]) or "_None kept._"
    rejected_examples = "\n".join(example_block(row, True) for row in rejected[:3]) or "_None rejected._"
    reasons = "\n".join(f"- `{reason}`: {count}" for reason, count in m["reject_reason_counts"].items()) or "- None."

    return (
        f"# Steering-to-data distillation report — `{spec.target_name}`\n\n"
        f"{_validation_banner(spec)}\n\n"
        f"## Source steer / config\n\n"
        f"- source: `{spec.source}`"
        + (f" · recipe `{spec.recipe_id}` (`{spec.recipe_status}`)" if spec.recipe_id else f" · status `{spec.recipe_status}`")
        + "\n"
        f"- model: `{spec.model_id or '—'}` · SAE: `{spec.sae_id or '—'}`\n"
        f"- intervention: layer {spec.layer}, feature {spec.feature_id}, strength {spec.strength}, mode `{spec.injection_mode}`\n"
        f"- target: **{spec.target_name}** — {spec.target_description}\n\n"
        f"## Results\n\n"
        f"- prompts: **{m['n_prompts']}** · kept: **{m['n_kept']}** ({m['keep_rate']:.0%}) · rejected: **{m['n_rejected']}** ({m['reject_rate']:.0%})\n"
        f"- avg target-score delta: **{m['avg_score_delta']}** (all) / **{m['avg_score_delta_kept']}** (kept)\n"
        f"- collapse rate: **{m['collapse_rate']:.0%}** · empty rate: **{m['empty_rate']:.0%}** · avg repetition: {m['avg_repetition']}\n"
        f"- exported: **{m['n_sft']}** SFT records, **{m['n_preference']}** preference pairs\n\n"
        f"### Reject reasons\n\n{reasons}\n\n"
        f"### Kept examples\n\n{kept_examples}\n\n"
        f"### Rejected examples\n\n{rejected_examples}\n\n"
        f"## Limitations & honesty\n\n"
        f"- This report describes **training data**, not a trained model. No claim is made that a model "
        f"fine-tuned on this data reproduces the `{spec.target_name}` behavior — that requires actually "
        f"training and evaluating an adapter (see the `eval` subcommand).\n"
        f"- The source steer was **`{spec.recipe_status}`**"
        + ("" if spec.validated else " — *not* validated against the full control battery") + ".\n"
        f"- Scores are heuristic proxies for `{spec.target_name}`, not ground-truth labels.\n"
        f"- Steered outputs can carry artifacts; inspect `pairs_rejected.jsonl` and a sample of `pairs_kept.jsonl` before training.\n\n"
        f"_Generated {m['generated_at']} · schema {m['schema_version']}._\n"
    )


def write_outputs(out_dir: str | Path, spec: SteerSpec, result: dict[str, Any], cfg: DistillConfig) -> dict[str, str]:
    """Write the full artifact set to ``out_dir`` and return a map of name -> path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = result["metrics"]
    paths = {
        "pairs_all": write_jsonl(out / "pairs_all.jsonl", result["all"]),
        "pairs_kept": write_jsonl(out / "pairs_kept.jsonl", result["kept"]),
        "pairs_rejected": write_jsonl(out / "pairs_rejected.jsonl", result["rejected"]),
        "sft": write_jsonl(out / "sft.jsonl", to_sft_records(result["kept"])),
        "preference": write_jsonl(out / "preference.jsonl", to_preference_records(result["kept"])),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "dataset_card.md").write_text(render_dataset_card(spec, metrics, cfg), encoding="utf-8")
    (out / "report.md").write_text(render_report(spec, result, cfg), encoding="utf-8")
    paths["metrics"] = str(out / "metrics.json")
    paths["dataset_card"] = str(out / "dataset_card.md")
    paths["report"] = str(out / "report.md")
    return paths


# --------------------------------------------------------------------------------------
# 8. Optional distilled-model evaluation
# --------------------------------------------------------------------------------------


def evaluate(arms: dict[str, list[dict[str, Any]]], cfg: DistillConfig) -> dict[str, Any]:
    """Score each evaluation arm with the target scorer and compare them.

    ``arms`` maps an arm name (e.g. ``baseline`` / ``runtime_steering`` / ``distilled`` /
    ``prompt_only``) to a list of ``{id, prompt, output, metadata}`` rows. Returns per-arm mean
    target score / collapse rate / mean length, the delta of each arm vs the ``baseline`` arm,
    and a ranking. Pure: the CLI fills the arms from URLs; tests pass crafted outputs.
    """
    target = get_target(cfg.target, cfg)
    summary: dict[str, Any] = {}
    for name, rows in arms.items():
        scores = [float(target.score(r.get("output", ""), r.get("metadata", {}))) for r in rows]
        collapses = [is_collapsed(r.get("output", ""))[0] for r in rows]
        lengths = [len(tokenize_text(r.get("output", ""))) for r in rows]
        n = len(rows)
        summary[name] = {
            "n": n,
            "mean_target_score": round(sum(scores) / n, 4) if n else 0.0,
            "collapse_rate": round(sum(collapses) / n, 4) if n else 0.0,
            "mean_tokens": round(sum(lengths) / n, 2) if n else 0.0,
        }
    base = summary.get("baseline", {}).get("mean_target_score")
    delta_vs_baseline = (
        {name: round(v["mean_target_score"] - base, 4) for name, v in summary.items()} if base is not None else {}
    )
    ranking = [name for name, _ in sorted(summary.items(), key=lambda kv: kv[1]["mean_target_score"], reverse=True)]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "target": cfg.target,
        "arms": summary,
        "delta_vs_baseline": delta_vs_baseline,
        "ranking": ranking,
    }


def render_eval_report(eval_result: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| `{name}` | {v['n']} | {v['mean_target_score']} | "
        f"{eval_result['delta_vs_baseline'].get(name, '—')} | {v['collapse_rate']:.0%} | {v['mean_tokens']} |"
        for name, v in eval_result["arms"].items()
    )
    winner = eval_result["ranking"][0] if eval_result["ranking"] else "—"
    distilled = eval_result["arms"].get("distilled", {}).get("mean_target_score")
    runtime = eval_result["arms"].get("runtime_steering", {}).get("mean_target_score")
    gap_line = ""
    if distilled is not None and runtime is not None:
        gap = round(runtime - distilled, 4)
        gap_line = (
            f"\nDistilled vs runtime-steering gap: **{gap}** "
            f"({'distilled recovers the steer' if gap <= 0 else 'distilled has not fully recovered the steer'}).\n"
        )
    return (
        f"# Distilled-model evaluation — target `{eval_result['target']}`\n\n"
        f"Higher target score is better. `delta_vs_baseline` is each arm minus the baseline arm.\n\n"
        f"| arm | n | mean target score | Δ vs baseline | collapse rate | mean tokens |\n"
        f"|---|---|---|---|---|---|\n{rows}\n\n"
        f"Best arm by target score: **`{winner}`**.\n"
        f"{gap_line}\n"
        f"> This compares *measured target scores* of each arm's outputs. It does not by itself prove the "
        f"distilled model is safe or generally capable — pair it with collateral/quality checks before deployment.\n\n"
        f"_Generated {eval_result['generated_at']} · schema {eval_result['schema_version']}._\n"
    )


# --------------------------------------------------------------------------------------
# 9. Synthetic fixtures — drive the whole pipeline with no model (CI / smoke)
# --------------------------------------------------------------------------------------


def _pair(
    pid: str,
    prompt: str,
    unsteered: str,
    steered: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": pid,
        "prompt": prompt,
        "metadata": metadata or {},
        "unsteered": unsteered,
        "steered": steered,
        "prompt_only": "",
        "random_control": "",
        "target_behavior": "synthetic",
        "recipe_status": "benchmarked",
        "recipe_id": "synthetic_smoke",
        "model_id": "synthetic/echo-distill",
        "layer": 6,
        "feature_id": 1234,
        "strength": 6.0,
        "hook_fired": True,
        "hidden_delta_norm": 6.0,
    }


def _synthetic_concise() -> list[dict[str, Any]]:
    return [
        _pair(
            "concise_keep_1",
            "Summarize the main causes of World War I.",
            "There were many interlocking causes of World War I, and historians still debate their relative "
            "weight, but the most commonly cited factors are militarism, the system of entangling alliances, "
            "imperialism, rising nationalism, and the immediate trigger of the assassination of Archduke Franz Ferdinand.",
            "Militarism, alliances, imperialism, nationalism, and the assassination of Franz Ferdinand caused World War I.",
        ),
        _pair(
            "concise_keep_2",
            "What is photosynthesis?",
            "Photosynthesis is a fairly involved biochemical process in which green plants and some other organisms "
            "use sunlight, water, and carbon dioxide together with the pigment chlorophyll to produce glucose for "
            "energy and release oxygen as a by-product into the atmosphere.",
            "Plants use sunlight, water, and carbon dioxide to make glucose and release oxygen, via chlorophyll.",
        ),
        _pair(
            "concise_reject_collapse",
            "Define entropy in thermodynamics.",
            "Entropy is a measure of the disorder or the number of microscopic configurations of a thermodynamic system.",
            "entropy entropy entropy entropy entropy entropy entropy entropy entropy entropy entropy entropy",
        ),
        _pair(
            "concise_reject_longer",
            "What is the capital of France?",
            "Paris is the capital of France.",
            "The capital of France, a country located in Western Europe, is the historic and beautiful city of "
            "Paris, which sits on the river Seine and has been the capital for many centuries.",
        ),
        _pair(
            "concise_reject_content",
            "Describe the water cycle.",
            "The water cycle describes how water moves through evaporation from oceans, condensation into clouds, "
            "precipitation as rain or snow, and collection in rivers, lakes, and groundwater before repeating.",
            "It is a natural process that simply happens over and over again.",
        ),
    ]


def _synthetic_json() -> list[dict[str, Any]]:
    return [
        _pair(
            "json_keep_1",
            "Return a JSON object with keys name and age for a fictional person.",
            "Sure! Here is a person: the name is Ada and the age is thirty-one years old.",
            '{"name": "Ada", "age": 31}',
        ),
        _pair(
            "json_keep_2",
            "Return a JSON object describing a search action with route and arguments.",
            "The route is search and the arguments contain a query about sparse autoencoders.",
            '{"route": "search", "arguments": {"query": "sparse autoencoder"}}',
        ),
        _pair(
            "json_reject_fenced",
            "Return a JSON object with a single key status set to ok.",
            "Here you go.",
            '```json\n{"status": "ok"}\n```',
        ),
        _pair(
            "json_reject_no_improvement",
            "Return a JSON object with key result set to 42.",
            '{"result": 42}',
            '{"result": 42}',
        ),
    ]


def _synthetic_calibrated() -> list[dict[str, Any]]:
    return [
        _pair(
            "calib_keep_1",
            "Will it rain tomorrow in Paris?",
            "It will definitely rain tomorrow in Paris, absolutely, no question about it.",
            "It might rain tomorrow in Paris; I'm not certain, but it seems fairly likely given the season.",
        ),
        _pair(
            "calib_keep_2",
            "How many people will attend the conference?",
            "There will certainly be exactly 500 attendees, guaranteed.",
            "There will probably be around 500 attendees, though I'm not sure — it could be somewhat fewer.",
        ),
        _pair(
            "calib_reject_empty",
            "Is this stock going to go up?",
            "This stock will absolutely go up, definitely, for sure.",
            "   ",
        ),
        _pair(
            "calib_reject_no_improvement",
            "What is the population of Mars?",
            "Mars probably has no permanent human population as far as I know.",
            "Mars probably has no permanent human population as far as I know.",
        ),
    ]


def build_synthetic_pairs(target: str = "concise") -> list[dict[str, Any]]:
    """A deterministic, hand-crafted set of pairs that yields a realistic keep/reject mix.

    Used by the ``synthetic-smoke`` CLI subcommand and the tests to drive the full
    score → filter → export → report path with no model and no network.
    """
    name = (target or "concise").strip().lower()
    if name in {"json", "json_validity", "valid_json"}:
        return _synthetic_json()
    if name in {"calibrated", "calibration", "uncertainty", "hedging"}:
        return _synthetic_calibrated()
    return _synthetic_concise()


def build_synthetic_eval_arms() -> dict[str, list[dict[str, Any]]]:
    """Crafted eval arms (concise target) demonstrating the ranking baseline < distilled ≈ runtime."""
    prompts = [
        ("e1", "Summarize the theory of relativity in one sentence."),
        ("e2", "Explain why the sky is blue, briefly."),
    ]
    baseline = [
        "The theory of relativity is a really long and elaborate set of ideas developed by Albert Einstein over "
        "many years that fundamentally changed how physicists understand space, time, gravity, and motion.",
        "The sky appears blue for a number of interrelated reasons having to do with the physics of light and "
        "the way the atmosphere scatters different wavelengths across the visible spectrum throughout the day.",
    ]
    runtime = [
        "Relativity links space, time, and gravity: motion and mass bend spacetime.",
        "Air scatters short blue wavelengths more than red, so the sky looks blue.",
    ]
    distilled = [
        "Relativity ties space, time, and gravity together through spacetime.",
        "The atmosphere scatters blue light most, making the sky look blue.",
    ]
    prompt_only = [
        "Relativity is Einstein's theory about space and time being relative to the observer and motion.",
        "The sky is blue because of how sunlight interacts with the atmosphere and scatters around.",
    ]

    def arm(outputs: list[str]) -> list[dict[str, Any]]:
        return [{"id": pid, "prompt": prompt, "output": out, "metadata": {}} for (pid, prompt), out in zip(prompts, outputs)]

    return {
        "baseline": arm(baseline),
        "runtime_steering": arm(runtime),
        "distilled": arm(distilled),
        "prompt_only": arm(prompt_only),
    }
