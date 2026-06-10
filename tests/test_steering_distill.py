"""Tests for the steering-to-data distillation pipeline. All CI-safe: no GPU, no model, no network."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from qwen_scope_lab.benchmark import EchoGenerationBackend
from qwen_scope_lab.experiments import steering_distill as sd
from qwen_scope_lab.recipe_schema import (
    FeatureRecipe,
    Intervention,
    ManifoldSpec,
    ModelMetadata,
    TargetBehavior,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "steering_to_data_distill.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("steering_cli_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------
# Recipe loading + explicit config parsing
# --------------------------------------------------------------------------------------


def _feature_recipe(status: str = "draft") -> FeatureRecipe:
    recipe = FeatureRecipe.create(
        target_behavior=TargetBehavior(name="concise", description="Steer toward concise answers."),
        model=ModelMetadata(model_id="qwen/2b", sae_id="qwen/sae"),
        interventions=[Intervention(layer=12, feature_id=1234, strength=6.0)],
    )
    recipe.status = status
    recipe.limitations = ["small sample"]
    return recipe


def test_spec_from_recipe_carries_provenance():
    spec = sd.SteerSpec.from_recipe(_feature_recipe(status="benchmarked"))
    assert spec.source == "recipe"
    assert (spec.feature_id, spec.layer, spec.strength) == (1234, 12, 6.0)
    assert spec.target_name == "concise"
    assert spec.recipe_status == "benchmarked"
    assert spec.validated is False
    assert spec.model_id == "qwen/2b" and spec.sae_id == "qwen/sae"
    assert "small sample" in spec.limitations


def test_spec_validated_flag_tracks_status():
    benchmarked = sd.SteerSpec.from_recipe(_feature_recipe(status="benchmarked"))
    assert benchmarked.validated is False
    # Mark a recipe validated by hand (bypassing the evidence gate) to exercise the flag only.
    recipe = _feature_recipe()
    recipe.status = "validated"
    spec = sd.SteerSpec.from_recipe(recipe)
    assert spec.validated is True


def test_spec_from_committed_recipe_file():
    path = ROOT / "recipes" / "concise_answers_l1_f2_v1" / "recipe.json"
    spec = sd.SteerSpec.from_recipe(sd.load_recipe(path))
    assert spec.source == "recipe"
    assert spec.feature_id == 2 and spec.layer == 1
    assert spec.recipe_status == "draft" and spec.validated is False


def test_from_recipe_rejects_manifold():
    recipe = FeatureRecipe.create_manifold(
        target_behavior=TargetBehavior(name="days", description="Traverse the days-of-week ring."),
        model=ModelMetadata(model_id="qwen/2b", sae_id="qwen/sae"),
        manifold=ManifoldSpec(concept="weekday", source="Monday", target="Friday", layer=6),
    )
    with pytest.raises(ValueError):
        sd.SteerSpec.from_recipe(recipe)


def test_explicit_spec_defaults_to_candidate():
    spec = sd.SteerSpec.explicit(feature_id=7, layer=3, strength=8.0, target_name="json")
    assert spec.source == "explicit"
    assert spec.recipe_status == "candidate" and spec.validated is False
    assert spec.target_description  # auto-filled
    iv = spec.intervention()
    assert (iv.layer, iv.feature_id, iv.strength) == (3, 7, 8.0)


def test_cli_build_spec_explicit_and_range_check():
    cli = _load_cli_module()
    args = argparse.Namespace(
        recipe=None, feature_id=5, layer=1, strength=6.0, target_name="concise", target_description=""
    )
    spec = cli.build_spec(args, {"model_id": "m", "sae_id": "s", "num_layers": 4, "d_sae": 1000})
    assert spec.model_id == "m" and spec.sae_id == "s"
    # out-of-range layer/feature must fail loudly
    bad = argparse.Namespace(recipe=None, feature_id=5, layer=99, strength=6.0, target_name="concise", target_description="")
    with pytest.raises(SystemExit):
        cli.build_spec(bad, {"num_layers": 4, "d_sae": 1000})


def test_cli_build_spec_missing_explicit_args_errors():
    cli = _load_cli_module()
    args = argparse.Namespace(recipe=None, feature_id=None, layer=None, strength=None, target_name=None)
    with pytest.raises(SystemExit):
        cli.build_spec(args, {})


# --------------------------------------------------------------------------------------
# Scoring functions
# --------------------------------------------------------------------------------------


def test_score_concise_shorter_scores_higher():
    long = "This is a fairly long and rambling answer that goes on for quite a while without getting to the point."
    short = "Short answer."
    assert sd.score_concise(short) > sd.score_concise(long)
    assert sd.score_concise("") == 0.0  # empty is not concise, it's broken


def test_score_json_validity():
    assert sd.score_json('{"a": 1}') == 1.0
    assert sd.score_json("```json\n{}\n```") == 0.0
    assert sd.score_json("not json") == 0.0


def test_score_calibration_rewards_hedging_over_overconfidence():
    confident = "This will definitely happen, absolutely, without a doubt, guaranteed."
    hedged = "This might happen; I'm not sure, but it seems possible and is probably likely."
    assert sd.score_calibration(hedged) > sd.score_calibration(confident)


def test_score_deference_uses_scenario_metadata():
    ctx = {"truth": "equals 4", "false_claim": "equals 5"}
    holds = sd.score_deference("No, two plus two equals 4, not 5.", ctx)
    caves = sd.score_deference("You're right, I was wrong, it equals 5.", ctx)
    assert holds > caves


def test_score_sentiment_rewards_positive_tone():
    cheerful = "What a wonderful, delightful day! I am so happy and grateful and excited."
    grim = "This is awful and miserable; I am angry, sad, and full of despair."
    assert sd.score_sentiment(cheerful) > sd.score_sentiment(grim)
    assert sd.score_sentiment("") == 0.0
    assert "sentiment" in sd.available_targets()


def test_collapse_and_overlap_helpers():
    assert sd.is_empty("   ") is True
    collapsed, rep = sd.is_collapsed("spam spam spam spam spam spam spam spam spam spam")
    assert collapsed is True and rep > 0.0
    assert sd.is_collapsed("A perfectly normal and diverse sentence about cats.")[0] is False
    # directional: steered grounded in baseline
    assert sd.content_overlap("alpha beta", "alpha beta gamma delta") == 1.0
    assert sd.content_overlap("alpha beta", "gamma delta") == 0.0


# --------------------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------------------


def _make_pair(unsteered: str, steered: str, metadata: dict | None = None) -> dict:
    return {
        "id": "x",
        "prompt": "Explain something at length.",
        "metadata": metadata or {},
        "unsteered": unsteered,
        "steered": steered,
    }


def test_filter_keeps_genuine_concise_improvement():
    cfg = sd.DistillConfig(target="concise")
    target = sd.get_target("concise", cfg)
    pair = _make_pair(
        "The mitochondria is the powerhouse of the cell, producing energy through respiration in a long-winded way.",
        "The mitochondria produces the cell's energy through respiration.",
    )
    result = sd.filter_pair(pair, target, cfg)
    assert result.keep is True and result.reasons == []
    assert result.score.delta > 0


def test_filter_rejects_collapsed_empty_longer_and_ungrounded():
    cfg = sd.DistillConfig(target="concise")
    target = sd.get_target("concise", cfg)
    base = "Paris is the capital of France and a major European cultural center."

    empty = sd.filter_pair(_make_pair(base, "   "), target, cfg)
    assert empty.keep is False and "steered_empty" in empty.reasons

    collapsed = sd.filter_pair(_make_pair(base, "no no no no no no no no no no no"), target, cfg)
    assert collapsed.keep is False and "steered_collapsed" in collapsed.reasons

    longer = sd.filter_pair(
        _make_pair("Paris.", "Paris is the lovely and historic capital city of the nation of France in Europe."),
        target,
        cfg,
    )
    assert longer.keep is False and "not_shorter" in longer.reasons

    ungrounded = sd.filter_pair(_make_pair(base, "Bananas are yellow tropical fruit."), target, cfg)
    assert ungrounded.keep is False and "content_not_preserved" in ungrounded.reasons


def test_filter_json_rejects_invalid_and_no_improvement():
    cfg = sd.DistillConfig(target="json")
    target = sd.get_target("json", cfg)
    invalid = sd.filter_pair(_make_pair("here is data", "```json\n{}\n```"), target, cfg)
    assert invalid.keep is False and "steered_invalid_json" in invalid.reasons
    no_gain = sd.filter_pair(_make_pair('{"a":1}', '{"a":1}'), target, cfg)
    assert no_gain.keep is False and "no_target_improvement" in no_gain.reasons


def test_generic_target_keeps_any_coherent_output():
    cfg = sd.DistillConfig(target="generic")
    target = sd.get_target("generic", cfg)
    assert target.gate_on_delta is False
    keep = sd.filter_pair(_make_pair("baseline output", "a totally different coherent answer"), target, cfg)
    assert keep.keep is True
    reject = sd.filter_pair(_make_pair("baseline", ""), target, cfg)
    assert reject.keep is False and "steered_empty" in reject.reasons


def test_command_scorer_plugs_in_and_gates():
    # A custom scorer that returns the token count of the candidate text.
    cmd = f'{sys.executable} -c "import sys;print(len(sys.stdin.read().split()))"'
    cfg = sd.DistillConfig(target="generic", score_command=cmd)
    target = sd.get_target("generic", cfg)
    assert target.gate_on_delta is True  # a custom scorer enables the delta gate
    assert target.score("one two three", None) == 3.0
    pair = _make_pair("short", "many more words than before here now")
    assert sd.filter_pair(pair, target, cfg).keep is True  # steered longer -> higher score -> kept


# --------------------------------------------------------------------------------------
# Distill metrics + exports
# --------------------------------------------------------------------------------------


def _distilled(target: str = "concise"):
    spec = sd.SteerSpec.explicit(feature_id=1, layer=1, strength=6.0, target_name=target)
    spec.source = "synthetic"
    cfg = sd.DistillConfig(target=target)
    return spec, cfg, sd.distill_pairs(sd.build_synthetic_pairs(target), spec, cfg)


@pytest.mark.parametrize("target", ["concise", "json", "calibrated"])
def test_synthetic_pairs_yield_keep_and_reject(target):
    _, _, result = _distilled(target)
    m = result["metrics"]
    assert m["n_kept"] >= 1 and m["n_rejected"] >= 1
    assert m["n_prompts"] == m["n_kept"] + m["n_rejected"]
    assert m["reject_reason_counts"]  # at least one reason recorded


def test_metrics_shape_and_rates():
    _, _, result = _distilled("concise")
    m = result["metrics"]
    assert m["keep_rate"] == round(m["n_kept"] / m["n_prompts"], 4)
    assert set(m["steer"]) >= {"feature_id", "layer", "strength", "model_id", "target_behavior"}
    assert m["schema_version"] == sd.SCHEMA_VERSION
    assert "validated" in m and "recipe_status" in m


def test_sft_export_schema():
    _, _, result = _distilled("concise")
    records = sd.to_sft_records(result["kept"])
    assert len(records) == result["metrics"]["n_kept"]
    for rec, kept in zip(records, result["kept"]):
        roles = [m["role"] for m in rec["messages"]]
        assert roles == ["user", "assistant"]
        assert rec["messages"][0]["content"] == kept["prompt"]
        assert rec["messages"][1]["content"] == kept["steered"]


def test_preference_export_schema_uses_steered_as_chosen():
    _, _, result = _distilled("concise")
    records = sd.to_preference_records(result["kept"])
    assert records, "expected at least one preference pair"
    for rec in records:
        assert set(rec) == {"prompt", "chosen", "rejected", "score_delta"}
        assert rec["score_delta"] > 0
    # chosen must be the steered output of some kept pair
    steered_outputs = {row["steered"] for row in result["kept"]}
    assert all(rec["chosen"] in steered_outputs for rec in records)


def test_write_outputs_creates_full_artifact_set(tmp_path):
    spec, cfg, result = _distilled("concise")
    paths = sd.write_outputs(tmp_path, spec, result, cfg)
    expected = {"pairs_all", "pairs_kept", "pairs_rejected", "sft", "preference", "metrics", "dataset_card", "report"}
    assert set(paths) == expected
    for name in expected:
        assert Path(paths[name]).exists()
    # metrics.json round-trips
    metrics = json.loads(Path(paths["metrics"]).read_text())
    assert metrics["n_prompts"] == result["metrics"]["n_prompts"]
    # every pairs_all line is valid JSON with a verdict
    for line in Path(paths["pairs_all"]).read_text().splitlines():
        row = json.loads(line)
        assert "keep" in row and "scores" in row and "reject_reasons" in row


def test_report_and_card_flag_unvalidated_source(tmp_path):
    spec, cfg, result = _distilled("concise")
    spec.recipe_status = "benchmarked"
    report = sd.render_report(spec, result, cfg)
    card = sd.render_dataset_card(spec, result["metrics"], cfg)
    assert "NOT validated" in report and "training data" in report
    assert "NOT validated" in card
    # a validated source flips the banner
    spec.recipe_status = "validated"
    assert "validated" in sd.render_report(spec, result, cfg).lower()


# --------------------------------------------------------------------------------------
# Pair generation wiring (echo backend — no model)
# --------------------------------------------------------------------------------------


class _DirectionStubBackend:
    """A model-free backend that supports direction steering, for testing the CAA path."""

    d_sae = 32768

    def generate(self, prompt, *, max_new_tokens, temperature, seed=0):
        return {"text": f"plain answer to: {prompt}"}

    def steer_direction(self, prompt, *, layer, strength, probe_id="", direction=None, max_new_tokens, temperature, seed=0):
        return {
            "unsteered_text": f"plain answer to: {prompt}",
            "steered_text": f"What a wonderful, delightful answer to: {prompt}",
            "hook_fired": True,
            "hidden_delta_norm": float(strength),
        }


def test_spec_from_probe_is_direction():
    spec = sd.SteerSpec.from_probe(probe_id="pos_sent_l12", layer=12, strength=6.0, target_name="sentiment", model_id="m")
    assert spec.is_direction is True and spec.kind == "direction"
    assert spec.probe_id == "pos_sent_l12" and spec.source == "probe" and spec.feature_id == -1


def test_generate_pairs_direction_mode():
    spec = sd.SteerSpec.from_probe(probe_id="pos_sent_l12", layer=12, strength=6.0, target_name="sentiment", model_id="m")
    prompts = [{"id": "a", "prompt": "Tell me about your day.", "metadata": {}}]
    pairs = sd.generate_pairs(spec, prompts, _DirectionStubBackend(), sd.GenParams(max_new_tokens=16))
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["kind"] == "direction" and pair["probe_id"] == "pos_sent_l12"
    assert "wonderful" in pair["steered"] and "wonderful" not in pair["unsteered"]
    # the whole distill path works on direction pairs
    cfg = sd.DistillConfig(target="sentiment")
    result = sd.distill_pairs(pairs, spec, cfg)
    assert result["metrics"]["n_kept"] == 1
    assert result["metrics"]["steer"]["kind"] == "direction"


def test_cli_build_spec_probe_mode():
    cli = _load_cli_module()
    args = argparse.Namespace(probe_id="pos_sent_l12", recipe=None, layer=12, strength=6.0,
                              target_name="sentiment", target_description="")
    spec = cli.build_spec(args, {"model_id": "m", "num_layers": 24, "d_sae": 32768})
    assert spec.is_direction and spec.probe_id == "pos_sent_l12" and spec.model_id == "m"


def test_generate_pairs_with_echo_backend():
    spec = sd.SteerSpec.explicit(feature_id=5, layer=1, strength=6.0, target_name="concise", model_id="echo", sae_id="echo")
    prompts = [{"id": "a", "prompt": "Tell me about steering.", "metadata": {}}]
    params = sd.GenParams(max_new_tokens=16, prompt_only_instruction="Be concise.", include_random_control=True)
    pairs = sd.generate_pairs(spec, prompts, EchoGenerationBackend(), params)
    assert len(pairs) == 1
    pair = pairs[0]
    required = {"id", "prompt", "unsteered", "steered", "prompt_only", "random_control", "feature_id", "layer", "strength", "hook_fired"}
    assert required <= set(pair)
    assert pair["unsteered"] and pair["steered"]
    assert pair["prompt_only"]  # instruction supplied
    assert pair["random_control"]  # control requested
    assert pair["feature_id"] == 5 and pair["layer"] == 1


# --------------------------------------------------------------------------------------
# Evaluation harness
# --------------------------------------------------------------------------------------


def test_evaluate_ranks_arms_and_deltas():
    cfg = sd.DistillConfig(target="concise")
    result = sd.evaluate(sd.build_synthetic_eval_arms(), cfg)
    assert result["delta_vs_baseline"]["baseline"] == 0.0
    # steered + distilled should beat the verbose baseline on the concise metric
    assert result["delta_vs_baseline"]["runtime_steering"] > 0
    assert result["delta_vs_baseline"]["distilled"] > 0
    assert result["ranking"][0] in {"distilled", "runtime_steering"}
    assert result["ranking"][-1] == "baseline"


def test_render_eval_report_has_table():
    cfg = sd.DistillConfig(target="concise")
    report = sd.render_eval_report(sd.evaluate(sd.build_synthetic_eval_arms(), cfg))
    assert "mean target score" in report and "baseline" in report and "distilled" in report


# --------------------------------------------------------------------------------------
# CLI subcommands (subprocess — exercises the real entrypoint)
# --------------------------------------------------------------------------------------


def test_cli_synthetic_smoke(tmp_path):
    out = tmp_path / "smoke"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "synthetic-smoke", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["metrics"]["n_kept"] >= 2
    for name in ("sft.jsonl", "preference.jsonl", "pairs_rejected.jsonl", "dataset_card.md", "report.md", "metrics.json"):
        assert (out / name).exists()


def test_cli_eval_synthetic(tmp_path):
    out = tmp_path / "eval"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "eval", "--synthetic", "--target", "concise", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["eval"]["ranking"][-1] == "baseline"
    assert (out / "eval_report.md").exists() and (out / "eval_metrics.json").exists()


def test_cli_generate_with_echo_backend(tmp_path):
    out = tmp_path / "gen"
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT), "generate", "--backend", "echo",
            "--feature-id", "5", "--layer", "1", "--strength", "6", "--target-name", "concise",
            "--prompts", str(ROOT / "data" / "experiments" / "steering_distill" / "prompts.jsonl"),
            "--out", str(out),
        ],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["metrics"]["n_prompts"] == 12
    assert (out / "report.md").exists()
