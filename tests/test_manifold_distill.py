"""C09 — manifold-to-data provenance compiler (first build step).

CI-safe: drives the full normalize → gate → export path from synthetic manifold payloads, with no
model and no network. Proves manifold-recipe acceptance, gate keep/reject, rejected-record
retention, equal-size arm accounting, SFT/preference shape, and the honesty/source-gap caveats.
"""
import json

import pytest

from qwen_scope_lab.experiments import manifold_distill as md
from qwen_scope_lab.recipe_schema import (
    FeatureRecipe,
    Intervention,
    ManifoldSpec,
    ModelMetadata,
    TargetBehavior,
)


def _manifold_recipe():
    return FeatureRecipe.create_manifold(
        TargetBehavior(name="rank_general", description="Steer military rank toward general."),
        ModelMetadata(model_id="mlx-community/Qwen3.5-2B-bf16", sae_id="mlx://none"),
        ManifoldSpec(concept="rank", source="private", target="general", layer=20, path="manifold", n_waypoints=5),
    )


def _spec():
    return md.ManifoldDataSpec.explicit(concept="rank", source="private", target="general", layer=20)


def test_manifold_recipe_accepted_feature_recipe_rejected():
    spec = md.spec_from_recipe(_manifold_recipe())
    assert spec.concept == "rank" and spec.source == "private" and spec.target == "general"
    assert spec.source_kind == "recipe" and spec.recipe_id

    feature_recipe = FeatureRecipe.create(
        TargetBehavior(name="concise", description="Shorter answers."),
        ModelMetadata(model_id="m", sae_id="s"),
        [Intervention(layer=6, feature_id=12, strength=6.0)],
    )
    with pytest.raises(ValueError):
        md.ManifoldDataSpec.from_recipe(feature_recipe)


def test_win_keeps_on_manifold_arms():
    graded = md.compile_payload(md.build_synthetic_payload("win"), _spec(), md.GateConfig(min_recovered_r=0.5))
    m = graded["metrics"]
    assert m["n_kept"] == m["n_records"] == 15
    assert m["per_arm"]["manifold"]["n_kept"] == 5
    assert m["per_arm"]["pullback"]["n_kept"] == 5
    assert m["reject_reason_counts"] == {}
    assert m["n_sft"] == 10 and m["n_preference"] == 10
    assert m["equal_size_n_per_arm"] == 5


def test_fail_rejects_on_geometry_and_collapse():
    graded = md.compile_payload(md.build_synthetic_payload("fail"), _spec(), md.GateConfig(min_recovered_r=0.5))
    m = graded["metrics"]
    # only the linear baseline survives; manifold/pullback fail energy/recovered_r gates
    assert m["per_arm"]["manifold"]["n_kept"] == 0
    assert m["per_arm"]["pullback"]["n_kept"] == 0
    assert m["per_arm"]["linear"]["n_kept"] == 5
    counts = m["reject_reason_counts"]
    assert counts.get("energy_above_linear", 0) >= 5
    assert counts.get("recovered_r_below_threshold", 0) == 5
    assert counts.get("collapsed", 0) == 1
    assert m["n_sft"] == 0 and m["n_preference"] == 0 and m["equal_size_n_per_arm"] == 0


def test_rejected_records_retained_with_reasons():
    graded = md.compile_payload(md.build_synthetic_payload("fail"), _spec(), md.GateConfig(min_recovered_r=0.5))
    assert len(graded["all"]) == len(graded["kept"]) + len(graded["rejected"])
    for r in graded["rejected"]:
        assert r["keep"] is False and r["reject_reasons"]
    for r in graded["kept"]:
        assert r["keep"] is True and r["reject_reasons"] == []


def test_sft_and_preference_record_shape():
    graded = md.compile_payload(md.build_synthetic_payload("win"), _spec())
    sft = md.to_sft_records(graded["kept"])
    assert sft and all(s["messages"][0]["role"] == "user" and s["messages"][1]["role"] == "assistant" for s in sft)
    assert all(s["messages"][1]["content"] for s in sft)
    assert all("provenance" in s and s["provenance"]["path"] in md.ON_MANIFOLD_PATHS for s in sft)

    pref = md.to_preference_records(graded)
    assert pref and all({"prompt", "chosen", "rejected", "provenance"} <= set(p) for p in pref)
    assert all(p["chosen"] != p["rejected"] for p in pref)


def test_report_and_card_carry_caveats():
    graded = md.compile_payload(md.build_synthetic_payload("win"), _spec())
    card = md.render_dataset_card(_spec(), graded["metrics"], md.GateConfig())
    report = md.render_report(_spec(), graded)
    for text in (card, report):
        assert "2604.28119" in text and "2605.05115" in text  # Goodfire attribution / source-gap
        assert "schema/export only" in text
    assert "falsification gate" in report.lower() or "falsification" in report.lower()


def test_write_outputs_roundtrip(tmp_path):
    spec = _spec()
    gate = md.GateConfig(min_recovered_r=0.5)
    graded = md.compile_payload(md.build_synthetic_payload("win"), spec, gate)
    paths = md.write_outputs(tmp_path, spec, graded, gate)
    for key in ("pairs_all", "pairs_kept", "pairs_rejected", "sft", "preference", "metrics", "dataset_card", "report"):
        assert key in paths
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["n_records"] == 15
    n_all = sum(1 for _ in (tmp_path / "pairs_all.jsonl").open())
    assert n_all == 15
    n_sft = sum(1 for _ in (tmp_path / "sft.jsonl").open())
    assert n_sft == metrics["n_sft"]
