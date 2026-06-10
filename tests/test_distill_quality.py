"""Tests for the v0.2 warm-but-useful quality layer. All CI-safe: no GPU, no model, no network."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from qwen_scope_lab.experiments import distill_quality as dq

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "steering_to_data_distill.py"


# --------------------------------------------------------------------------------------
# Quality metrics
# --------------------------------------------------------------------------------------


def test_relevance_rewards_on_topic():
    prompt = "Describe the office coffee machine."
    on = dq.relevance_score(prompt, "The coffee machine brews espresso in the office.")
    off = dq.relevance_score(prompt, "It is a wonderful opportunity to share this with you.")
    assert on > off
    assert off == 0.0  # generic positivity echoes none of the task terms
    # instruction verbs are not counted as task terms
    assert "describe" not in dq.task_terms(prompt)


def test_repetition_and_stock_phrase():
    assert dq.repetition_score("the the the the the the the the") > 0.3
    assert dq.repetition_score("A perfectly ordinary and varied sentence about cats.") == 0.0
    assert dq.has_repeated_stock_phrase("I look forward to the opportunity. I look forward to the opportunity.") is True
    assert dq.has_repeated_stock_phrase("I look forward to seeing the results.") is False


def test_genericness_and_stock_share():
    generic = "It is a wonderful opportunity to share! I look forward to the opportunity to share."
    substantive = "The standup covered the API migration, two blockers, and today's deploy plan."
    assert dq.genericness_score(generic) > dq.genericness_score(substantive)
    assert dq.stock_phrase_share(generic) > 0.2
    assert dq.stock_phrase_share(substantive) < 0.1


def test_unsupported_specifics():
    prompt = "Tell me about the meeting."
    assert dq.unsupported_specifics_score(prompt, "The meeting is at 3pm in Room 204 with 17 people.") > 0
    assert dq.unsupported_specifics_score(prompt, "The meeting is later today.") == 0.0


def test_think_tags_and_negative_context():
    assert dq.has_think_tags("<think>plan</think> answer") is True
    assert dq.has_think_tags("a normal answer") is False
    assert dq.is_negative_context("Write an incident report about the outage.") is True
    assert dq.is_negative_context("Describe the lunch menu.") is False
    # metadata overrides keyword detection
    assert dq.is_negative_context("Tell me about the project.", {"appropriate_tone": "somber"}) is True


def test_score_quality_shape():
    q = dq.score_quality("Describe the coffee machine.", "Our coffee machine cheerfully brews espresso!", grounding="The coffee machine makes espresso.")
    assert 0.0 <= q.sentiment <= 1.0 and 0.0 <= q.relevance <= 1.0
    assert q.has_think is False and q.negative_context is False
    assert q.content_overlap > 0.0


# --------------------------------------------------------------------------------------
# Warmth filter — every reject reason + a genuine keep
# --------------------------------------------------------------------------------------


def _pair(prompt, steered, unsteered="", metadata=None):
    return {"prompt": prompt, "steered": steered, "unsteered": unsteered, "metadata": metadata or {}}


def test_warmth_filter_keeps_genuine_warm_useful():
    res = dq.filter_warmth(_pair(
        "Tell me about the team standup.",
        "Great news from the standup! We happily cleared the blockers and the API migration is on track.",
        "The standup covered blockers and the API migration.",
    ))
    assert res.keep is True and res.reasons == []


@pytest.mark.parametrize("pair,reason", [
    (_pair("Tell me about the bus ride.", "<think>plan</think> What a wonderful ride!", "The bus ride was fine."), "contains_think_tags"),
    (_pair("Explain how to reset a password.", "It is a wonderful opportunity to share! I am delighted to share this with you.", "Click forgot password."), "low_relevance"),
    (_pair("Describe lunch options.", "I look forward to the opportunity to share. I look forward to the opportunity to share!", "Sandwiches and salad."), "repetitive"),
    (_pair("Write an incident report about the outage.", "What a wonderful and exciting opportunity! I am absolutely delighted!", "The database was down for two hours.", {"appropriate_tone": "serious"}), "inappropriate_positivity"),
    (_pair("Give a status update on the report.", "The report is drafted and in review.", "The report is drafted and in review."), "not_warmer"),
])
def test_warmth_filter_reject_reasons(pair, reason):
    res = dq.filter_warmth(pair)
    assert res.keep is False
    assert reason in res.reasons


# --------------------------------------------------------------------------------------
# Phrase concentration
# --------------------------------------------------------------------------------------


def test_phrase_concentration_warns_above_threshold():
    outputs = [
        "It is a wonderful opportunity to share.",
        "What a wonderful day, a wonderful opportunity!",
        "I look forward to the wonderful opportunity.",
        "The report is in review.",
    ]
    pc = dq.phrase_concentration(outputs, warn_fraction=0.2)
    assert pc["stock_phrase_fraction"]["wonderful"] >= 0.5
    assert any("wonderful" in w for w in pc["warnings"])
    assert pc["top_bigrams"]  # recurring bigrams surfaced
    assert any(p == "wonderful opportunity" for p, _ in pc["top_bigrams"])


# --------------------------------------------------------------------------------------
# Dataset audit
# --------------------------------------------------------------------------------------


def test_audit_dataset_synthetic_covers_reasons():
    audit = dq.audit_dataset(dq.build_synthetic_warmth_pairs())
    m = audit["metrics"]
    assert m["n_pairs"] == 8 and m["n_kept_v2"] == 2
    # the crafted set exercises the major reject reasons
    assert {"contains_think_tags", "low_relevance", "generic_positivity", "repetitive", "inappropriate_positivity", "hallucinated_specifics"} <= set(m["reject_reason_counts"])
    assert audit["phrase_concentration"]["warnings"]
    # kept pairs are more relevant + less generic than the full set
    assert m["kept_pairs"]["relevance"] >= m["all_pairs"]["relevance"]


# --------------------------------------------------------------------------------------
# Quality eval of arms — not-run handling + warm-but-useful verdict
# --------------------------------------------------------------------------------------


def test_quality_eval_flags_gamed_and_reports_not_run():
    ev = dq.evaluate_quality_arms(dq.build_synthetic_quality_arms())
    assert ev["verdict"]["status"] == "warm_but_gamed"
    assert ev["verdict"]["lexicon_tone_improved"] is True
    assert ev["verdict"]["useful_warmth_improved"] is False
    # absent arms are reported, not omitted
    assert ev["arms"]["runtime_steer_2b"]["status"] == "not run"
    assert ev["arms"]["prompt_only_2b"]["status"] == "not run"
    assert set(dq.CANONICAL_ARMS) <= set(ev["arms"])


def test_quality_eval_warm_and_useful_when_relevance_preserved():
    arms = {
        "baseline_4b": [{"prompt": "Describe the coffee machine.", "output": "The coffee machine makes espresso and drip coffee.", "metadata": {}}],
        "distilled_4b_from_steered_data": [{"prompt": "Describe the coffee machine.", "output": "Our lovely coffee machine happily brews espresso and drip coffee — a delightful perk!", "metadata": {}}],
    }
    ev = dq.evaluate_quality_arms(arms)
    assert ev["verdict"]["status"] == "warm_and_useful"
    assert ev["verdict"]["useful_warmth_improved"] is True


def test_quality_eval_incomplete_without_both_arms():
    ev = dq.evaluate_quality_arms({"baseline_4b": [{"prompt": "x", "output": "y", "metadata": {}}]})
    assert ev["verdict"]["status"] == "incomplete"


def test_optional_command_judge_hook():
    # a deterministic LOCAL judge (no network): returns word count / 10
    judge = dq.make_command_judge(f'{sys.executable} -c "import sys;print(len(sys.stdin.read().split())/10)"')
    arms = {
        "baseline_4b": [{"prompt": "Describe lunch.", "output": "Lunch is sandwiches.", "metadata": {}}],
        "distilled_4b_from_steered_data": [{"prompt": "Describe lunch.", "output": "Lunch is a warm and tasty sandwich.", "metadata": {}}],
    }
    ev = dq.evaluate_quality_arms(arms, judge=judge)
    assert "judge" in ev["arms"]["baseline_4b"] and ev["arms"]["baseline_4b"]["judge"] > 0
    # without a judge, no judge key
    ev2 = dq.evaluate_quality_arms(arms)
    assert "judge" not in ev2["arms"]["baseline_4b"]


def test_alias_maps_v1_distilled_name():
    arms = {
        "baseline_4b": [{"prompt": "Describe lunch.", "output": "Lunch is sandwiches and salad.", "metadata": {}}],
        "distilled_4b": [{"prompt": "Describe lunch.", "output": "Wonderful! I am delighted to share the lunch of sandwiches and salad!", "metadata": {}}],
    }
    ev = dq.evaluate_quality_arms(arms)
    assert ev["arms"]["distilled_4b_from_steered_data"]["status"] == "run"


# --------------------------------------------------------------------------------------
# Renderers don't crash
# --------------------------------------------------------------------------------------


def test_render_functions():
    audit = dq.audit_dataset(dq.build_synthetic_warmth_pairs())
    md = dq.render_dataset_audit(audit, title="synthetic")
    assert "warmth audit" in md and "Phrase concentration" in md
    ev = dq.evaluate_quality_arms(dq.build_synthetic_quality_arms())
    em = dq.render_quality_eval(ev)
    assert "warm_but_gamed" in em and "not run" in em


# --------------------------------------------------------------------------------------
# CLI audit subcommand (subprocess — no model)
# --------------------------------------------------------------------------------------


def test_cli_audit_synthetic(tmp_path):
    out = tmp_path / "audit"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "audit", "--synthetic", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["dataset"]["n_kept_v2"] == 2
    assert payload["eval"]["verdict"] == "warm_but_gamed"
    for name in ("dataset_audit.md", "sft_v2.jsonl", "pairs_rejected_v2.jsonl", "phrase_concentration.json", "eval_quality.md"):
        assert (out / name).exists()
