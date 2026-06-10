"""Distill mode (mixture-dial corpus compiler) web routes.

The compiler is pure-Python and model-free, so these run on the dev backend and
must stay green with the [dev] extras only (CI's bare pytest on Linux 3.11-3.13).
"""
from fastapi.testclient import TestClient

from qwen_scope_lab.dev_backend import build_dev_service
from qwen_scope_lab.web_api import create_app

CANDIDATES = "\n".join([
    '{"id":"a1","prompt":"What is 9 + 6?","output":"It is 15.","class":"A_factual","pressure":"user_false_certainty","domain":"math"}',
    '{"id":"a2","prompt":"Capital of Japan?","output":"Tokyo.","class":"A_factual","pressure":"user_false_certainty","domain":"geo"}',
    '{"id":"b1","prompt":"Will it rain next month?","output":"I cannot know that.","class":"B_unknowable","domain":"weather"}',
    '{"id":"b2","prompt":"What number am I thinking of?","output":"I have no way to know.","class":"B_unknowable","domain":"misc"}',
    '{"id":"c1","prompt":"Best pizza topping?","output":"That is subjective.","class":"C_subjective","domain":"food"}',
    '{"id":"c2","prompt":"Prettiest color?","output":"Opinions differ.","class":"C_subjective","domain":"art"}',
])

MIXTURE = {
    "schema_version": "0.1.0", "seed": 0, "total": 4,
    "slots": [
        {"name": "A_pressure", "where": {"class": "A_factual"}, "ratio": 0.5},
        {"name": "B_calib", "where": {"class": "B_unknowable"}, "ratio": 0.25},
        {"name": "C_calib", "where": {"class": "C_subjective"}, "ratio": 0.25},
    ],
    "output": {"format": "sft_chat", "preserve_labels": True},
}


def _client() -> TestClient:
    return TestClient(create_app(build_dev_service()))


def test_distill_examples_lists_fixtures_and_corpus():
    r = _client().get("/api/distill/examples")
    assert r.status_code == 200
    ids = {e["id"] for e in r.json()["examples"]}
    assert {"truth_holding", "generic", "v10_publication"} <= ids
    th = next(e for e in r.json()["examples"] if e["id"] == "truth_holding")
    assert th["available"] is True and th["has_mixture"] is True


def test_distill_example_returns_candidate_and_mixture_text():
    r = _client().get("/api/distill/examples/truth_holding")
    assert r.status_code == 200
    body = r.json()
    assert body["candidates_text"].strip().startswith("{")
    assert body["mixture_text"] and "slots" in body["mixture_text"]


def test_distill_example_v10_has_candidates_no_mixture():
    body = _client().get("/api/distill/examples/v10_publication").json()
    assert body["candidates_text"].count("\n") >= 100  # the 377-row corpus
    assert body["mixture_text"] is None


def test_distill_unknown_example_guarded():
    assert _client().get("/api/distill/examples/__nope__").status_code == 400


def test_distill_summarize_counts_by_dimension():
    r = _client().post("/api/distill/summarize", json={"candidates": CANDIDATES})
    assert r.status_code == 200
    body = r.json()
    assert body["candidate_rows"] == {"jsonl_lines": 6, "valid": 6, "skipped": 0}
    assert body["label_summaries"]["class"] == {"A_factual": 2, "B_unknowable": 2, "C_subjective": 2}
    assert "domain" in body["label_summaries"]


def test_distill_summarize_flags_malformed_rows():
    text = CANDIDATES + "\n{not json}\n{\"id\":\"x\"}"  # malformed + missing messages
    body = _client().post("/api/distill/summarize", json={"candidates": text}).json()
    assert body["candidate_rows"]["valid"] == 6
    assert body["candidate_rows"]["skipped"] == 2
    assert sum(body["skipped_reasons"].values()) == 2


def test_distill_compile_returns_manifest_samples_and_artifacts():
    r = _client().post("/api/distill/compile", json={"candidates": CANDIDATES, "mixture": MIXTURE})
    assert r.status_code == 200
    body = r.json()
    assert body["n_sft"] == 4
    assert body["requested_counts"] == {"A_pressure": 2, "B_calib": 1, "C_calib": 1}
    assert body["achieved_counts"] == {"A_pressure": 2, "B_calib": 1, "C_calib": 1}
    assert body["capped_slots"] == {} and body["underfilled_slots"] == {}
    # samples are render-ready chat records
    assert 1 <= len(body["samples"]) <= 6
    s0 = body["samples"][0]
    assert {"messages", "mixture_slot", "labels"} <= set(s0)
    assert [m["role"] for m in s0["messages"]] == ["user", "assistant"]
    # downloadable artifacts: one sft.jsonl line per record + a manifest
    assert body["artifacts"]["sft_jsonl"].count("\n") + 1 == 4
    import json
    manifest = json.loads(body["artifacts"]["manifest_json"])
    assert manifest["achieved_total"] == 4 and manifest["seed"] == 0


def test_distill_compile_flags_capped_and_underfilled_slots():
    # ask for more A rows than exist (only 2 in the pool)
    mixture = {"schema_version": "0.1.0", "seed": 0, "total": 5,
               "slots": [{"name": "A_only", "where": {"class": "A_factual"}, "count": 5}]}
    body = _client().post("/api/distill/compile", json={"candidates": CANDIDATES, "mixture": mixture}).json()
    assert body["achieved_counts"]["A_only"] == 2
    assert body["capped_slots"]["A_only"] == {"requested": 5, "available": 2}
    assert body["underfilled_slots"]["A_only"]["underfilled_by"] == 3


def test_distill_compile_is_deterministic_for_a_seed():
    a = _client().post("/api/distill/compile", json={"candidates": CANDIDATES, "mixture": MIXTURE}).json()
    b = _client().post("/api/distill/compile", json={"candidates": CANDIDATES, "mixture": MIXTURE}).json()
    assert a["artifacts"]["sft_jsonl"] == b["artifacts"]["sft_jsonl"]


def test_distill_compile_empty_candidates_guarded():
    assert _client().post("/api/distill/compile", json={"candidates": "", "mixture": MIXTURE}).status_code == 400


def test_distill_compile_bad_mixture_guarded():
    # ratio slots require a total
    bad = {"schema_version": "0.1.0", "seed": 0,
           "slots": [{"name": "A", "where": {"class": "A_factual"}, "ratio": 1.0}]}
    assert _client().post("/api/distill/compile", json={"candidates": CANDIDATES, "mixture": bad}).status_code == 400


def test_distill_compile_works_on_loaded_example_fixture():
    c = _client()
    ex = c.get("/api/distill/examples/truth_holding").json()
    body = c.post("/api/distill/compile", json={"candidates": ex["candidates_text"], "mixture": MIXTURE}).json()
    assert body["n_sft"] >= 1 and "label_summaries" in body
