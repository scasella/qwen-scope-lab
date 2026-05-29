from fastapi.testclient import TestClient

from qwen_scope_steering_gui.dev_backend import build_dev_service
from qwen_scope_steering_gui.web_api import create_app


def _client() -> TestClient:
    return TestClient(create_app(build_dev_service()))


def test_status_reports_dev_model():
    r = _client().get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["configured_model_id"].startswith("dev/")
    assert body["model_loaded"] is True


def test_inspect_returns_tokens_and_features():
    r = _client().post("/api/inspect", json={"prompt": "The capital of France is Paris", "layer": 3, "top_k": 6})
    assert r.status_code == 200
    body = r.json()
    assert body["tokens"][0] == "The"
    assert len(body["top_features_by_token"]) == len(body["tokens"])
    first = body["top_features_by_token"][0]["features"]
    assert first and all("feature_id" in f and "activation" in f for f in first)


def test_compare_returns_contrastive_features():
    r = _client().post("/api/compare", json={"positive": "concise answer", "negative": "long rambling story", "layer": 3, "limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert "positive_stronger" in body and "negative_stronger" in body


def test_steer_fires_hook_and_changes_output():
    r = _client().post("/api/steer", json={"prompt": "Write one sentence about Paris.", "layer": 3, "feature_id": 42, "strength": 10.0, "max_new_tokens": 12})
    assert r.status_code == 200
    body = r.json()
    assert body["hook_fired"] is True
    assert body["hidden_delta_norm"] > 0
    assert body["unsteered_text"] != body["steered_text"]


def test_sweep_returns_a_frame_per_strength():
    r = _client().post("/api/sweep", json={"prompt": "Write one sentence about Paris.", "layer": 3, "feature_id": 42, "strengths": [-5, 0, 5], "max_new_tokens": 8})
    assert r.status_code == 200
    assert [f["strength"] for f in r.json()["frames"]] == [-5.0, 0.0, 5.0]


def test_invalid_feature_id_is_guarded():
    r = _client().post("/api/steer", json={"prompt": "x", "layer": 3, "feature_id": 999999, "strength": 5.0})
    assert r.status_code == 400


def test_invalid_layer_is_guarded():
    r = _client().post("/api/inspect", json={"prompt": "x", "layer": 99})
    assert r.status_code == 400


def test_recipes_list_and_missing_detail():
    c = _client()
    assert c.get("/api/recipes").status_code == 200
    assert isinstance(c.get("/api/recipes").json(), list)
    assert c.get("/api/recipes/__does_not_exist__").status_code in (400, 404)


def test_save_notebook_entry():
    r = _client().post("/api/notebook", json={"feature_id": 42, "layer": 3, "human_label": "dev test label"})
    assert r.status_code == 200


def test_benchmark_returns_seven_methods_and_verdict():
    ps = '{"id":"p1","prompt":"Explain sparse autoencoders."}\n{"id":"p2","prompt":"Return a JSON object with key city and value Paris."}'
    r = _client().post("/api/benchmark", json={"prompt_set": ps, "feature_id": 42, "strength": 8.0, "layer": 3, "max_new_tokens": 8})
    assert r.status_code == 200
    body = r.json()
    assert len(body["methods"]) == 7
    assert set(body["method_scores"]) == set(body["methods"])
    decision = body["validation_decision"]
    assert "status" in decision and "reason" in decision and "passed" in decision
    assert len(body["examples"]) >= 1


def test_save_recipe_requires_prior_benchmark_then_persists(tmp_path):
    c = TestClient(create_app(build_dev_service(), recipes_root=str(tmp_path)))
    assert c.post("/api/recipes").status_code == 400  # nothing benchmarked yet
    ps = '{"id":"p1","prompt":"Explain sparse autoencoders."}'
    assert c.post("/api/benchmark", json={"prompt_set": ps, "feature_id": 42, "strength": 8.0, "layer": 3, "max_new_tokens": 6}).status_code == 200
    saved = c.post("/api/recipes")
    assert saved.status_code == 200
    body = saved.json()
    assert "recipe_id" in body and "status" in body
    assert any(row["recipe_id"] == body["recipe_id"] for row in c.get("/api/recipes").json())


def test_autopilot_discovers_benchmarks_and_saves(tmp_path):
    c = TestClient(create_app(build_dev_service(), recipes_root=str(tmp_path)))
    r = c.post("/api/autopilot", json={
        "positive_examples": "Paris is the capital of France.\nThe capital is Paris.",
        "negative_examples": "Once upon a time a long rambling tale.\nA winding story with digressions.",
        "validation_prompts": '{"id":"v1","prompt":"What is the capital of France?"}',
        "candidate_count": 2, "candidate_layers": [3], "max_new_tokens": 6,
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["candidates"]) >= 1
    assert len(body["methods"]) == 7
    assert "feature_id" in body["best_candidate"]
    assert body["recipe_id"]
    assert any(row["recipe_id"] == body["recipe_id"] for row in c.get("/api/recipes").json())


def test_atlas_aggregates_features_across_prompts():
    c = _client()
    prompts = ["The capital of France is Paris.", "Return a JSON object with name and age.", "The museum was full of art and golden light."]
    r = c.post("/api/atlas", json={"prompts": prompts, "layer": 3, "top_k": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["n_prompts"] == 3
    assert body["features"], "expected at least one mapped feature"
    f0 = body["features"][0]
    assert set(["feature_id", "peak", "n_prompts", "fingerprint", "top_tokens"]) <= set(f0)
    assert len(f0["fingerprint"]) == 3
    assert 1 <= f0["n_prompts"] <= 3
    # peak-sorted
    peaks = [f["peak"] for f in body["features"]]
    assert peaks == sorted(peaks, reverse=True)


def test_atlas_requires_prompts():
    assert _client().post("/api/atlas", json={"prompts": []}).status_code == 400


def test_manifold_presets_lists_concepts():
    r = _client().get("/api/manifold/presets")
    assert r.status_code == 200
    names = {p["name"] for p in r.json()["presets"]}
    assert {"days_of_week", "integers_0_20"} <= names


def test_manifold_fit_returns_3d_geometry():
    r = _client().post("/api/manifold/fit", json={"concept": "days_of_week", "layer": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "cyclic" and body["n_items"] == 7
    assert len(body["points_3d"]) == 7 and all(len(p["xyz"]) == 3 for p in body["points_3d"])
    assert len(body["curve_3d"]) > 10
    assert body["synthetic"] is True  # dev backend uses the synthetic ring


def test_manifold_steer_returns_waypoints_and_fires_hook():
    r = _client().post("/api/manifold/steer", json={
        "concept": "days_of_week", "source": "Monday", "target": "Friday",
        "layer": 3, "n_waypoints": 3, "max_new_tokens": 6})
    assert r.status_code == 200
    body = r.json()
    assert len(body["waypoints"]) == 3
    assert body["waypoints"][-1]["value"] == "Friday"
    assert body["hook_fired"] is True
    assert isinstance(body["position"], int) and body["position"] >= 0
    assert len(body["path_3d"]) == 3 and len(body["path_3d"][0]) == 3


def test_manifold_unknown_concept_guarded():
    assert _client().post("/api/manifold/fit", json={"concept": "__nope__"}).status_code == 400


def test_manifold_compare_returns_both_paths_with_perplexity():
    r = _client().post("/api/manifold/compare", json={
        "concept": "days_of_week", "source": "Monday", "target": "Thursday",
        "layer": 3, "n_waypoints": 3, "max_new_tokens": 6})
    assert r.status_code == 200
    body = r.json()
    assert set(["manifold", "linear", "unsteered_text"]) <= set(body)
    for k in ("manifold", "linear"):
        leg = body[k]
        assert leg["path"] == k
        assert len(leg["path_3d"]) == 3 and len(leg["waypoints"]) == 3
        assert leg["perplexity"] is None or leg["perplexity"] > 0
        assert leg["mean_energy"] is None or leg["mean_energy"] >= 0  # behavior-manifold distance
        assert all("energy" in w for w in leg["waypoints"])


def test_manifold_pullback_returns_three_legs():
    r = _client().post("/api/manifold/pullback", json={
        "concept": "days_of_week", "source": "Monday", "target": "Thursday",
        "layer": 3, "n_waypoints": 3, "max_new_tokens": 6, "lbfgs_iters": 3})
    assert r.status_code == 200
    body = r.json()
    assert set(["manifold", "linear", "pullback"]) <= set(body)
    pb = body["pullback"]
    assert pb["path"] == "pullback" and len(pb["path_3d"]) == 3 and len(pb["waypoints"]) == 3
    assert pb["mean_energy"] is None or pb["mean_energy"] >= 0
    assert "recovered_r" in body["manifold"] and "recovered_r" in pb


def test_manifold_recipe_save_list_detail_roundtrip(tmp_path):
    c = TestClient(create_app(build_dev_service(), recipes_root=str(tmp_path)))
    # saving a manifold recipe requires a prior pullback (the manifold benchmark)
    assert c.post("/api/recipes", json={"kind": "manifold"}).status_code == 400
    pb = c.post("/api/manifold/pullback", json={
        "concept": "days_of_week", "source": "Monday", "target": "Thursday",
        "layer": 3, "n_waypoints": 3, "max_new_tokens": 6, "lbfgs_iters": 3})
    assert pb.status_code == 200
    saved = c.post("/api/recipes", json={"kind": "manifold"})
    assert saved.status_code == 200
    rid = saved.json()["recipe_id"]
    assert saved.json()["status"] in {"validated", "benchmarked", "candidate", "draft"}
    row = next(r for r in c.get("/api/recipes").json() if r["recipe_id"] == rid)
    assert row["kind"] == "manifold"
    assert row["concept"] == "days_of_week" and row["source"] == "Monday" and row["target"] == "Thursday"
    detail = c.get(f"/api/recipes/{rid}").json()
    assert detail["kind"] == "manifold" and detail["interventions"] == []
    assert detail["manifold"]["concept"] == "days_of_week" and detail["manifold"]["layer"] == 3
    assert set(detail["benchmark"].get("legs", {})) & {"manifold", "linear", "pullback"}


def test_feature_recipe_save_still_works_without_kind(tmp_path):
    c = TestClient(create_app(build_dev_service(), recipes_root=str(tmp_path)))
    ps = '{"id":"p1","prompt":"Explain sparse autoencoders."}'
    assert c.post("/api/benchmark", json={"prompt_set": ps, "feature_id": 42, "strength": 8.0, "layer": 3, "max_new_tokens": 6}).status_code == 200
    saved = c.post("/api/recipes")  # no body → feature path
    assert saved.status_code == 200
    row = next(r for r in c.get("/api/recipes").json() if r["recipe_id"] == saved.json()["recipe_id"])
    assert row.get("kind", "feature") == "feature" and row["feature_id"] == 42


def test_manifold_sae_coverage_returns_tiling():
    r = _client().post("/api/manifold/sae_coverage", json={"concept": "days_of_week", "layer": 3, "top_k": 4})
    assert r.status_code == 200
    body = r.json()
    assert len(body["per_value"]) == 7
    f0 = body["per_value"][0]
    assert "dominant_feature" in f0 and 1 <= len(f0["features"]) <= 4
    assert body["n_distinct_features"] >= 1 and isinstance(body["tiling"], list)


def test_autopilot_requires_examples():
    r = _client().post("/api/autopilot", json={
        "positive_examples": "", "negative_examples": "neg", "validation_prompts": '{"prompt":"x"}'})
    assert r.status_code == 400


# ---- agent-research layer: async job API + experiment log ----

def _wait_job(c, job_id, timeout=20.0):
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = c.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    return c.get(f"/api/jobs/{job_id}").json()


def test_job_submit_poll_and_experiment_log(tmp_path):
    with TestClient(create_app(build_dev_service(), recipes_root=str(tmp_path),
                               experiments_root=str(tmp_path / "exp"))) as c:
        r = c.post("/api/jobs", json={"op": "benchmark", "params": {
            "prompt_set": '{"id":"p1","prompt":"Explain sparse autoencoders."}',
            "feature_id": 42, "strength": 8.0, "layer": 3, "max_new_tokens": 6}})
        assert r.status_code == 200
        job = _wait_job(c, r.json()["job_id"])
        assert job["status"] == "done", job
        assert len(job["result"]["methods"]) == 7 and "method_scores" in job["result"]
        exps = c.get("/api/experiments").json()
        assert any(e["op"] == "benchmark" and e["status"] == "done" for e in exps)
        assert exps[-1]["summary"].get("validation_decision") is not None


def test_jobs_unknown_op_rejected():
    assert _client().post("/api/jobs", json={"op": "frobnicate", "params": {}}).status_code == 400


def test_concurrent_jobs_both_complete_no_500(tmp_path):
    with TestClient(create_app(build_dev_service(), recipes_root=str(tmp_path))) as c:
        ids = []
        for fid in (10, 20):
            r = c.post("/api/jobs", json={"op": "steer", "params": {
                "prompt": "Write about Paris.", "layer": 3, "feature_id": fid, "strength": 5.0, "max_new_tokens": 4}})
            assert r.status_code == 200
            ids.append(r.json()["job_id"])
        for jid in ids:
            job = _wait_job(c, jid)
            assert job["status"] == "done", job
            assert job["result"]["hook_fired"] is True
        assert len(c.get("/api/jobs").json()) >= 2


def test_reads_stay_available_and_no_log_when_disabled(tmp_path):
    # experiments_root omitted -> logging is a no-op, /api/experiments is empty
    with TestClient(create_app(build_dev_service(), recipes_root=str(tmp_path))) as c:
        assert c.get("/api/status").status_code == 200
        job = _wait_job(c, c.post("/api/jobs", json={"op": "inspect", "params": {"prompt": "x", "layer": 3}}).json()["job_id"])
        assert job["status"] == "done"
        assert c.get("/api/experiments").json() == []


# ---- feature-as-monitor: discover -> evaluate -> score -> save ----

def _mon_client(tmp_path):
    return TestClient(create_app(build_dev_service(), recipes_root=str(tmp_path / "r"), monitors_root=str(tmp_path / "m")))


def test_monitor_discover_score_save_roundtrip(tmp_path):
    c = _mon_client(tmp_path)
    pos = "I refuse to do that.\nI can't help with this.\nNo, I won't assist.\nSorry, I cannot."
    neg = "Sure, here's how.\nAbsolutely, let me help.\nYes, of course.\nHappy to assist."
    r = c.post("/api/monitor/discover", json={"behavior": "refusal", "positive_examples": pos,
                                              "negative_examples": neg, "layer": 3, "top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["features"] and {"auc", "control_auc", "f1"} <= set(body["metrics"])
    assert body["validation_decision"]["status"] in {"validated", "benchmarked"}
    # score new text against the discovered features
    sc = c.post("/api/monitor/score", json={"text": "I cannot help with that.", "features": body["features"],
                                            "layer": 3, "threshold": body["threshold"]})
    assert sc.status_code == 200 and {"fires", "score"} <= set(sc.json())
    # save the discovered monitor -> list -> detail
    saved = c.post("/api/monitors")
    assert saved.status_code == 200
    mid = saved.json()["monitor_id"]
    assert any(row["monitor_id"] == mid for row in c.get("/api/monitors").json())
    detail = c.get(f"/api/monitors/{mid}").json()
    assert detail["features"] == body["features"] and detail["layer"] == 3


def test_monitor_save_requires_discover_first(tmp_path):
    assert _mon_client(tmp_path).post("/api/monitors").status_code == 400


def test_monitor_score_by_saved_id(tmp_path):
    c = _mon_client(tmp_path)
    c.post("/api/monitor/discover", json={"behavior": "refusal",
        "positive_examples": "No.\nI can't.\nI refuse.\nSorry, no.",
        "negative_examples": "Sure.\nYes.\nOf course.\nHappy to.", "layer": 3, "top_k": 2})
    mid = c.post("/api/monitors").json()["monitor_id"]
    sc = c.post("/api/monitor/score", json={"text": "I won't do that.", "monitor_id": mid})
    assert sc.status_code == 200 and "fires" in sc.json()


def test_monitor_discover_as_job(tmp_path):
    with _mon_client(tmp_path) as c:
        r = c.post("/api/jobs", json={"op": "monitor_discover", "params": {"behavior": "refusal",
            "positive_examples": "No.\nI can't.\nI refuse.\nSorry, no.",
            "negative_examples": "Sure.\nYes.\nOf course.\nHappy to.", "layer": 3, "top_k": 2}})
        job = _wait_job(c, r.json()["job_id"])
        assert job["status"] == "done"
        assert "features" in job["result"] and "validation_decision" in job["result"]
