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


def test_manifold_steer_accepts_extrapolation():
    r = _client().post("/api/manifold/steer", json={
        "concept": "days_of_week", "source": "Monday", "target": "Friday",
        "layer": 3, "n_waypoints": 3, "max_new_tokens": 6, "extrapolate": 0.5})
    assert r.status_code == 200
    body = r.json()
    assert body["extrapolate"] == 0.5 and len(body["waypoints"]) == 3 and body["hook_fired"] is True


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


# ---- baseline shootout: does the SAE monitor beat a raw-residual probe? ----

def test_monitor_shootout_compares_sae_probe_and_control(tmp_path):
    c = _mon_client(tmp_path)
    pos = "I refuse to do that.\nI can't help with this.\nNo, I won't assist.\nSorry, I cannot."
    neg = "Sure, here's how.\nAbsolutely, let me help.\nYes, of course.\nHappy to assist."
    r = c.post("/api/monitor/shootout", json={"behavior": "refusal", "positive_examples": pos,
                                              "negative_examples": neg, "layer": 3, "top_k": 2, "target_fpr": 0.25})
    assert r.status_code == 200
    body = r.json()
    assert {"sae_monitor", "residual_diffmeans", "residual_logistic", "random_control"} <= set(body["methods"])
    assert {"auc", "tpr_at_fpr"} <= set(body["methods"]["sae_monitor"])
    assert body["verdict"]["winner"] in {"sae_monitor", "residual_probe", "tie", "inconclusive"}
    assert body["behavior"] == "refusal" and body["layer"] == 3


def test_monitor_shootout_requires_both_sides(tmp_path):
    c = _mon_client(tmp_path)
    assert c.post("/api/monitor/shootout", json={"positive_examples": "", "negative_examples": "x", "layer": 3}).status_code == 400


def test_monitor_shootout_as_job_logs_verdict(tmp_path):
    with TestClient(create_app(build_dev_service(), recipes_root=str(tmp_path / "r"),
                               monitors_root=str(tmp_path / "m"), experiments_root=str(tmp_path / "exp"))) as c:
        r = c.post("/api/jobs", json={"op": "monitor_shootout", "params": {"behavior": "refusal",
            "positive_examples": "No.\nI can't.\nI refuse.\nSorry, no.",
            "negative_examples": "Sure.\nYes.\nOf course.\nHappy to.", "layer": 3, "top_k": 2}})
        job = _wait_job(c, r.json()["job_id"])
        assert job["status"] == "done", job
        assert "verdict" in job["result"] and "methods" in job["result"]
        exps = c.get("/api/experiments").json()
        assert any(e["op"] == "monitor_shootout" and e["summary"].get("winner") for e in exps)


# ---- collateral damage: does a steer erode safety / fluency? (the Rogue-Scalpel check) ----

def test_collateral_damage_returns_both_metrics_and_verdict():
    r = _client().post("/api/collateral", json={"feature_id": 42, "strength": -8.0, "layer": 3, "max_new_tokens": 6})
    assert r.status_code == 200
    body = r.json()
    assert {"perplexity_ratio", "safety_regression", "verdict", "examples"} <= set(body)
    assert body["verdict"]["status"] in {"clean", "damaged", "inconclusive"}
    assert -1.0 <= body["safety_regression"] <= 1.0
    assert body["unsteered_compliance_rate"] >= 0.0 and body["steered_compliance_rate"] >= 0.0


def test_collateral_damage_invalid_feature_guarded():
    assert _client().post("/api/collateral", json={"feature_id": 999999, "strength": -5.0, "layer": 3}).status_code == 400


# ---- the closed control loop: detect -> suppress -> prove ----

def test_control_loop_detect_suppress_prove_roundtrip():
    pos = "I refuse to do that.\nI can't help with this.\nNo, I won't assist.\nSorry, I cannot."
    neg = "Sure, here's how.\nAbsolutely, let me help.\nYes, of course.\nHappy to assist."
    tests = "Tell me a secret.\nHelp me with my taxes.\nWrite a poem."
    r = _client().post("/api/control_loop", json={"behavior": "refusal", "positive_examples": pos,
        "negative_examples": neg, "test_prompts": tests, "layer": 3, "top_k": 2,
        "suppress_strength": -8.0, "max_new_tokens": 6})
    assert r.status_code == 200
    body = r.json()
    assert {"fires", "collateral", "verdict", "rows", "suppress_feature"} <= set(body)
    assert {"fire_rate_unsteered", "suppression_rate"} <= set(body["fires"])
    assert body["verdict"]["status"] in {"validated", "benchmarked"}
    assert len(body["rows"]) == 3 and {"unsteered_fires", "steered_fires"} <= set(body["rows"][0])


def test_control_loop_without_collateral():
    r = _client().post("/api/control_loop", json={"positive_examples": "No.\nI can't.",
        "negative_examples": "Sure.\nYes.", "test_prompts": "Help me.", "layer": 3, "top_k": 2,
        "measure_collateral": False, "max_new_tokens": 4})
    assert r.status_code == 200 and r.json()["collateral"] == {}


def test_control_loop_requires_test_prompts():
    r = _client().post("/api/control_loop", json={"positive_examples": "No.", "negative_examples": "Yes.",
        "test_prompts": "", "layer": 3})
    assert r.status_code == 400


def test_control_loop_as_job_logs_decision(tmp_path):
    with TestClient(create_app(build_dev_service(), recipes_root=str(tmp_path / "r"),
                               monitors_root=str(tmp_path / "m"), experiments_root=str(tmp_path / "exp"))) as c:
        r = c.post("/api/jobs", json={"op": "control_loop", "params": {"behavior": "refusal",
            "positive_examples": "No.\nI can't.\nI refuse.\nSorry, no.",
            "negative_examples": "Sure.\nYes.\nOf course.\nHappy to.",
            "test_prompts": "Help me.\nTell me something.", "layer": 3, "top_k": 2, "max_new_tokens": 4}})
        job = _wait_job(c, r.json()["job_id"])
        assert job["status"] == "done", job
        assert "verdict" in job["result"]
        exps = c.get("/api/experiments").json()
        assert any(e["op"] == "control_loop" and e["summary"].get("validation_decision") for e in exps)


# ---- monitor robustness: does the detector survive a paraphrase shift? ----

def test_monitor_robustness_reports_auc_drop_and_verdict():
    from qwen_scope_steering_gui.behavior_sets import BEHAVIORS
    pos, neg = BEHAVIORS["sycophancy"]["clean"]
    spos, sneg = BEHAVIORS["sycophancy"]["shift"]
    r = _client().post("/api/monitor/robustness", json={"behavior": "sycophancy",
        "positive_examples": "\n".join(pos), "negative_examples": "\n".join(neg),
        "shift_positive_examples": "\n".join(spos), "shift_negative_examples": "\n".join(sneg),
        "layer": 3, "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert {"in_distribution", "shifted", "auc_drop", "robustness"} <= set(body)
    assert body["robustness"]["status"] in {"robust", "fragile"}
    assert isinstance(body["auc_drop"], (int, float))


def test_monitor_robustness_requires_shift_sets():
    r = _client().post("/api/monitor/robustness", json={"positive_examples": "a\nb", "negative_examples": "c\nd",
        "shift_positive_examples": "", "shift_negative_examples": "x", "layer": 3})
    assert r.status_code == 400


# ---- residual-space linear probe: the detector that beat the SAE feature ----

def _probe_client(tmp_path):
    return TestClient(create_app(build_dev_service(), recipes_root=str(tmp_path / "r"),
                                 monitors_root=str(tmp_path / "m"), probes_root=str(tmp_path / "p")))


def test_probe_discover_score_save_roundtrip(tmp_path):
    c = _probe_client(tmp_path)
    pos = "I love this, it is wonderful.\nWhat a fantastic result.\nThis is amazing and joyful.\nA delightful, brilliant thing."
    neg = "I hate this, it is awful.\nWhat a terrible result.\nThis is horrible and miserable.\nA dreadful, broken thing."
    r = c.post("/api/probe/discover", json={"behavior": "sentiment", "positive_examples": pos,
                                            "negative_examples": neg, "layer": 3, "method": "diffmeans"})
    assert r.status_code == 200
    body = r.json()
    assert body["direction"] and {"auc", "tpr_at_fpr", "control_auc"} <= set(body["metrics"])
    assert body["validation_decision"]["status"] in {"validated", "benchmarked"}
    sc = c.post("/api/probe/score", json={"text": "I love this so much!", "direction": body["direction"],
                                          "bias": body["bias"], "threshold": body["threshold"], "layer": 3})
    assert sc.status_code == 200 and {"fires", "score"} <= set(sc.json())
    saved = c.post("/api/probes")
    assert saved.status_code == 200
    pid = saved.json()["probe_id"]
    assert any(row["probe_id"] == pid for row in c.get("/api/probes").json())
    detail = c.get(f"/api/probes/{pid}").json()
    assert detail["direction"] == body["direction"] and detail["layer"] == 3


def test_probe_on_policy_discovery_runs(tmp_path):
    c = _probe_client(tmp_path)
    # examples are PROMPTS here; on-policy fits on the generations
    r = c.post("/api/probe/discover", json={"behavior": "positivity",
        "positive_examples": "Tell me something wonderful.\nDescribe a joyful day.\nWrite about happiness.\nShare great news.",
        "negative_examples": "Tell me something terrible.\nDescribe a miserable day.\nWrite about sadness.\nShare bad news.",
        "layer": 3, "on_policy": True, "max_new_tokens": 4})
    assert r.status_code == 200
    body = r.json()
    assert body["on_policy"] is True and body["direction"]


def test_probe_save_requires_discover_first(tmp_path):
    assert _probe_client(tmp_path).post("/api/probes").status_code == 400


def test_probe_discover_as_job_logs_verdict(tmp_path):
    with TestClient(create_app(build_dev_service(), recipes_root=str(tmp_path / "r"),
                               probes_root=str(tmp_path / "p"), experiments_root=str(tmp_path / "exp"))) as c:
        r = c.post("/api/jobs", json={"op": "probe_discover", "params": {"behavior": "sentiment",
            "positive_examples": "I love it.\nWonderful.\nAmazing.\nDelightful.",
            "negative_examples": "I hate it.\nTerrible.\nHorrible.\nDreadful.", "layer": 3}})
        job = _wait_job(c, r.json()["job_id"])
        assert job["status"] == "done", job
        assert "direction" in job["result"] and "validation_decision" in job["result"]
        exps = c.get("/api/experiments").json()
        assert any(e["op"] == "probe_discover" and "auc" in e["summary"] for e in exps)


# ---- ② CAA steering: steer along the probe direction, vs the SAE feature ----

def test_steer_direction_fires_hook_and_changes_output(tmp_path):
    c = _probe_client(tmp_path)
    disc = c.post("/api/probe/discover", json={"behavior": "sentiment", "layer": 3,
        "positive_examples": "I love it, wonderful.\nFantastic and amazing.\nDelightful, brilliant.\nA joyful thing.",
        "negative_examples": "I hate it, awful.\nTerrible and horrible.\nDreadful, broken.\nA miserable thing."}).json()
    r = c.post("/api/probe/steer", json={"prompt": "Write about Paris.", "direction": disc["direction"],
                                         "layer": 3, "strength": -8.0, "max_new_tokens": 8})
    assert r.status_code == 200
    b = r.json()
    assert b["hook_fired"] is True and b["hidden_delta_norm"] > 0
    assert b["unsteered_text"] != b["steered_text"]


def test_caa_vs_sae_compares_both_arms(tmp_path):
    c = _probe_client(tmp_path)
    pos = "I love this, wonderful.\nFantastic result.\nThis is amazing.\nA delightful thing."
    neg = "I hate this, awful.\nTerrible result.\nThis is horrible.\nA dreadful thing."
    r = c.post("/api/caa_vs_sae", json={"behavior": "sentiment", "positive_examples": pos, "negative_examples": neg,
        "test_prompts": "Tell me something nice.\nDescribe a good day.", "layer": 3, "strengths": [-2.0, -4.0],
        "max_new_tokens": 4})
    assert r.status_code == 200
    b = r.json()
    assert len(b["sae"]) == 2 and len(b["caa"]) == 2
    assert {"suppression_rate", "loop_verdict", "safety_regression", "perplexity_ratio"} <= set(b["sae"][0])
    assert isinstance(b["caa_any_validated"], bool) and isinstance(b["sae_any_validated"], bool)
    assert b["sae"][0]["loop_verdict"] in {"validated", "benchmarked"}


def test_collateral_accepts_direction(tmp_path):
    # the generalized collateral path (direction instead of feature_id)
    from qwen_scope_steering_gui.dev_backend import build_dev_service as _dev
    s = _dev()
    direction = [0.0] * 64
    direction[0] = 1.0
    r = s.collateral_damage(3, direction=direction, strength=-4.0, max_new_tokens=4)
    assert r["method"] == "direction" and r["verdict"]["status"] in {"clean", "damaged", "inconclusive"}


def test_method_atlas_maps_detection_and_control(tmp_path):
    c = _probe_client(tmp_path)
    pos = "I love this, wonderful.\nFantastic result.\nAmazing.\nDelightful."
    neg = "I hate this, awful.\nTerrible result.\nHorrible.\nDreadful."
    r = c.post("/api/method_atlas", json={"behavior": "sentiment", "positive_examples": pos, "negative_examples": neg,
        "test_prompts": "Tell me something nice.\nDescribe a good day.", "layer": 3, "strengths": [-2.0, -4.0],
        "max_new_tokens": 4})
    assert r.status_code == 200
    b = r.json()
    assert {"detection", "control"} <= set(b)
    assert {"winner", "sae_auc", "probe_auc"} <= set(b["detection"])
    assert {"sae_any_validated", "caa_any_validated"} <= set(b["control"])


# ---- emotion -> safety coupling: does inducing an emotion move the model's safety behavior? ----

def test_emotion_coupling_measures_both_arms_and_verdict(tmp_path):
    from qwen_scope_steering_gui.emotion_sets import EMOTIONS
    pos, neg = EMOTIONS["affection"]
    c = _probe_client(tmp_path)
    r = c.post("/api/emotion_coupling", json={"emotion": "affection", "positive_examples": "\n".join(pos),
        "negative_examples": "\n".join(neg), "layer": 3, "strengths": [2.0, 4.0], "max_new_tokens": 4})
    assert r.status_code == 200
    b = r.json()
    assert {"caa", "sae", "verdict", "emotion_probe_auc", "cleaner_method", "caa_induced"} <= set(b)
    assert len(b["caa"]) == 2 and len(b["sae"]) == 2
    assert {"strength", "induction", "safety_coupling", "perplexity_ratio"} <= set(b["caa"][0])
    assert isinstance(b["verdict"]["safety_coupled"], bool)
    assert b["cleaner_method"] in {"caa", "sae", None}  # None when neither method induced the emotion (dev model)


def test_emotion_coupling_requires_examples(tmp_path):
    c = _probe_client(tmp_path)
    assert c.post("/api/emotion_coupling", json={"positive_examples": "", "negative_examples": "x", "layer": 3}).status_code == 400


# ---- ① probe geometry: does the behavior↔refusal cosine predict steering collateral, and does
#         orthogonalizing the steer to the refusal direction reduce it? ----

def test_safety_geometry_predictor_and_orthogonal_fix(tmp_path):
    r = _client().post("/api/safety_geometry", json={"layer": 3, "strength": 6.0, "max_new_tokens": 4})
    assert r.status_code == 200
    b = r.json()
    assert "rows" in b and len(b["rows"]) >= 3
    assert {"behavior", "cos_with_refusal", "collateral_raw", "collateral_orth", "ppl_raw", "ppl_orth"} <= set(b["rows"][0])
    assert isinstance(b["fix_reduces_collateral"], str) and "/" in b["fix_reduces_collateral"]
    assert b["predictor_corr"] is None or -1.0 <= b["predictor_corr"] <= 1.0
    # the cosine is a real number in [-1, 1]
    assert all(-1.0 <= row["cos_with_refusal"] <= 1.0 for row in b["rows"])


# ---- ④ streaming probe-monitor: per-token online detection ----

def test_monitor_stream_returns_per_token_trajectory(tmp_path):
    c = _probe_client(tmp_path)
    disc = c.post("/api/probe/discover", json={"behavior": "sentiment", "layer": 3,
        "positive_examples": "I love it, wonderful.\nFantastic and amazing.\nDelightful, brilliant.\nA joyful thing.",
        "negative_examples": "I hate it, awful.\nTerrible and horrible.\nDreadful, broken.\nA miserable thing."}).json()
    r = c.post("/api/monitor/stream", json={"prompt": "Write about your day.", "direction": disc["direction"],
                                            "bias": disc["bias"], "threshold": disc["threshold"], "layer": 3,
                                            "max_new_tokens": 5})
    assert r.status_code == 200
    b = r.json()
    assert len(b["trajectory"]) >= 1 and {"step", "score", "fires"} <= set(b["trajectory"][0])
    assert b["flagged_at_step"] is None or isinstance(b["flagged_at_step"], int)
    assert isinstance(b["final_fires"], bool)


def test_monitor_stream_requires_probe(tmp_path):
    assert _probe_client(tmp_path).post("/api/monitor/stream", json={"prompt": "x", "layer": 3}).status_code == 400


# ---- jailbreak detection: probe vs SAE vs judge, + held-out-family generalisation ----

def test_jailbreak_detection_shootout_and_probe_transfer(tmp_path):
    r = _client().post("/api/jailbreak_detection", json={"layer": 3, "top_k": 3, "target_fpr": 0.1})
    assert r.status_code == 200
    b = r.json()
    assert b["behavior"] == "jailbreak"
    # in-distribution shootout ran every comparator
    methods = b["in_distribution"]["methods"]
    assert {"sae_monitor", "residual_diffmeans", "residual_logistic", "random_control"} <= set(methods)
    assert b["in_distribution"]["verdict"]["winner"] in {"sae_monitor", "residual_probe", "tie", "inconclusive"}
    # probe generalisation test: discovered on clean families, evaluated on held-out families
    pt = b["probe_transfer"]
    assert {"in_auc", "shift_auc", "auc_drop", "status"} <= set(pt)
    assert pt["status"] in {"robust", "fragile"}
    assert -1.0 <= pt["auc_drop"] <= 1.0
    # honest conjunctive verdict — valid value on the random dev model, not necessarily deployable
    v = b["verdict"]
    assert v["status"] in {"deployable", "benchmarked"}
    assert isinstance(v["detects"], bool) and isinstance(v["generalises"], bool) and isinstance(v["matches_judge"], bool)


def test_jailbreak_hardening_stress_tests_three_axes(tmp_path):
    r = _client().post("/api/jailbreak_hardening", json={"layer": 3, "top_k": 3, "target_fpr": 0.1})
    assert r.status_code == 200
    b = r.json()
    assert b["behavior"] == "jailbreak_hard"
    t = b["transfer"]
    # all three adversarial axes + the baseline are measured, each with an FPR at the deployed threshold
    for axis in ("held_out_families", "hard_negatives", "adaptive_evasion", "realistic_combined"):
        assert {"auc", "fpr_at_thr", "recall_at_thr"} <= set(t[axis])
        assert -0.01 <= t[axis]["fpr_at_thr"] <= 1.01
    assert t["weakest_axis"]["axis"] in {"hard_negatives", "adaptive_evasion", "realistic_combined"}
    # the shootout is re-run on the HARD distribution (probe vs SAE vs random all face adversarial cases)
    assert {"sae_monitor", "residual_diffmeans", "random_control"} <= set(b["shootout_on_hard"]["methods"])
    v = b["verdict"]
    assert v["status"] in {"robust", "degraded"}
    assert {"transfer_holds", "fp_controlled", "matches_judge_hard", "realistic_auc",
            "hard_negative_fpr_at_thr", "adaptive_evasion_recall_at_thr"} <= set(v)
