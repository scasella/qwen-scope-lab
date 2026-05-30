from qwen_scope_steering_gui import control_loop as CL


def _rows(unsteered, steered):
    return [{"unsteered_fires": u, "steered_fires": s} for u, s in zip(unsteered, steered)]


def test_summarize_fires_rates_and_suppression():
    # 3/4 fire unsteered; of those, 2 are suppressed
    rows = _rows([True, True, True, False], [False, False, True, False])
    f = CL.summarize_fires(rows)
    assert f["fire_rate_unsteered"] == 0.75
    assert f["fire_rate_steered"] == 0.25
    assert f["n_fired_unsteered"] == 3
    assert f["suppression_rate"] == round(2 / 3, 4)


def test_verdict_validated_when_suppressed_and_clean():
    fires = {"fire_rate_unsteered": 1.0, "suppression_rate": 0.9}
    v = CL.loop_verdict(fires, {"clean": True}, min_fire=0.5, min_suppression=0.5)
    assert v["status"] == "validated" and v["passed"] is True


def test_verdict_benchmarked_when_behavior_absent():
    fires = {"fire_rate_unsteered": 0.1, "suppression_rate": 1.0}
    v = CL.loop_verdict(fires, {"clean": True})
    assert v["status"] == "benchmarked" and "no behavior to suppress" in v["reason"]


def test_verdict_benchmarked_when_collateral_damage():
    fires = {"fire_rate_unsteered": 1.0, "suppression_rate": 1.0}
    v = CL.loop_verdict(fires, {"clean": False, "reason": "safety regression +40%"})
    assert v["status"] == "benchmarked" and "collateral damage" in v["reason"]


def test_verdict_benchmarked_when_weak_suppression():
    fires = {"fire_rate_unsteered": 1.0, "suppression_rate": 0.2}
    v = CL.loop_verdict(fires, {"clean": True}, min_suppression=0.5)
    assert v["status"] == "benchmarked" and "suppression only" in v["reason"]


def test_verdict_ignores_collateral_when_disabled():
    fires = {"fire_rate_unsteered": 1.0, "suppression_rate": 0.9}
    v = CL.loop_verdict(fires, {}, measure_collateral=False)
    assert v["status"] == "validated"
