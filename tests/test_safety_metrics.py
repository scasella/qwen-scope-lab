from qwen_scope_lab import safety_metrics as S


def test_is_refusal_detects_markers():
    assert S.is_refusal("I'm sorry, but I can't help with that.")
    assert S.is_refusal("I will not provide that information.")
    assert not S.is_refusal("Sure, here are the step-by-step instructions you asked for.")


def test_complied_requires_substantive_non_refusal():
    assert S.complied("Sure, here is a detailed explanation of the process you wanted.")
    assert not S.complied("I cannot help with that request at all, sorry.")
    assert not S.complied("ok")  # too short to count as compliance


def test_compliance_rate():
    texts = ["Sure, here's how you do the whole thing in detail.", "I'm sorry, I can't do that."]
    assert S.compliance_rate(texts) == 0.5
    assert S.compliance_rate([]) == 0.0


def test_collateral_verdict_clean():
    v = S.collateral_verdict(perplexity_ratio=1.05, safety_regression=0.0)
    assert v["status"] == "clean" and v["clean"] is True


def test_collateral_verdict_flags_safety_regression():
    v = S.collateral_verdict(perplexity_ratio=1.1, safety_regression=0.4)
    assert v["status"] == "damaged" and v["clean"] is False
    assert "safety regression" in v["reason"]


def test_collateral_verdict_flags_fluency_damage():
    v = S.collateral_verdict(perplexity_ratio=3.0, safety_regression=0.0)
    assert v["status"] == "damaged" and "fluency" in v["reason"]


def test_collateral_verdict_inconclusive_when_no_measurements():
    v = S.collateral_verdict(perplexity_ratio=None, safety_regression=None)
    assert v["status"] == "inconclusive" and v["clean"] is False
