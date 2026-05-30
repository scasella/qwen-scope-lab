from qwen_scope_steering_gui import judge as J
from qwen_scope_steering_gui.dev_backend import build_dev_service


def test_parse_score_extracts_and_normalises():
    assert J.parse_score("85") == 0.85
    assert J.parse_score("The score is 100 out of 100") == 1.0
    assert J.parse_score("no number here") == 0.0
    assert J.parse_score("250") == 1.0  # clamped


def test_available_judge_off_by_default(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-fake")
    assert J.available_judge(enabled=False) is None  # disabled -> no judge even with a key


def test_available_judge_needs_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert J.available_judge(enabled=True) is None  # enabled but no key -> none, no call


class _FakeJudge:
    """A keyword judge — stands in for the LLM so tests make no network call."""
    def score(self, text: str, behavior_description: str) -> float:
        t = text.lower()
        return 1.0 if any(w in t for w in ("love", "wonderful", "fantastic", "amazing", "delightful")) else 0.0


def test_shootout_includes_injected_judge_as_fourth_method():
    s = build_dev_service()
    pos = ["I love this, it is wonderful.", "What a fantastic result.", "This is amazing.", "A delightful thing."]
    neg = ["I hate this, it is awful.", "What a terrible result.", "This is horrible.", "A dreadful thing."]
    r = s.monitor_shootout(pos, neg, layer=3, top_k=3, judge=_FakeJudge(), behavior="sentiment")
    assert "prompted_judge" in r["methods"]
    assert r["methods"]["prompted_judge"]["auc"] == 1.0  # keyword judge separates clean sentiment
    assert r["verdict"]["judge_auc"] == 1.0


class _AlwaysComply:
    def score(self, text: str, behavior_description: str) -> float:
        return 1.0


def test_collateral_uses_injected_judge_for_compliance():
    # an injected judge replaces refusal-string matching as the compliance scorer
    s = build_dev_service()
    r = s.collateral_damage(3, feature_id=42, strength=-4.0, max_new_tokens=4, judge=_AlwaysComply())
    assert r["steered_compliance_rate"] == 1.0 and r["unsteered_compliance_rate"] == 1.0
