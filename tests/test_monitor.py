import pytest

from qwen_scope_lab import monitor as M


def test_activation_map_takes_max_across_tokens():
    insp = {"top_features_by_token": [
        {"features": [{"feature_id": 1, "activation": 0.3}]},
        {"features": [{"feature_id": 1, "activation": 0.8}, {"feature_id": 2, "activation": 0.5}]},
    ]}
    m = M.activation_map(insp)
    assert m[1] == 0.8 and m[2] == 0.5


def test_discover_separable_validates():
    pos = [{1: 2.0}, {1: 1.8, 9: 0.2}, {1: 2.1}, {1: 1.9}]
    neg = [{2: 1.0}, {3: 0.9}, {2: 1.1}, {4: 0.8}]
    r = M.discover(pos, neg, top_k=2, d_sae=256)
    assert 1 in r["features"]
    assert r["metrics"]["auc"] == 1.0
    assert r["metrics"]["control_auc"] < 0.9
    assert r["validation_decision"]["status"] == "validated"


def test_discover_random_is_benchmarked_not_validated():
    import random
    rng = random.Random(0)
    pos = [{rng.randint(1, 40): 1.0} for _ in range(8)]
    neg = [{rng.randint(1, 40): 1.0} for _ in range(8)]
    r = M.discover(pos, neg, top_k=2, d_sae=256)
    assert r["validation_decision"]["status"] == "benchmarked"


def test_score_fires_above_threshold():
    assert M.score([1, 2], 0.5, {1: 0.9}) == {"score": 0.9, "fires": True}
    assert M.score([1, 2], 0.5, {3: 0.9})["fires"] is False


def test_discover_requires_both_sides():
    with pytest.raises(ValueError):
        M.discover([], [{1: 1.0}])
