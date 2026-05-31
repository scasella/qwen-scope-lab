import pytest

from qwen_scope_lab_bench import probes as P


def _separable(n=8, d=6):
    pos = [[3.0 + 0.1 * i, 0.0, 0.0, 0.0, 0.0, 0.0] for i in range(n)]
    neg = [[0.0, 3.0 + 0.1 * i, 0.0, 0.0, 0.0, 0.0] for i in range(n)]
    return pos, neg


def test_discover_probe_validates_on_separable():
    pos, neg = _separable()
    r = P.discover_probe(pos, neg, method="diffmeans")
    assert r["metrics"]["auc"] == 1.0
    assert r["metrics"]["control_auc"] < 0.85
    assert r["validation_decision"]["status"] == "validated"
    assert len(r["direction"]) == 6 and "threshold" in r


def test_discover_probe_logistic_also_separates():
    pos, neg = _separable()
    r = P.discover_probe(pos, neg, method="logistic")
    assert r["metrics"]["auc"] == 1.0


def test_discover_probe_ensemble_separates():
    pos, neg = _separable()
    r = P.discover_probe(pos, neg, method="ensemble")
    assert r["metrics"]["auc"] == 1.0 and len(r["direction"]) == 6


def test_discover_probe_random_is_benchmarked():
    import random
    rng = random.Random(0)
    pos = [[rng.gauss(0, 1) for _ in range(6)] for _ in range(8)]
    neg = [[rng.gauss(0, 1) for _ in range(6)] for _ in range(8)]
    r = P.discover_probe(pos, neg)
    assert r["validation_decision"]["status"] == "benchmarked"


def test_score_probe_fires_on_positive_side():
    pos, neg = _separable()
    r = P.discover_probe(pos, neg)
    hit = P.score_probe(r["direction"], r["bias"], r["threshold"], [3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    miss = P.score_probe(r["direction"], r["bias"], r["threshold"], [0.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    assert hit["fires"] is True and miss["fires"] is False


def test_unit_direction_is_normalised():
    u = P.unit_direction([3.0, 4.0, 0.0])
    assert abs((u[0] ** 2 + u[1] ** 2 + u[2] ** 2) ** 0.5 - 1.0) < 1e-6


def test_discover_probe_requires_both_sides():
    with pytest.raises(ValueError):
        P.discover_probe([], [[1.0, 2.0]])
