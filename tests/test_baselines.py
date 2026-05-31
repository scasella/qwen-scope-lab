import pytest

from qwen_scope_lab_bench import baselines as B


def test_eval_scores_perfect_separation():
    m = B.eval_scores([1.0, 1.0, 0.9], [0.0, 0.1, 0.0], thr=0.5)
    assert m["auc"] == 1.0 and m["f1"] == 1.0 and m["fpr"] == 0.0


def test_threshold_at_fpr_budget():
    neg = [float(i) for i in range(1, 11)]  # 1..10
    # allow 10% FPR -> exactly one of ten negatives may fire
    thr = B.threshold_at_fpr(neg, 0.1)
    assert sum(n >= thr for n in neg) == 1
    # a zero FPR budget admits no false positives
    thr0 = B.threshold_at_fpr(neg, 0.0)
    assert sum(n >= thr0 for n in neg) == 0


def test_diff_means_probe_separates_residuals():
    pos = [[2.0, 0.0, 0.0], [2.1, 0.1, 0.0], [1.9, 0.0, 0.1]]
    neg = [[0.0, 2.0, 0.0], [0.1, 2.1, 0.0], [0.0, 1.9, 0.1]]
    w, b = B.diff_means_probe(pos, neg)
    ps, ns = B._project(pos, w, b), B._project(neg, w, b)
    assert min(ps) > max(ns)  # cleanly linearly separable


def _separable_inputs():
    pos_maps = [{1: 2.0}, {1: 1.8, 9: 0.2}, {1: 2.1}, {1: 1.9}]
    neg_maps = [{2: 1.0}, {3: 0.9}, {2: 1.1}, {4: 0.8}]
    pos_res = [[2.0, 0.0], [2.1, 0.0], [1.9, 0.1], [2.0, 0.0]]
    neg_res = [[0.0, 2.0], [0.0, 2.1], [0.1, 1.9], [0.0, 2.0]]
    return pos_maps, neg_maps, pos_res, neg_res


def test_shootout_structure_and_methods():
    r = B.shootout(*_separable_inputs(), top_k=2, d_sae=256, target_fpr=0.25)
    assert {"sae_monitor", "random_control", "residual_diffmeans", "residual_logistic"} <= set(r["methods"])
    for name in ("sae_monitor", "residual_diffmeans", "residual_logistic"):
        assert {"auc", "f1", "tpr_at_fpr", "threshold"} <= set(r["methods"][name])
    assert r["verdict"]["winner"] in {"sae_monitor", "residual_probe", "tie", "inconclusive"}
    assert r["methods"]["sae_monitor"]["features"]  # the selected SAE feature(s)


def test_shootout_both_detectors_separate_clean_data():
    r = B.shootout(*_separable_inputs(), top_k=2, d_sae=256)
    assert r["methods"]["sae_monitor"]["auc"] == 1.0
    assert r["methods"]["residual_diffmeans"]["auc"] == 1.0
    assert r["verdict"]["control_auc"] < 0.9  # random features do not separate


def test_shootout_requires_aligned_residuals():
    with pytest.raises(ValueError):
        B.shootout([{1: 1.0}], [{2: 1.0}], [[1.0]], [], top_k=1, d_sae=8)


def test_shootout_requires_both_sides():
    with pytest.raises(ValueError):
        B.shootout([], [{1: 1.0}], [], [[1.0]])
