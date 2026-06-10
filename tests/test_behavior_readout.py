"""C05 — behavior_readout='first_token'|'full_string' on the manifold behavior-energy path.

These run on the GPU-free dev backend (no model download, CI-safe). They prove the structural
contract — first-token collisions, full-string separation, readout threading, cache split, and
a normalized full-string distribution — not real-model numbers (that's the MLX/Modal audit).
"""
import math

import pytest

from qwen_scope_lab.concept_presets import get_concept
from qwen_scope_lab.dev_backend import build_dev_service


@pytest.fixture(scope="module")
def svc():
    return build_dev_service()


def _layer(svc, concept_name):
    return svc._build_manifold(concept_name, None)[2]


def test_first_token_collides_where_full_string_separates(svc):
    """'strongly disagree' / 'strongly agree' share a first token but differ over the full string —
    the exact tokenization risk C05 audits. first_token collapses them; full_string must not."""
    c = get_concept("agreement")
    first_ids = svc._concept_token_ids(c)
    conts = [tuple(x) for x in svc._value_continuation_ids(c)]

    assert len(set(first_ids)) < len(first_ids), "expected a first-token collision among Likert labels"
    assert len(set(conts)) == len(conts), "full-string continuations must be distinct per value"
    # every continuation starts with the value's first-token id (the readouts agree on token 0)
    assert [conts[i][0] for i in range(len(conts))] == first_ids


def test_compare_threads_readout_for_both_modes(svc):
    c = "agreement"
    default = svc.manifold_compare(c, target=None)
    assert default["behavior_readout"] == "first_token", "default must stay first_token"

    ft = svc.manifold_compare(c, target=None, behavior_readout="first_token")
    fs = svc.manifold_compare(c, target=None, behavior_readout="full_string")
    assert ft["behavior_readout"] == "first_token"
    assert fs["behavior_readout"] == "full_string"
    for legs in (ft, fs):
        for leg in ("manifold", "linear"):
            assert legs[leg]["mean_energy"] is not None
            assert legs[leg]["hook_fired"] is True
            assert legs[leg]["behavior_readout"] == legs["behavior_readout"]


def test_behavior_cache_splits_by_readout(svc):
    name = "rank"
    layer = _layer(svc, name)
    c = get_concept(name)
    b_ft = svc._build_behavior_manifold(c, layer, "first_token")
    b_fs = svc._build_behavior_manifold(c, layer, "full_string")
    assert b_ft["behavior_readout"] == "first_token"
    assert b_fs["behavior_readout"] == "full_string"
    assert (name, layer, "first_token") in svc._behavior_cache
    assert (name, layer, "full_string") in svc._behavior_cache
    # same value-grid shape, independently cached objects
    assert b_ft["dense_p"].shape == b_fs["dense_p"].shape
    assert b_ft is not b_fs


def test_full_string_distribution_is_a_valid_simplex(svc):
    name = "education"  # multi-token labels ('middle school', 'high school')
    c = get_concept(name)
    layer = _layer(svc, name)
    item0 = c.items[0]
    prompt = c.steer_prompt.format(item=item0)
    pos = svc._locate_item_position(svc.bundle.tokenizer, c.steer_prompt, item0, prompt)
    dist = svc._value_string_distribution(prompt, layer, None, pos, c)
    assert len(dist) == len(c.items)
    assert all(p >= 0.0 for p in dist)
    assert math.isclose(float(sum(dist)), 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_compare_request_schema_carries_readout():
    from qwen_scope_lab.web_api import ManifoldCompareReq

    req = ManifoldCompareReq(concept="agreement", target="strongly agree", behavior_readout="full_string")
    dumped = req.model_dump()
    assert dumped["behavior_readout"] == "full_string"
    # default preserves first_token
    assert ManifoldCompareReq(concept="rank", target="general").model_dump()["behavior_readout"] == "first_token"
