from qwen_scope_steering_gui import behavior_sets as BS


def test_sycophancy_has_clean_and_shift_balanced_pairs():
    pos, neg = BS.BEHAVIORS["sycophancy"]["clean"]
    spos, sneg = BS.BEHAVIORS["sycophancy"]["shift"]
    assert len(pos) == len(neg) >= 6
    assert len(spos) == len(sneg) >= 6
    # shift forms must be distinct surface strings, not copies of the clean set
    assert not (set(pos) & set(spos))
    assert BS.BEHAVIORS["sycophancy"]["test_prompts"]


def test_refusal_clean_pair_present():
    pos, neg = BS.BEHAVIORS["refusal"]["clean"]
    assert len(pos) == len(neg) >= 6


def test_sentiment_is_safety_decoupled_behavior_with_tests():
    b = BS.BEHAVIORS["sentiment"]
    pos, neg = b["clean"]
    spos, sneg = b["shift"]
    assert len(pos) == len(neg) >= 6 and len(spos) == len(sneg) >= 6
    assert b["test_prompts"] and not (set(pos) & set(spos))
