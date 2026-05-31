from qwen_scope_lab import behavior_sets as BS


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


def test_jailbreak_clean_and_shift_are_different_attack_families():
    b = BS.BEHAVIORS["jailbreak"]
    pos, neg = b["clean"]
    spos, sneg = b["shift"]
    assert len(pos) == len(neg) >= 6 and len(spos) == len(sneg) >= 6
    # shift families must be held-out (distinct strings) so robustness is real generalisation
    assert not (set(pos) & set(spos))
    # and they should NOT share the clean set's overt trigger tokens — that's the whole point:
    # detecting them tests manipulation-intent, not keyword memorisation.
    clean_blob, shift_blob = " ".join(pos).lower(), " ".join(spos).lower()
    for trigger in ("dan", "developer mode", "ignore all previous", "no rules"):
        assert trigger in clean_blob and trigger not in shift_blob


def test_jailbreak_hardening_sets_present_and_distinct():
    b = BS.BEHAVIORS["jailbreak"]
    hard_neg, evasion = b["hard_negatives"], b["evasion"]
    assert len(hard_neg) >= 6 and len(evasion) >= 6
    # hard negatives are BENIGN but wear jailbreak surface forms (false-positive bait)
    assert any(w in " ".join(hard_neg).lower() for w in ("ignore", "roleplay", "pretend", "developer", "grandmother"))
    # adaptive evasion are jailbreaks WITHOUT the overt clean-set trigger tokens (false-negative bait)
    ev_blob = " ".join(evasion).lower()
    for trigger in ("dan", "developer mode", "ignore all previous", "no rules"):
        assert trigger not in ev_blob
    # the hardening sets are disjoint from the clean/shift sets
    pos, _ = b["clean"]
    spos, _ = b["shift"]
    assert not (set(evasion) & (set(pos) | set(spos)))
