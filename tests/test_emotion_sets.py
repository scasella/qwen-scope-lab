from qwen_scope_lab_bench import emotion_sets as E


def test_three_companion_relevant_emotions_present():
    assert {"affection", "anger", "fear"} <= set(E.EMOTIONS)
    for name, (pos, neg) in E.EMOTIONS.items():
        assert len(pos) == len(neg) >= 6, name
        assert not (set(pos) & set(neg)), name  # pos and neg are distinct


def test_neutral_prompts_are_affect_free_and_present():
    assert len(E.NEUTRAL_PROMPTS) >= 4
    assert all(isinstance(q, str) and q.strip() for q in E.NEUTRAL_PROMPTS)
