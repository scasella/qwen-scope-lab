from qwen_scope_lab.candidate_search import fake_inspection, rank_candidates_from_inspections


def inspection(feature_id, activation):
    return {"top_features_by_token": [{"features": [{"feature_id": feature_id, "activation": activation}]}]}


def test_positive_only_features_rank_high():
    ranked = rank_candidates_from_inspections(
        layer=1,
        positive_inspections=[inspection(10, 10.0)],
        negative_inspections=[inspection(20, 10.0)],
        limit=2,
    )

    assert ranked[0].feature_id == 10


def test_features_active_in_both_are_penalized_and_ties_stable():
    ranked = rank_candidates_from_inspections(
        layer=1,
        positive_inspections=[inspection(5, 5.0), inspection(6, 5.0)],
        negative_inspections=[inspection(5, 5.0)],
        limit=2,
    )

    assert ranked[0].feature_id == 6


def test_fake_inspection_uses_stable_feature_ids():
    first = fake_inspection('{"name":"Ada","age":31}', layer=0, positive=True)
    second = fake_inspection('{"name":"Ada","age":31}', layer=0, positive=True)

    assert first["top_features_by_token"][0]["features"][0]["feature_id"] == 967
    assert first == second
