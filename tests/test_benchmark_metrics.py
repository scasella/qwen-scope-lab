from qwen_scope_lab_bench.benchmark_metrics import aggregate_metrics, json_validity, repeated_ngram_rate, required_forbidden_terms, text_metrics


def test_json_validity_metric():
    assert json_validity('{"a":1}')["json_validity"] == 1.0
    assert json_validity("not json")["json_validity"] == 0.0


def test_length_repetition_and_distinct_metrics():
    metrics = text_metrics("yes yes yes yes")

    assert metrics["output_length_tokens"] == 4
    assert repeated_ngram_rate("a b c a b c", n=3) > 0
    assert metrics["distinct_1"] < 1.0


def test_required_forbidden_terms():
    metrics = required_forbidden_terms("alpha beta", ["alpha"], ["gamma"])

    assert metrics["contains_required_terms"] is True
    assert metrics["excludes_forbidden_terms"] is True


def test_aggregate_metric_calculation():
    aggregate = aggregate_metrics([{"json_validity": 1.0}, {"json_validity": 0.0}])

    assert aggregate["json_validity"] == 0.5
