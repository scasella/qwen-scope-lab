"""Tests for the offline Mixture Dial Distill compiler."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from qwen_scope_lab.experiments import mixture_dial_distill as md

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mixture_dial_distill.py"


def _write_jsonl(path: Path, rows: list[dict] | list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row if isinstance(row, str) else json.dumps(row))
            f.write("\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_requested_counts_mixed_count_and_ratio_largest_remainder():
    spec = md.normalize_mixture_spec(
        {
            "seed": 0,
            "total": 7,
            "slots": [
                {"name": "fixed", "where": {"class": "fixed"}, "count": 2},
                {"name": "a", "where": {"class": "a"}, "ratio": 1},
                {"name": "b", "where": {"class": "b"}, "ratio": 1},
                {"name": "c", "where": {"class": "c"}, "ratio": 1},
            ],
        }
    )
    assert md.requested_counts_for_slots(spec.slots, spec.total) == {"fixed": 2, "a": 2, "b": 2, "c": 1}


def test_truth_holding_fixture_compiles_balanced_corpus(tmp_path):
    out = tmp_path / "truth"
    summary = md.compile_to_dir(
        ROOT / "examples" / "mixture_dial" / "truth_holding_mixture.yml",
        ROOT / "examples" / "mixture_dial" / "truth_holding_candidates.jsonl",
        out,
    )
    assert summary["n_sft"] == 12
    manifest = json.loads((out / "mixture_manifest.json").read_text(encoding="utf-8"))
    assert manifest["requested_counts"] == {
        "A_factual_false_pressure": 6,
        "B_unknowable_calibration": 3,
        "C_subjective_calibration": 3,
    }
    assert manifest["achieved_counts"] == manifest["requested_counts"]
    assert manifest["capped_slots"] == {}
    assert manifest["dropped_rows"]["total"] == 0
    sft = _read_jsonl(out / "sft.jsonl")
    assert {row["behavioral_class"] for row in sft} == {"A_factual", "B_unknowable", "C_subjective"}
    assert all([m["role"] for m in row["messages"]] == ["user", "assistant"] for row in sft)
    assert "skipped" not in manifest["output_paths"]


def test_sampling_is_seeded_deterministic_without_replacement(tmp_path):
    rows = [
        {"id": f"r{i}", "prompt": f"prompt {i}", "output": f"answer {i}", "class": "x", "pressure": "p"}
        for i in range(10)
    ]
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(candidates, rows)
    config = tmp_path / "mixture.yml"
    config.write_text(
        "\n".join(
            [
                'schema_version: "0.1.0"',
                "seed: 4",
                "total: 4",
                "slots:",
                "  - name: x",
                "    where: {class: x}",
                "    count: 4",
            ]
        ),
        encoding="utf-8",
    )
    md.compile_to_dir(config, candidates, tmp_path / "a")
    md.compile_to_dir(config, candidates, tmp_path / "b")
    a = [row["source_id"] for row in _read_jsonl(tmp_path / "a" / "sft.jsonl")]
    b = [row["source_id"] for row in _read_jsonl(tmp_path / "b" / "sft.jsonl")]
    assert a == b
    assert len(a) == len(set(a)) == 4


def test_underfilled_slot_reports_cap(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(
        candidates,
        [
            {"id": "a", "prompt": "p1", "output": "o1", "class": "scarce"},
            {"id": "b", "prompt": "p2", "output": "o2", "class": "scarce"},
        ],
    )
    config = tmp_path / "mixture.yml"
    config.write_text(
        "\n".join(
            [
                'schema_version: "0.1.0"',
                "seed: 0",
                "total: 5",
                "slots:",
                "  - name: scarce",
                "    where: {class: scarce}",
                "    count: 5",
            ]
        ),
        encoding="utf-8",
    )
    md.compile_to_dir(config, candidates, tmp_path / "out")
    manifest = json.loads((tmp_path / "out" / "mixture_manifest.json").read_text(encoding="utf-8"))
    assert manifest["achieved_total"] == 2
    assert manifest["capped_slots"]["scarce"] == {"requested": 5, "available": 2}
    assert manifest["underfilled_slots"]["scarce"]["underfilled_by"] == 3


def test_malformed_and_excluded_rows_write_skipped_jsonl(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(
        candidates,
        [
            {"id": "good", "prompt": "p", "output": "o", "class": "x", "quality_label": "clean"},
            {"id": "missing_output", "prompt": "p", "class": "x", "quality_label": "clean"},
            {"id": "excluded", "prompt": "p", "output": "o", "class": "x", "quality_label": "drop"},
        ],
    )
    config = tmp_path / "mixture.yml"
    config.write_text(
        "\n".join(
            [
                'schema_version: "0.1.0"',
                "seed: 0",
                "total: 1",
                "dimensions: [class, rejection_mode]",
                "aliases:",
                "  rejection_mode: [quality_label]",
                "exclude_rejection_modes: [drop]",
                "slots:",
                "  - name: x",
                "    where: {class: x}",
                "    count: 1",
            ]
        ),
        encoding="utf-8",
    )
    md.compile_to_dir(config, candidates, tmp_path / "out")
    skipped = _read_jsonl(tmp_path / "out" / "skipped.jsonl")
    reasons = [row["reason"] for row in skipped]
    assert reasons == ["malformed_row", "excluded_rejection_mode"]
    manifest = json.loads((tmp_path / "out" / "mixture_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dropped_rows"]["malformed"] == 1
    assert manifest["dropped_rows"]["excluded_by_rejection_mode"] == 1


def test_generic_fixture_uses_custom_labels_and_rejection_alias(tmp_path):
    out = tmp_path / "generic"
    md.compile_to_dir(
        ROOT / "examples" / "mixture_dial" / "generic_mixture.yml",
        ROOT / "examples" / "mixture_dial" / "generic_candidates.jsonl",
        out,
    )
    manifest = json.loads((out / "mixture_manifest.json").read_text(encoding="utf-8"))
    assert manifest["requested_counts"] == {
        "supportive_low_pressure": 3,
        "exact_format_clean": 3,
        "boundary_pushback": 2,
    }
    assert manifest["achieved_total"] == 8
    assert manifest["dropped_rows"]["excluded_by_rejection_mode"] == 1
    assert manifest["label_summaries"]["selected"]["class"] == {"boundary": 2, "exact": 3, "supportive": 3}
    sft = _read_jsonl(out / "sft.jsonl")
    assert all(row["labels"]["rejection_mode"] == ["clean"] for row in sft)
    assert (out / "skipped.jsonl").exists()


def test_cli_compile_smokes_for_both_fixtures(tmp_path):
    truth_out = tmp_path / "truth_cli"
    generic_out = tmp_path / "generic_cli"
    truth = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "compile",
            "--mixture",
            str(ROOT / "examples" / "mixture_dial" / "truth_holding_mixture.yml"),
            "--candidates",
            str(ROOT / "examples" / "mixture_dial" / "truth_holding_candidates.jsonl"),
            "--out",
            str(truth_out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    generic = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "compile",
            "--mixture",
            str(ROOT / "examples" / "mixture_dial" / "generic_mixture.yml"),
            "--candidates",
            str(ROOT / "examples" / "mixture_dial" / "generic_candidates.jsonl"),
            "--out",
            str(generic_out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(truth.stdout)["n_sft"] == 12
    assert json.loads(generic.stdout)["n_sft"] == 8
    for out in (truth_out, generic_out):
        assert (out / "sft.jsonl").exists()
        assert (out / "mixture_manifest.json").exists()
