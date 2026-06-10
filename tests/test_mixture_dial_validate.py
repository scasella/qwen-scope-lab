"""Tests for the Mixture Dial Distiller dry-run validation scaffold."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mixture_dial_validate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("mixture_dial_validate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


mv = _load_module()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _row(i: int, cls: str) -> dict:
    return {
        "id": f"{cls}_{i}",
        "prompt": f"question {cls} {i}",
        "output": f"answer {cls} {i}",
        "behavioral_class": cls,
        "pressure_type": "test_pressure",
        "domain": "test_domain",
        "rejection_mode": "kept",
    }


def _candidate_file(tmp_path: Path, counts: dict[str, int]) -> Path:
    rows = []
    for cls, n in counts.items():
        rows.extend(_row(i, cls) for i in range(n))
    path = tmp_path / "candidates.jsonl"
    _write_jsonl(path, rows)
    return path


def _run(tmp_path: Path, candidates: Path, *, n: int = 8, seeds: str = "0,1", calib_fracs: str = "0.0,0.25,0.50,0.75") -> dict:
    args = mv.build_parser().parse_args(
        [
            "--out",
            str(tmp_path / "out"),
            "--candidates",
            str(candidates),
            "--n",
            str(n),
            "--seeds",
            seeds,
            "--calib-fracs",
            calib_fracs,
        ]
    )
    return mv.run_validation(args)


def test_arm_matrix_has_expected_arms_x_seeds_and_fixed_n(tmp_path):
    candidates = _candidate_file(tmp_path, {"A_factual": 8, "B_unknowable": 6, "C_subjective": 4})
    plan = _run(tmp_path, candidates)

    expected_arms = {
        "truth_only",
        "calib_frac_0.25",
        "calib_frac_0.50",
        "calib_frac_0.75",
        "random_same_size",
        "naive_stratified",
    }
    assert {arm["name"] for arm in plan["arms"]} == expected_arms
    assert plan["arm_seed_count"] == len(expected_arms) * 2
    assert {entry["seed"] for entry in plan["matrix"]} == {0, 1}
    assert {entry["target_total"] for entry in plan["matrix"]} == {8}


def test_dose_response_counts_are_largest_remainder_and_zero_is_truth_only(tmp_path):
    candidates = _candidate_file(tmp_path, {"A_factual": 8, "B_unknowable": 6, "C_subjective": 4})
    plan = _run(tmp_path, candidates)
    arms = {arm["name"]: arm for arm in plan["arms"]}

    assert arms["truth_only"]["calib_frac"] == 0.0
    assert arms["truth_only"]["requested_class_counts"] == {"A_factual": 8}
    assert arms["calib_frac_0.25"]["requested_class_counts"] == {
        "A_factual": 6,
        "B_unknowable": 1,
        "C_subjective": 1,
    }
    assert arms["calib_frac_0.50"]["requested_class_counts"] == {
        "A_factual": 4,
        "B_unknowable": 2,
        "C_subjective": 2,
    }
    assert arms["calib_frac_0.75"]["requested_class_counts"] == {
        "A_factual": 2,
        "B_unknowable": 4,
        "C_subjective": 2,
    }


def test_naive_stratified_reproduces_natural_class_proportions_at_n(tmp_path):
    candidates = _candidate_file(tmp_path, {"A_factual": 8, "B_unknowable": 6, "C_subjective": 4})
    plan = _run(tmp_path, candidates)
    naive = next(arm for arm in plan["arms"] if arm["name"] == "naive_stratified")

    assert plan["candidate_vocabulary"]["class_counts"] == {
        "A_factual": 8,
        "B_unknowable": 6,
        "C_subjective": 4,
    }
    assert naive["requested_class_counts"] == {
        "A_factual": 3,
        "B_unknowable": 3,
        "C_subjective": 2,
    }


def test_compiled_sft_count_matches_achieved_total_and_underfill_is_not_padded(tmp_path):
    candidates = _candidate_file(tmp_path, {"A_factual": 2, "B_unknowable": 1, "C_subjective": 1})
    plan = _run(tmp_path, candidates, seeds="0", calib_fracs="0.75")

    assert any(entry["underfilled_slots"] for entry in plan["matrix"])
    for entry in plan["matrix"]:
        sft_rows = _read_jsonl(Path(entry["sft"]))
        assert len(sft_rows) == entry["achieved_total"]

    high_calib = next(entry for entry in plan["matrix"] if entry["arm"] == "calib_frac_0.75")
    assert high_calib["requested_class_counts"] == {
        "A_factual": 2,
        "B_unknowable": 3,
        "C_subjective": 3,
    }
    assert high_calib["achieved_class_counts"] == {
        "A_factual": 2,
        "B_unknowable": 1,
        "C_subjective": 1,
    }
    assert high_calib["achieved_total"] == 4
    assert high_calib["underfilled_slots"]["B_unknowable"]["underfilled_by"] == 2
    assert high_calib["underfilled_slots"]["C_subjective"]["underfilled_by"] == 2


def test_plan_files_are_written_with_kill_criteria(tmp_path):
    candidates = _candidate_file(tmp_path, {"A_factual": 8, "B_unknowable": 6, "C_subjective": 4})
    plan = _run(tmp_path, candidates)
    out = tmp_path / "out"

    plan_json = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    plan_md = (out / "plan.md").read_text(encoding="utf-8")
    assert plan_json["arm_seed_count"] == plan["arm_seed_count"]
    assert set(plan_json["kill_criteria"]) == {"K1", "K2", "K3"}
    assert "KILL CRITERIA" in plan_md
    assert "K1: dose-response is flat" in plan_md
    assert "K2: dials are approximately equal to naive_stratified" in plan_md
    assert "K3: the win does not survive matched-size" in plan_md
