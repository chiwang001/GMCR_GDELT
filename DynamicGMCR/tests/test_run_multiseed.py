from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "cod" / "run_multiseed.py"
SPEC = importlib.util.spec_from_file_location("demo4_multiseed", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def fake_args(input_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        input_dir=input_dir,
        solver_output_dir=input_dir.parent / "output",
        preset="quick",
        perturbation_seed=20260701,
        de_early_stop_patience=7,
        local_search_early_stop_patience=11,
        max_workers=1,
        force=False,
        postprocess_stability=False,
        prediction_month=202002,
    )


def write_complete_run(output_dir: Path, seed: int, name: str = "31-12-00") -> Path:
    run_dir = output_dir / name
    run_dir.mkdir(parents=True)
    (run_dir / f"parameters_{name}.json").write_text(
        json.dumps({"vector": [0.0] * 42}), encoding="utf-8"
    )
    (run_dir / f"summary_{name}.json").write_text(
        json.dumps(
            {
                "config": {"seed": seed},
                "objective": {
                    "loss": 1.25,
                    "fit_objective": -1.25,
                    "seq_stability_penalty": 0.1,
                },
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_parse_int_list_accepts_ten_seeds_and_rejects_duplicates() -> None:
    assert runner.parse_int_list("1, 2,3") == (1, 2, 3)
    try:
        runner.parse_int_list("1,1")
    except argparse.ArgumentTypeError:
        pass
    else:
        raise AssertionError("Duplicate seeds should be rejected")


def test_child_command_passes_optimizer_and_fixed_perturbation_seeds() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        args = fake_args(root / "input")
        command = runner.child_command(args, 20250731, root / "output")

    assert command[command.index("--seed") + 1] == "20250731"
    assert command[command.index("--perturbation-seed") + 1] == "20260701"
    assert command[command.index("--output-dir") + 1].endswith("output")
    assert command[command.index("--de-early-stop-patience") + 1] == "7"


def test_complete_run_requires_matching_seed_and_required_files() -> None:
    with tempfile.TemporaryDirectory() as temp:
        run_dir = write_complete_run(Path(temp) / "output", 5)
        assert runner.result_is_complete(run_dir, 5)
        assert not runner.result_is_complete(run_dir, 6)
        (run_dir / "parameters_31-12-00.json").unlink()
        assert not runner.result_is_complete(run_dir, 5)


def test_find_complete_run_ignores_newer_incomplete_directory() -> None:
    with tempfile.TemporaryDirectory() as temp:
        output_dir = Path(temp) / "output"
        expected = write_complete_run(output_dir, 7, "31-12-00")
        incomplete = output_dir / "31-12-01"
        incomplete.mkdir()
        assert runner.find_complete_run(output_dir, 7) == expected


def test_summary_extracts_objective_and_resume_skips_solver() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        seed_dir = root / "seed_9"
        expected = write_complete_run(root / "output", 9)
        seed_dir.mkdir()
        (seed_dir / "run_status.json").write_text(
            json.dumps({"run_dir": str(expected)}), encoding="utf-8"
        )
        row = runner.summarize_seed(seed_dir, 9)
        assert row["run_dir"] == str(expected)
        assert row["loss"] == 1.25

        with patch.object(runner.subprocess, "run") as subprocess_run:
            skipped = runner.run_one(fake_args(root / "input"), root, 9)
        subprocess_run.assert_not_called()
        assert skipped["status"] == "skipped_existing"


def test_force_does_not_accept_an_old_result_as_the_new_run() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        seed_dir = root / "seed_12"
        old_run = write_complete_run(root / "output", 12)
        seed_dir.mkdir()
        (seed_dir / "run_status.json").write_text(
            json.dumps({"run_dir": str(old_run)}), encoding="utf-8"
        )
        args = fake_args(root / "input")
        args.force = True
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch.object(runner.subprocess, "run", return_value=completed):
            row = runner.run_one(args, root, 12)

        assert row["status"] == "failed"
        assert row["run_dir"] == ""
        assert "found 0" in row["error"]
