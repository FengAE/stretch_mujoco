from pathlib import Path

from tools.evaluate_openpi_checkpoints import checkpoint_steps, select_best, summarize


def test_checkpoint_selection_keeps_intervals_and_latest(tmp_path: Path) -> None:
    for name in ("5000", "9000", "10000", "14999"):
        (tmp_path / name).mkdir()
    (tmp_path / "wandb_id.txt").write_text("run")

    assert [step for step, _ in checkpoint_steps(tmp_path)] == [5000, 10000, 14999]


def test_summary_and_best_checkpoint_prioritize_closed_loop_success() -> None:
    weak = summarize(
        5000,
        Path("5000"),
        {
            "success_rate": 0.1,
            "episodes": [{"bilateral_contact": True, "max_lift_m": 0.1}],
        },
    )
    strong = summarize(
        10000,
        Path("10000"),
        {
            "success_rate": 0.5,
            "episodes": [{"bilateral_contact": False, "max_lift_m": 0.0}],
        },
    )

    assert select_best([weak, strong]) == strong
