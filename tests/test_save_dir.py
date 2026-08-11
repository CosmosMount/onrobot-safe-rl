from pathlib import Path

from train.loop import prepare_save_dir


def test_prepare_save_dir_clears_fresh_run_but_preserves_resume(tmp_path: Path):
    root = tmp_path / "experiment"
    (root / "step_000000000100").mkdir(parents=True)
    (root / "step_000000000100" / "agent.pt").write_text("old")
    (root / "manifest.json").write_text("old")

    prepare_save_dir(str(root), resume=False)
    assert list(root.iterdir()) == []

    (root / "manifest.json").write_text("keep")
    prepare_save_dir(str(root), resume=True)
    assert (root / "manifest.json").read_text() == "keep"
