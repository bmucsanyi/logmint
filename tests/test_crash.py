"""Preemption is the normal case, not the exception (spec sections 3, 6 and 10)."""

import signal
import subprocess  # noqa: S404 -- spawns a real writer to SIGKILL
import sys
import time
from pathlib import Path

import logmint
from logmint._identity import Config
from logmint._run import records
from tests.conftest import base_config

_CHILD = """
import sys, time
import logmint
run = logmint.init(sys.argv[1], {"method": "scrub", "seed": 99})
step = 0
while True:
    run.metric("acc", 0.5, step=step, split="forget")
    step += 1
"""


def _kill_a_writer_mid_stream(root: Path) -> Path:
    """Start a writer, let it run, and SIGKILL it.

    Args:
        root: The corpus root.

    Returns:
        The attempt file the killed process left behind.

    """
    child = subprocess.Popen(  # noqa: S603 -- fixed argv, our own interpreter
        [sys.executable, "-c", _CHILD, str(root)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    directory = root / "runs" / logmint.run_id({"method": "scrub", "seed": 99})
    path = directory / "events.00.jsonl"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size > 100_000:
            break
        time.sleep(0.05)
    child.send_signal(signal.SIGKILL)
    child.wait(timeout=30)
    return path


def test_sigkill_leaves_a_readable_corpus(root: Path) -> None:
    """A killed writer costs at most the line it was writing, and never the run."""
    path = _kill_a_writer_mid_stream(root)
    assert path.stat().st_size > 0

    parsed = records(path)
    metrics = [r for r in parsed if r.get("kind") == "metric"]
    assert len(metrics) > 100
    assert [r["step"] for r in metrics] == list(range(len(metrics)))

    raw = path.read_text(encoding="utf-8").splitlines()
    assert (
        len(raw) - len(parsed) <= 1
    )  # at most one line failed to parse, and it is the last


def test_a_crashed_run_is_not_a_corpus_error(root: Path) -> None:
    """Preemption is expected. verify reports corruption, not interruption."""
    _kill_a_writer_mid_stream(root)
    assert [p for p in logmint.verify(root) if p.level == "error"] == []
    assert logmint.census(root) == {
        "runs": 1,
        "finished": 0,
        "failed": 0,
        "incomplete": 1,
    }


def test_a_crashed_run_never_reaches_a_figure(root: Path) -> None:
    """A crashed run is excluded from the frame and counted, not dropped quietly."""
    _kill_a_writer_mid_stream(root)
    assert logmint.load(root).is_empty()


def _crash_after(root: Path, config: Config, steps: list[int]) -> None:
    """Write an attempt that dies mid-line, exactly as a SIGKILL leaves it.

    Args:
        root: The corpus root.
        config: The run config.
        steps: The steps to log before the interruption.

    """
    run = logmint.init(root, config)
    for step in steps:
        run.metric("acc", 0.5 + step / 1000, step=step, split="forget")
    path = run.dir / f"events.{run.attempt:02d}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"kind":"metric","step":999,"split":"forg')


def test_a_truncated_tail_line_is_dropped(root: Path) -> None:
    """The reader tolerates the one corruption a crash can produce."""
    config = base_config(seed=20)
    _crash_after(root, config, [100, 200])
    with logmint.init(root, config) as run:
        run.metric("acc", 0.3, step=300, split="forget")

    frame = logmint.load(root, names=["acc"]).sort("step")
    assert frame["step"].to_list() == [100, 200, 300]
    assert 999 not in frame["step"].to_list()


def test_resume_dedups_overlapping_steps(root: Path) -> None:
    """A job resumed from a checkpoint re-logs steps; the later attempt wins."""
    config = base_config(seed=21)
    _crash_after(root, config, [100, 200])

    previous = logmint.last_blob(root, config, "checkpoint")
    assert previous is None

    with logmint.init(root, config, resumed_from=100) as run:
        assert run.attempt == 1
        run.metric("acc", 0.11, step=200, split="forget")  # overlaps attempt 0
        run.metric("acc", 0.12, step=300, split="forget")

    frame = logmint.load(root, names=["acc"]).sort("step")
    assert frame["step"].to_list() == [100, 200, 300]
    assert frame.filter(step=100)["attempt"].to_list() == [0]
    assert frame.filter(step=200)["attempt"].to_list() == [1]
    assert frame.filter(step=200)["value"].to_list() == [0.11]
