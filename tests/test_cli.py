"""The command line interface (spec section 8)."""

from pathlib import Path

import numpy as np
import pytest

import logmint
from logmint.cli import main
from tests.conftest import base_config, finished_run


def test_ls_lists_runs_with_their_status(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The corpus census is one command away."""
    finished_run(root, seed=0)
    logmint.init(root, base_config(seed=1)).metric("acc", 0.5, step=1)
    assert main(["--root", str(root), "ls"]) == 0
    out = capsys.readouterr().out
    assert "finished" in out
    assert "incomplete" in out
    assert '"runs": 2' in out


def test_ls_filters_by_status(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Finding what still has to run does not need an index."""
    finished_run(root, seed=0)
    logmint.init(root, base_config(seed=1)).metric("acc", 0.5, step=1)
    main(["--root", str(root), "ls", "--status", "incomplete"])
    out = capsys.readouterr().out
    assert out.count("incomplete") >= 1
    assert "finished" not in out.split('{"runs"')[0]


def test_show_prints_config_provenance_and_attempts(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run explains itself without a name table."""
    rid = finished_run(root)
    assert main(["--root", str(root), "show", rid]) == 0
    out = capsys.readouterr().out
    assert rid in out
    assert '"method":"scrub"' in out
    assert "attempt 0" in out
    assert "status=finished" in out


def test_show_reports_an_unknown_run(root: Path) -> None:
    """Asking for a run that is not there is an error, not an empty page."""
    (root / "runs").mkdir(parents=True)
    assert main(["--root", str(root), "show", "0" * 16]) == 1


def test_query_runs_sql(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The three views are defined for the query, and nothing is exported first."""
    finished_run(root)
    assert (
        main(["--root", str(root), "query", "SELECT count(*) AS n FROM metrics"]) == 0
    )
    assert "2" in capsys.readouterr().out


def test_verify_exits_nonzero_on_an_error(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verification is what runs before you make a figure."""
    rid = finished_run(root)
    assert main(["--root", str(root), "verify"]) == 0

    (root / "runs" / rid / "run.json").write_text(
        '{"method":"other"}', encoding="utf-8"
    )
    assert main(["--root", str(root), "verify"]) == 1
    assert "error" in capsys.readouterr().out


def test_verify_passes_on_a_warning(root: Path) -> None:
    """A warning is reported without failing the check."""
    config = base_config()
    logmint.init(root, config).finish()
    directory = root / "runs" / logmint.run_id(config)
    for attempt, commit in enumerate(["cafebabe", "deadbeef"]):
        (directory / f"events.{attempt:02d}.jsonl").write_text(
            f'{{"kind":"start","time":1,"git":"{commit}"}}\n'
            '{"kind":"status","status":"finished","wall_s":1,"coords":{}}\n',
            encoding="utf-8",
        )
    assert [p.level for p in logmint.verify(root)] == ["warning"]
    assert main(["--root", str(root), "verify"]) == 0


def test_gc_reports_and_removes(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Unreachable blobs are deleted; referenced ones are not."""
    with logmint.init(root, base_config()) as run:
        kept = run.blob("eigvals", np.ones(3), step=1)
    orphan = logmint.put(root, b"orphan")

    assert main(["--root", str(root), "gc", "--dry-run", "--grace", "0"]) == 0
    assert "would remove" in capsys.readouterr().out
    assert main(["--root", str(root), "gc", "--grace", "0"]) == 0
    assert "1 unreachable blob(s)" in capsys.readouterr().out

    assert logmint.get(root, kept) is not None
    with pytest.raises(logmint.BlobError, match="does not resolve"):
        logmint.get(root, orphan)


def test_schema_writes_the_agent_entry_point(root: Path) -> None:
    """The file an agent reads before touching anything."""
    finished_run(root)
    assert main(["--root", str(root), "schema"]) == 0
    assert (root / "SCHEMA.md").is_file()


def test_gc_leaves_a_fresh_blob_alone(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The default grace period is what makes gc safe to run against a live corpus."""
    ref = logmint.put(root, np.arange(3))
    assert main(["--root", str(root), "gc"]) == 0
    assert "0 unreachable" in capsys.readouterr().out
    assert logmint.get_path(root, ref).is_file()
