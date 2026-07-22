"""Provenance capture (spec section 3).

Every probe here is best effort. The reason to test it is that it runs inside
every ``init``, and a training job must not die because git is missing, the repo
is absent, or there is no GPU.
"""

import shutil
import subprocess  # noqa: S404 -- runs real git in a scratch repo
from pathlib import Path

import pytest

import logmint
from logmint import _provenance

_SHA_CHARS = 40
_DIGEST_CHARS = 64
_KEYS = {"git", "dirty", "lock", "host", "gpus", "slurm_job_id"}


def _git(cwd: Path, *args: str) -> None:
    """Run a git command in a scratch repository.

    Args:
        cwd: The repository.
        *args: The arguments to git.

    """
    exe = shutil.which("git")
    assert exe is not None
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "PATH": "/usr/bin:/bin",
    }
    subprocess.run([exe, *args], cwd=cwd, check=True, capture_output=True, env=env)  # noqa: S603


def test_provenance_outside_a_repository_records_nulls_and_does_not_raise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run launched from a directory with no repository still runs."""
    monkeypatch.chdir(tmp_path)
    found = _provenance.collect()
    assert set(found) == _KEYS
    assert found["git"] is None
    assert found["lock"] is None
    assert found["host"]


def test_provenance_with_nothing_on_the_path_does_not_raise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No git, no nvidia-smi, no problem: a missing probe records a null."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")
    found = _provenance.collect()
    assert found["git"] is None
    assert found["gpus"] == []


def test_provenance_records_the_commit_and_hashes_the_uncommitted_diff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The commit says which code ran; the diff hash says it was uncommitted."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "train.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "train.py")
    _git(tmp_path, "commit", "-qm", "first")
    monkeypatch.chdir(tmp_path)

    clean = _provenance.collect()
    assert isinstance(clean["git"], str)
    assert len(clean["git"]) == _SHA_CHARS
    assert clean["dirty"] is None

    (tmp_path / "train.py").write_text("x = 2\n", encoding="utf-8")
    dirty = _provenance.collect()
    assert dirty["git"] == clean["git"]
    assert isinstance(dirty["dirty"], str)
    assert len(dirty["dirty"]) == _DIGEST_CHARS


def test_provenance_hashes_the_lockfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lockfile hash is what makes a number reproducible a year later."""
    monkeypatch.chdir(tmp_path)
    assert _provenance.collect()["lock"] is None

    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    found = _provenance.collect()["lock"]
    assert isinstance(found, str)
    assert len(found) == _DIGEST_CHARS


def test_provenance_records_the_scheduler_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The job id is what connects a run to the scheduler's logs."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "1234567")
    assert _provenance.collect()["slurm_job_id"] == "1234567"


def test_a_run_records_its_provenance_per_attempt(
    root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provenance sits on the attempt: a resubmission can carry a different commit."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    config = {"method": "scrub", "seed": 0}
    with logmint.init(root, config, runtime={"workers": 8}) as run:
        run.metric("acc", 0.5, step=1)

    start = logmint.query(root, "SELECT * FROM events WHERE kind = 'start'")
    assert start["slurm_job_id"][0] == "42"
    assert start["runtime"][0] == {"workers": 8}


def test_a_probe_that_cannot_be_spawned_records_a_null(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cluster whose fork limit is exhausted still starts the run."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError(12, "Cannot allocate memory")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", refuse)
    found = _provenance.collect()
    assert found["git"] is None
    assert found["gpus"] == []
