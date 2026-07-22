"""Provenance for the ``start`` record (spec section 3).

Provenance is per attempt, not per run: a resubmission can carry a different
commit, and a run whose attempts disagree on the commit is something ``verify``
warns about.

Every probe here is best effort. A missing git, a missing GPU, or a missing
lockfile records a null and never interrupts a training run.
"""

import hashlib
import os
import shutil
import socket
import subprocess  # noqa: S404 -- guarded calls to git and nvidia-smi below
from pathlib import Path

type Provenance = dict[str, object]

_TIMEOUT_S = 10
_LOCKFILES = ("uv.lock", "poetry.lock", "requirements.txt")


def _capture(exe: str, *args: str) -> str | None:
    """Run a command and return its stdout, or None if it cannot be run.

    Args:
        exe: The name of the executable to look up on PATH.
        *args: Arguments to pass to it.

    Returns:
        The stripped stdout, or None if the executable is absent or the command failed.

    """
    path = shutil.which(exe)
    if path is None:
        return None
    try:
        done = subprocess.run(  # noqa: S603 -- fixed argv, absolute path from shutil.which
            [path, *args],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _git() -> tuple[str | None, str | None]:
    """Capture the current commit and a hash of the uncommitted diff.

    Returns:
        The commit sha and the sha256 of ``git diff HEAD``, either of which may be None.

    """
    head = _capture("git", "rev-parse", "HEAD")
    if head is None:
        return None, None
    diff = _capture("git", "diff", "HEAD")
    if not diff:
        return head, None
    return head, hashlib.sha256(diff.encode("utf-8")).hexdigest()


def _lockfile(cwd: Path) -> str | None:
    """Hash the first lockfile found in the working directory.

    Args:
        cwd: The directory to look in.

    Returns:
        The sha256 of the lockfile, or None if there is none.

    """
    for name in _LOCKFILES:
        path = cwd / name
        if path.is_file():
            return hashlib.sha256(path.read_bytes()).hexdigest()
    return None


def _gpus() -> list[str]:
    """List the visible GPUs.

    Returns:
        The GPU model names, empty if none are visible or nvidia-smi is absent.

    """
    out = _capture("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
    return [line.strip() for line in out.splitlines() if line.strip()] if out else []


def collect() -> Provenance:
    """Capture the provenance of the current process.

    Returns:
        A mapping with the commit, the dirty-diff hash, the lockfile hash, the
        host, the GPUs, and the scheduler job id, each of which may be null.

    """
    cwd = Path.cwd()
    head, dirty = _git()
    return {
        "git": head,
        "dirty": dirty,
        "lock": _lockfile(cwd),
        "host": socket.gethostname(),
        "gpus": _gpus(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
