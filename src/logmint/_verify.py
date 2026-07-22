"""The invariants (spec section 9).

``verify`` is what you run before you make a figure. Five of the six are errors;
the sixth, a run whose attempts disagree on the commit, is a warning, because
resuming across an unrelated bugfix is legitimate and resuming across a change to
the loss is not, and only you can tell which.
"""

import hashlib
import json
import operator
from pathlib import Path
from typing import NamedTuple

from logmint import _blobs
from logmint._errors import ConfigError
from logmint._identity import canonical, run_id, validate
from logmint._run import attempt_index, attempt_paths, records

ERROR = "error"
WARNING = "warning"
_CHUNK = 1 << 20
_METRIC_CORE = frozenset({"kind", "step", "split", "name", "value", "nonfinite"})


class Problem(NamedTuple):
    """One violated invariant."""

    level: str
    run: str
    message: str

    def __str__(self) -> str:
        """Render the problem for a terminal.

        Returns:
            One line, prefixed by level and run.

        """
        return f"{self.level:<7} {self.run:<16} {self.message}"


def _key(record: dict[str, object]) -> tuple[object, ...]:
    """Build the deduplication key of a metric record.

    Args:
        record: A metric record.

    Returns:
        Its key: name, step, split, and every coordinate.

    """
    coords = {
        k: v for k, v in record.items() if k not in {"kind", "value", "nonfinite"}
    }
    return tuple(sorted(coords.items(), key=operator.itemgetter(0)))


def _digest_file(path: Path) -> str:
    """Hash a file's contents.

    Args:
        path: The file.

    Returns:
        Its sha256, hex encoded.

    """
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            hasher.update(chunk)
    return hasher.hexdigest()


def _check_identity(directory: Path) -> list[Problem]:
    """Check that the directory name is the hash of the config inside it.

    Args:
        directory: The run directory.

    Returns:
        Any problems found.

    """
    path = directory / "run.json"
    if not path.is_file():
        return [Problem(ERROR, directory.name, "no run.json")]
    text = path.read_text(encoding="utf-8")
    try:
        config = json.loads(text)
    except ValueError as err:
        return [Problem(ERROR, directory.name, f"run.json does not parse: {err}")]
    try:
        validate(config)
    except ConfigError as err:
        # A hand-written or migrated config can be anything. Diagnosing that is the job.
        return [
            Problem(ERROR, directory.name, f"run.json is not a valid config: {err}")
        ]
    if canonical(config) != text:
        return [Problem(ERROR, directory.name, "run.json is not in canonical form")]
    if run_id(config) != directory.name:
        return [
            Problem(ERROR, directory.name, "directory name is not the hash of run.json")
        ]
    return []


def _config_keys(directory: Path) -> set[str]:
    """Read the config's keys, which no coordinate may shadow.

    Args:
        directory: The run directory.

    Returns:
        The keys, or an empty set if there is no readable config.

    """
    path = directory / "run.json"
    if not path.is_file():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except ValueError:
        return set()


def _declaration(
    parsed: list[dict[str, object]],
) -> tuple[dict[str, object] | None, bool]:
    """Extract the coordinate declaration of a terminated attempt.

    Args:
        parsed: The records of an attempt file.

    Returns:
        The declared coordinates, and whether the attempt terminated at all.

    """
    status = next((r for r in parsed if r.get("kind") == "status"), None)
    if status is None:
        return None, False
    declared = status.get("coords")
    if not isinstance(declared, dict):
        return None, True
    return {str(key): value for key, value in declared.items()}, True


def _check_attempts(directory: Path) -> list[Problem]:
    """Check a run's attempt files against the stream invariants.

    Attempt indices must be contiguous from zero, each terminal status must sit
    last, no metric key may repeat, and the attempts must agree on the commit.

    Args:
        directory: The run directory.

    Returns:
        Any problems found.

    """
    problems: list[Problem] = []
    paths = attempt_paths(directory)
    rid = directory.name
    keys = _config_keys(directory)

    indices = [attempt_index(p) for p in paths]
    if indices != list(range(len(indices))):
        problems.append(
            Problem(ERROR, rid, f"attempt indices are not contiguous from 0: {indices}")
        )

    commits: set[str] = set()
    for path in paths:
        parsed = records(path)  # the one parse of this file; every check below reads it
        problems.extend(_check_attempt(rid, path, parsed, keys))
        commits.update(
            str(r["git"])
            for r in parsed
            if r.get("kind") == "start" and isinstance(r.get("git"), str)
        )

    if len(commits) > 1:
        problems.append(
            Problem(WARNING, rid, f"attempts disagree on the commit: {sorted(commits)}")
        )
    return problems


def _check_attempt(
    rid: str,
    path: Path,
    parsed: list[dict[str, object]],
    keys: set[str],
) -> list[Problem]:
    """Check one attempt file against the invariants of a record stream.

    Args:
        rid: The run id.
        path: The attempt file.
        parsed: Its records, parsed once.
        keys: The config's keys, which no coordinate may shadow.

    Returns:
        Any problems found.

    """
    problems: list[Problem] = []

    # The read frame is the join of the config onto the record, so the two
    # namespaces are one. A corpus that breaks that arrives in the frame with two
    # columns of one name.
    declared, _ = _declaration(parsed)
    shadowed = sorted(keys & set(declared or {}))
    if shadowed:
        problems.append(
            Problem(
                ERROR,
                rid,
                f"{path.name} uses config keys {shadowed} as coordinates; "
                f"the read frame joins the two",
            )
        )
    problems.extend(_check_declaration(rid, path, parsed))

    statuses = [i for i, r in enumerate(parsed) if r.get("kind") == "status"]
    if len(statuses) > 1:
        problems.append(
            Problem(ERROR, rid, f"{path.name} holds {len(statuses)} status records")
        )
    elif statuses and statuses[0] != len(parsed) - 1:
        problems.append(
            Problem(ERROR, rid, f"{path.name} has a status record before the end")
        )

    seen: set[tuple[object, ...]] = set()
    for record in parsed:
        if record.get("kind") != "metric":
            continue
        key = _key(record)
        if key in seen:
            problems.append(
                Problem(ERROR, rid, f"{path.name} logs a duplicate metric key: {key}")
            )
        seen.add(key)
    return problems


def _check_declaration(
    rid: str, path: Path, parsed: list[dict[str, object]]
) -> list[Problem]:
    """Check that a terminated attempt declared the coordinates it actually used.

    The reader discovers the schema from this declaration rather than by scanning
    every line, so a declaration that disagrees with the records would silently
    cost a column.

    Args:
        rid: The run id.
        path: The attempt file.
        parsed: Its records.

    Returns:
        Any problems found.

    """
    declared, terminated = _declaration(parsed)
    if not terminated:
        return []
    if declared is None:
        return [
            Problem(
                ERROR, rid, f"{path.name} terminated without declaring its coordinates"
            )
        ]
    used = {
        key
        for record in parsed
        if record.get("kind") == "metric"
        for key in record
        if key not in _METRIC_CORE
    }
    if used != set(declared):
        return [
            Problem(
                ERROR,
                rid,
                f"{path.name} declares coordinates {sorted(declared)} "
                f"but uses {sorted(used)}",
            )
        ]
    return []


def _check_blobs(root: Path, *, rehash: bool) -> list[Problem]:
    """Check that every reference resolves, optionally re-hashing the bytes.

    Args:
        root: The corpus root.
        rehash: Re-read every blob and check its content against its name.

    Returns:
        Any problems found.

    """
    problems: list[Problem] = []
    for ref in sorted(_blobs.reachable(root)):
        path = _blobs.path_for(root, ref)
        if not path.is_file():
            problems.append(Problem(ERROR, "-", f"dangling reference: {ref}"))
        elif rehash and _digest_file(path) != _blobs.digest_of(ref):
            problems.append(
                Problem(ERROR, "-", f"blob content does not match its name: {ref}")
            )
    return problems


def verify(root: Path | str, *, blobs: bool = False) -> list[Problem]:
    """Check every invariant of a corpus.

    Args:
        root: The corpus root.
        blobs: Re-hash blob contents rather than only checking that they resolve.

    Returns:
        Every problem found, errors and warnings together, in run order.

    """
    root = Path(root)
    problems: list[Problem] = []
    directory = root / "runs"
    if directory.is_dir():
        for run in sorted(p for p in directory.iterdir() if p.is_dir()):
            problems.extend(_check_identity(run))
            problems.extend(_check_attempts(run))
    problems.extend(_check_blobs(root, rehash=blobs))
    return problems
