"""The writer: run directories, attempt files, and records (spec sections 2 and 3).

An attempt file is created with ``O_CREAT | O_EXCL`` at the lowest free index,
which makes mutual exclusion a byproduct of creating the file: two processes
holding the same config never interleave into one stream, and there is no lock to
go stale when a job is preempted. One file per attempt also confines corruption to
a single line, because a SIGKILL can only cut the line being written and that line
is always the last one in its file.
"""

import contextlib
import json
import math
import os
import time
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import TracebackType
from typing import NamedTuple, Self, TextIO

import numpy as np

from logmint import _blobs, _provenance
from logmint._blobs import Blobbable
from logmint._errors import AlreadyFinishedError, CollisionError, RecordError
from logmint._identity import IDENT, RESERVED, Config, canonical, run_id, validate

type Coordinate = str | int | float | bool
type Record = dict[str, object]

ATTEMPT_GLOB = "events.*.jsonl"
BLOB_KIND = '"kind":"blob"'
"""What the writer emits for a blob record, used to skip lines without parsing them."""

_COORD_TYPES: dict[type, str] = {
    bool: "BOOLEAN",
    str: "VARCHAR",
    int: "BIGINT",
    float: "DOUBLE",
}
_NUMERIC = ("BIGINT", "DOUBLE")


def _widen(seen: str, kind: str, key: str) -> str:
    """Reconcile two types seen for one coordinate.

    An integer and a real number are the same column, because a coordinate written
    0 and later 0.5 is one axis. Anything else is two axes wearing one name.

    Args:
        seen: The type recorded so far.
        kind: The type just written.
        key: The coordinate name, for the error message.

    Returns:
        The type that holds both.

    Raises:
        RecordError: The two types are not both numeric.

    """
    if seen in _NUMERIC and kind in _NUMERIC:
        return "DOUBLE"
    msg = (
        f"coordinate {key!r} was written as {seen} and is now {kind}; "
        "a column has one type"
    )
    raise RecordError(msg)


_ATTEMPT_PAD = 2
_TAIL_BYTES = 1 << 16
_FINISHED = "finished"
_FAILED = "failed"


class BlobRecord(NamedTuple):
    """A blob written by an earlier attempt."""

    step: int | None
    ref: str


def run_dir(root: Path, rid: str) -> Path:
    """Return the directory of a run.

    Args:
        root: The corpus root.
        rid: The run id.

    Returns:
        The run directory, which may not exist.

    """
    return root / "runs" / rid


def attempt_paths(directory: Path) -> list[Path]:
    """List a run's attempt files in numeric index order.

    Args:
        directory: The run directory.

    Returns:
        The attempt files, ordered by index rather than lexicographically.

    """
    if not directory.is_dir():
        return []
    return sorted(directory.glob(ATTEMPT_GLOB), key=attempt_index)


def attempt_index(path: Path) -> int:
    """Extract the attempt index from an attempt file name.

    Args:
        path: An ``events.<NN>.jsonl`` path.

    Returns:
        The index as an integer.

    """
    return int(path.name.split(".")[1])


def records(path: Path) -> list[Record]:
    """Parse an attempt file, dropping any line that does not parse.

    At most one line per file fails to parse, and it is always the last, because a
    SIGKILL can only cut the line being written.

    Args:
        path: The attempt file.

    Returns:
        The records it holds.

    """
    out: list[Record] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                out.append(record)
    return out


def tail_line(path: Path) -> str:
    """Read the last complete line of a file without reading the whole file.

    Args:
        path: The file to read.

    Returns:
        The last line, or an empty string if there is none.

    """
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(max(0, size - _TAIL_BYTES))
        lines = handle.read().splitlines()
    return lines[-1].decode("utf-8", errors="replace") if lines else ""


def _finished_attempt(path: Path) -> bool:
    """Report whether an attempt file ends in a ``finished`` status.

    The status record is always the last line of its file, so this reads the tail only.

    Args:
        path: The attempt file.

    Returns:
        True if the attempt recorded a terminal ``finished`` status.

    """
    try:
        record = json.loads(tail_line(path))
    except ValueError:
        return False
    return (
        isinstance(record, dict)
        and record.get("kind") == "status"
        and record.get("status") == _FINISHED
    )


def is_finished(root: Path, rid: str) -> bool:
    """Report whether any attempt of a run recorded a ``finished`` status.

    Args:
        root: The corpus root.
        rid: The run id.

    Returns:
        True if the run is complete.

    """
    return any(_finished_attempt(path) for path in attempt_paths(run_dir(root, rid)))


def done(root: Path | str, config: Config) -> bool:
    """Report whether this config has already finished, for idempotent resubmission.

    This is a stat and a tail read, so a sweep script can call it for every config
    without building an index.

    Args:
        root: The corpus root.
        config: The config to look for.

    Returns:
        True if a finished run of that config exists.

    """
    return is_finished(Path(root), run_id(config))


def _blob_records(path: Path) -> Iterator[Record]:
    """Yield the blob records of an attempt file without parsing the rest.

    A resume calls this on the file the crashed attempt left behind, which can be
    hundreds of megabytes of metric records. Parsing all of them to find a
    checkpoint reference would cost minutes; testing for a substring first costs
    nothing, and cannot give a wrong answer, because a line that passes the test is
    still parsed and still has to say it is a blob record.

    Args:
        path: The attempt file.

    Yields:
        Its blob records, in order.

    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if BLOB_KIND not in line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and record.get("kind") == "blob":
                yield record


def _last_blob_in(paths: Iterable[Path], name: str) -> BlobRecord | None:
    """Find the last blob of a given name across a sequence of attempt files.

    Args:
        paths: The attempt files, in index order.
        name: The blob name.

    Returns:
        The last matching blob, or None.

    """
    found: BlobRecord | None = None
    for path in paths:
        for record in _blob_records(path):
            if record.get("name") == name:
                step = record.get("step")
                found = BlobRecord(
                    step=step if isinstance(step, int) else None, ref=str(record["ref"])
                )
    return found


def last_blob(root: Path | str, config: Config, name: str) -> BlobRecord | None:
    """Find the most recent blob of a given name across all attempts of a run.

    Callable before ``init``, which is what lets a resuming job resolve its
    checkpoint and pass the step it resumes from into the ``start`` record of the
    new attempt.

    Args:
        root: The corpus root.
        config: The config identifying the run.
        name: The blob name, such as ``checkpoint``.

    Returns:
        The last matching blob, or None if the run never wrote one.

    """
    return _last_blob_in(attempt_paths(run_dir(Path(root), run_id(config))), name)


def _pythonize(value: object) -> object:
    """Reduce a numpy scalar, a zero-dimensional array or a tensor to a Python scalar.

    ``np.float64`` happens to subclass ``float`` and ``np.float32`` does not, so a
    writer that tests for ``float`` accepts a metric from a float64 array and
    rejects the same metric from a float32 one. Every deep learning loop produces
    the second. The ``item`` protocol is what numpy and torch both expose to get a
    Python scalar out, and an array that holds more than one number raises from it,
    which is the answer anyway.

    Args:
        value: Whatever the caller passed.

    Returns:
        A Python scalar if the value could produce one, and the value itself otherwise.

    """
    item = getattr(value, "item", None)
    if item is None or isinstance(value, str | bytes):
        return value
    try:
        return item()
    except (TypeError, ValueError):
        return value


def _open_attempt(directory: Path) -> tuple[int, TextIO]:
    """Create the lowest free attempt file, exclusively.

    Args:
        directory: The run directory.

    Returns:
        The attempt index and a line-buffered text handle onto the new file.

    """
    index = 0
    while True:
        path = directory / f"events.{index:0{_ATTEMPT_PAD}d}.jsonl"
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            index += 1
            continue
        return index, os.fdopen(fd, "w", buffering=1, encoding="utf-8")


def _write_config(directory: Path, config: Config) -> None:
    """Write ``run.json`` once, or check that the existing one is the same config.

    Args:
        directory: The run directory.
        config: The config to write.

    Raises:
        CollisionError: A different config already occupies this directory.

    """
    path = directory / "run.json"
    text = canonical(config)
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        if existing != text:
            msg = (
                f"run id collision in {directory.name}: the directory holds a "
                f"different config. stored={existing} incoming={text}"
            )
            raise CollisionError(msg)
        return
    tmp = directory / f".run.json.{uuid.uuid4().hex}"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)  # noqa: PTH105 -- atomic rename, so a half-written config is never seen


class Run:
    """An open attempt: one writer, one append-only stream."""

    def __init__(
        self,
        root: Path,
        rid: str,
        attempt: int,
        handle: TextIO,
        keys: frozenset[str],
    ) -> None:
        """Bind an open attempt file.

        Args:
            root: The corpus root.
            rid: The run id.
            attempt: The attempt index.
            handle: A line-buffered text handle onto the attempt file.
            keys: The config's keys, which coordinates may not shadow.

        """
        self.root = root
        self.run_id = rid
        self.attempt = attempt
        self.dir = run_dir(root, rid)
        self.path = self.dir / f"events.{attempt:0{_ATTEMPT_PAD}d}.jsonl"
        self._handle = handle
        self._keys = keys
        self._pid = os.getpid()
        self._monotonic = time.monotonic()
        self._closed = False
        self._coords: dict[str, str] = {}

    def __enter__(self) -> Self:
        """Enter the run context.

        Returns:
            This run.

        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Record a terminal status on the way out.

        A crash is labelled rather than omitted.

        Args:
            exc_type: The exception type, if the block raised.
            exc: The exception, if the block raised.
            tb: The traceback, if the block raised.

        """
        del exc, tb
        if self._closed:
            return
        if exc_type is None:
            self.finish()
        else:
            self.fail(exc_type.__name__)

    def _emit(self, record: Record) -> None:
        """Append one record to the attempt file.

        Args:
            record: The record to write.

        Raises:
            RecordError: The attempt is closed, or the record holds a value JSON
                cannot represent.

        """
        if self._closed:
            msg = f"run {self.run_id} attempt {self.attempt} is closed"
            raise RecordError(msg)
        if os.getpid() != self._pid:
            msg = (
                f"run {self.run_id} was opened in process {self._pid} and is "
                f"being written from {os.getpid()}. A fork inherits the file "
                "descriptor, so both processes would append to one file and "
                "interleave. Open the run after the fork, or log from one rank "
                "only"
            )
            raise RecordError(msg)
        try:
            line = json.dumps(
                record, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
        except (TypeError, ValueError) as err:
            msg = f"record holds a value JSON cannot represent: {err}"
            raise RecordError(msg) from err
        self._handle.write(line + "\n")

    def _start(
        self, runtime: dict[str, object] | None, resumed_from: int | None
    ) -> None:
        """Emit the ``start`` record that opens every attempt.

        Args:
            runtime: Knobs that do not belong to identity, such as worker count.
            resumed_from: The step this attempt picks up from, if any.

        """
        record: Record = {"kind": "start", "time": time.time()}
        record.update(_provenance.collect())
        record["resumed_from"] = resumed_from
        record["runtime"] = runtime or {}
        self._emit(record)

    @staticmethod
    def _check_extra(
        record: Record,
        extra: dict[str, Coordinate],
        what: str,
    ) -> dict[str, Coordinate]:
        """Check the keys a caller supplied as ``**kwargs`` on a record.

        A record is a flat row of scalars, and nothing a caller passes may take
        over a field the record already carries. Without this,
        ``run.event("x", kind="metric", value=9)`` writes a metric record and
        ``run.blob(..., ref="blob:0")`` writes a blob record that points somewhere
        else, which makes the real blob unreachable and hands it to the collector.

        Args:
            record: The record built so far, whose fields are already spoken for.
            extra: The keys the caller supplied.
            what: What to call them in an error message.

        Returns:
            The same keys with numpy scalars reduced to Python ones.

        Raises:
            RecordError: A key is reserved, shadows a field of the record, is not
                an identifier, or its value is not a finite scalar.

        """
        checked: dict[str, Coordinate] = {}
        for key, given in extra.items():
            if key in record:
                msg = (
                    f"{what} {key!r} would overwrite the {record['kind']} "
                    f"record's own {key!r}"
                )
                raise RecordError(msg)
            if key in RESERVED:
                msg = f"{what} {key!r} is a reserved name of a read-frame column"
                raise RecordError(msg)
            if not IDENT.fullmatch(key):
                msg = f"{what} {key!r} is not an identifier, so it cannot be a column"
                raise RecordError(msg)
            value = _pythonize(given)
            if not isinstance(value, str | int | float | bool):
                msg = f"{what} {key!r} must be a scalar, got {type(given).__name__}"
                raise RecordError(msg)
            if isinstance(value, float) and not math.isfinite(value):
                msg = (
                    f"{what} {key!r} is non-finite; it indexes a result, so it "
                    "cannot be NaN"
                )
                raise RecordError(msg)
            checked[key] = value
        return checked

    def _check_coords(
        self,
        record: Record,
        coords: dict[str, Coordinate],
    ) -> dict[str, Coordinate]:
        """Check the coordinates of a metric record.

        Args:
            record: The metric record built so far.
            coords: The coordinates supplied with it.

        Returns:
            The coordinates, with numpy scalars reduced to Python ones.

        Raises:
            RecordError: A coordinate is invalid, or shadows a config key. The read
                frame joins the config onto the record, so the two namespaces are
                one.

        """
        checked = self._check_extra(record, coords, "coordinate")
        shadowed = sorted(set(coords) & self._keys)
        if shadowed:
            msg = (
                f"coordinate(s) {shadowed} are also config keys. The read frame "
                "joins the config onto the record, so the corpus would carry two "
                "columns of each name and no way to tell them apart; rename the "
                "coordinate"
            )
            raise RecordError(msg)
        return checked

    @staticmethod
    def _check_core(name: str, step: object, split: str | None) -> None:
        """Check the pinned fields of a metric record against the columns they land in.

        Args:
            name: The metric name.
            step: The step, already reduced to a Python scalar.
            split: The split.

        Raises:
            RecordError: A field does not match the type of its column.

        """
        if not isinstance(name, str) or not name:
            msg = f"metric name must be a non-empty string, got {name!r}"
            raise RecordError(msg)
        if step is not None and (isinstance(step, bool) or not isinstance(step, int)):
            msg = (
                f"step must be an integer, got {type(step).__name__}; "
                "the column is BIGINT"
            )
            raise RecordError(msg)
        if split is not None and not isinstance(split, str):
            msg = (
                f"split must be a string, got {type(split).__name__}; "
                "the column is VARCHAR"
            )
            raise RecordError(msg)

    def metric(
        self,
        name: str,
        value: float,
        *,
        step: int | None = None,
        split: str | None = None,
        **coords: Coordinate,
    ) -> None:
        """Record one scalar result.

        A non-finite value is written as ``null`` with a ``nonfinite`` tag, which
        leaves a gap in the curve exactly where the divergence happened and keeps
        every line valid JSON. RFC 8259 has no ``NaN`` token, and a log that only
        some parsers can read is not a log.

        Args:
            name: The metric name.
            value: A real number. Booleans are rejected, being neither measurement
                nor coordinate.
            step: The step, omitted for once-per-run results.
            split: The data split, if the metric has one.
            **coords: Any further keys, which index the result.

        Raises:
            RecordError: The value is not a real number, or a coordinate is invalid.

        """
        number = _pythonize(value)
        if isinstance(number, bool) or not isinstance(number, int | float):
            msg = (
                f"metric {name!r} must be a real number, got {type(value).__name__}; "
                f"if it is a tensor, pass what .item() gives you"
            )
            raise RecordError(msg)
        at = _pythonize(step)
        self._check_core(name, at, split)

        record: Record = {"kind": "metric"}
        if at is not None:
            record["step"] = at
        if split is not None:
            record["split"] = split
        record["name"] = name
        if math.isfinite(number):
            record["value"] = float(number)
        else:
            record["value"] = None
            sign = "+inf" if number > 0 else "-inf"
            record["nonfinite"] = "nan" if math.isnan(number) else sign
        checked = self._check_coords(record, coords)
        record.update(checked)
        self._declare(checked)
        self._emit(record)

    def _declare(self, coords: dict[str, Coordinate]) -> None:
        """Record the type of each coordinate seen.

        The reader never has to scan for it. This costs one entry per distinct
        coordinate name, not one per record, and turns discovery into a tail read
        of each attempt file rather than a pass over every line of the corpus.

        Args:
            coords: The coordinates of one metric record.

        """
        for key, value in coords.items():
            kind = _COORD_TYPES[type(value)]
            seen = self._coords.setdefault(key, kind)
            if seen != kind:
                # An interpolation coordinate is written 0, then 0.5. One column,
                # one type.
                self._coords[key] = _widen(seen, kind, key)

    def blob(
        self,
        name: str,
        obj: Blobbable,
        *,
        step: int | None = None,
        **meta: Coordinate,
    ) -> str:
        """Write an output blob and record the reference to it.

        Args:
            name: The blob name, such as ``checkpoint``.
            obj: An array, raw bytes, or a path to a file already on disk.
            step: The step it belongs to.
            **meta: Extra metadata to record alongside the reference.

        Returns:
            The blob reference.

        """
        ref = _blobs.put(self.root, obj)
        record: Record = {"kind": "blob"}
        if step is not None:
            record["step"] = step
        record["name"] = name
        record["ref"] = ref
        record["format"] = _blobs.format_of(obj)
        record["bytes"] = _blobs.get_path(self.root, ref).stat().st_size
        if isinstance(obj, np.ndarray):
            record["shape"] = list(obj.shape)
            record["dtype"] = str(obj.dtype)
        record.update(self._check_extra(record, meta, "blob metadata"))
        self._emit(record)
        return ref

    def event(
        self, name: str, *, step: int | None = None, **fields: Coordinate
    ) -> None:
        """Record a runtime fact worth querying, keeping the text log free of results.

        Args:
            name: The event name, such as ``grad_overflow``.
            step: The step it happened at.
            **fields: Any further keys.

        """
        record: Record = {"kind": "event"}
        if step is not None:
            record["step"] = step
        record["name"] = name
        record.update(self._check_extra(record, fields, "event field"))
        self._emit(record)

    def last_blob(self, name: str) -> BlobRecord | None:
        """Find the most recent blob of a given name written by an earlier attempt.

        Args:
            name: The blob name.

        Returns:
            The last matching blob, or None.

        """
        earlier = [
            p for p in attempt_paths(self.dir) if attempt_index(p) < self.attempt
        ]
        return _last_blob_in(earlier, name)

    def close(self) -> None:
        """Release the attempt file without terminating the run.

        Closing the descriptor is not the same as finishing the run: no status
        record is written, so the attempt stays incomplete, which is what it is.
        This exists because a driver that opens a run, catches an exception and
        moves on would otherwise hold the descriptor until the process exits, and a
        sweep that does that a few thousand times runs out of them.
        """
        if self._closed:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._closed = True

    def __del__(self) -> None:
        """Release the attempt file if the caller never did."""
        with contextlib.suppress(Exception):
            self.close()

    def _terminate(self, status: str, exc: str | None) -> None:
        """Emit the terminal status record and close the file.

        Args:
            status: Either ``finished`` or ``failed``.
            exc: The exception type name, for a failure.

        """
        record: Record = {"kind": "status", "status": status}
        if exc is not None:
            record["exc"] = exc
        record["wall_s"] = round(time.monotonic() - self._monotonic, 3)
        record["coords"] = dict(sorted(self._coords.items()))
        self._emit(record)
        self.close()

    def finish(self) -> None:
        """Record that the run completed.

        This is the last line of the file, and the only thing that distinguishes a
        converged run from one that died at step three.
        """
        self._terminate(_FINISHED, None)

    def fail(self, exc: str) -> None:
        """Record that the run failed.

        Args:
            exc: The exception type name.

        """
        self._terminate(_FAILED, exc)


def init(
    root: Path | str,
    config: Config,
    *,
    runtime: dict[str, object] | None = None,
    resumed_from: int | None = None,
    allow_rerun: bool = False,
) -> Run:
    """Open a new attempt of the run identified by this config.

    Args:
        root: The corpus root.
        config: The config, which is hashed to give the run id.
        runtime: Knobs that do not belong to identity, recorded in the ``start`` record.
        resumed_from: The step this attempt picks up from, recorded but not interpreted.
        allow_rerun: Open an attempt even though the run already finished.

    Returns:
        The open run.

    Raises:
        AlreadyFinishedError: The run already finished and ``allow_rerun`` is false.
        RecordError: The ``runtime`` dict holds a value JSON cannot represent.

    """
    validate(config)
    root = Path(root)
    rid = run_id(config)
    directory = run_dir(root, rid)
    directory.mkdir(parents=True, exist_ok=True)
    _write_config(directory, config)

    if not allow_rerun and is_finished(root, rid):
        msg = (
            f"run {rid} already recorded a finished status; guard your sweep with "
            f"logmint.done(root, config), or pass allow_rerun=True"
        )
        raise AlreadyFinishedError(msg)

    attempt, handle = _open_attempt(directory)
    run = Run(root, rid, attempt, handle, frozenset(config))
    try:
        run._start(runtime, resumed_from)  # noqa: SLF001 -- the start record opens the attempt
    except RecordError:
        # runtime is the caller's dict and can hold anything. An attempt that never
        # recorded its start never began, so it leaves no file to be counted as one.
        run.close()
        run.path.unlink(missing_ok=True)
        raise
    return run
