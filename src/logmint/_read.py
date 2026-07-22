"""The reader (spec sections 6 and 7).

A run's frame is the concatenation of its attempt files in index order, minus
lines that do not parse, deduplicated on ``(name, step, split, coordinates)``
keeping the highest attempt. A run is complete iff some attempt recorded
``finished``; incomplete runs are excluded from ``load`` and counted, never
dropped quietly.

The schema is always pinned. A terminating attempt declares its coordinates in
its ``status`` record, so discovering them, classifying the run, and finding out
whether it was ever resumed are all answered by one tail read per attempt file,
in a single walk of the corpus.

The only text interpolated into SQL is the corpus path, with quotes escaped, and
column names, which the writer has already validated as identifiers.
"""

import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import NamedTuple

import duckdb
import polars as pl

from logmint._errors import CorpusError
from logmint._run import attempt_paths, run_dir, tail_line

logger = logging.getLogger("logmint")

CORE: dict[str, str] = {
    "kind": "VARCHAR",
    "step": "BIGINT",
    "split": "VARCHAR",
    "name": "VARCHAR",
    "value": "DOUBLE",
    "nonfinite": "VARCHAR",
    "ref": "VARCHAR",
    "status": "VARCHAR",
}
"""The columns every event stream has. Everything else on a metric record is a
coordinate."""

FINISHED = "finished"
FAILED = "failed"
INCOMPLETE = "incomplete"

_METRIC_CORE = frozenset({"kind", "step", "split", "name", "value", "nonfinite"})
_JSON_TYPES = {
    "VARCHAR": "VARCHAR",
    "BOOLEAN": "BOOLEAN",
    "UBIGINT": "BIGINT",
    "BIGINT": "BIGINT",
}
_NUMERIC = ("BIGINT", "DOUBLE")
_DERIVED = frozenset({"run", "attempt", "value", "nonfinite"})


class RunInfo(NamedTuple):
    """What one tail read per attempt file tells you about a run."""

    rid: str
    paths: list[Path]
    coords: dict[str, str]
    undeclared: list[Path]
    status: str


def _quote(path: Path) -> str:
    """Render a path as the body of a SQL string literal.

    Args:
        path: The path.

    Returns:
        The escaped literal body.

    """
    return str(path).replace("'", "''")


def _events_glob(root: Path) -> str:
    """Return the glob matching every attempt file in a corpus.

    Args:
        root: The corpus root.

    Returns:
        The escaped glob pattern.

    """
    return _quote(root / "runs" / "*" / "events.*.jsonl")


def _has_runs(root: Path) -> bool:
    """Report whether the corpus holds any attempt file at all.

    Args:
        root: The corpus root.

    Returns:
        True if at least one attempt file exists.

    """
    return (root / "runs").is_dir() and any((root / "runs").glob("*/events.*.jsonl"))


def _run_ids(root: Path) -> list[str]:
    """List the run ids of a corpus.

    Args:
        root: The corpus root.

    Returns:
        The sorted run ids.

    """
    directory = root / "runs"
    if not directory.is_dir():
        return []
    return sorted(p.name for p in directory.iterdir() if p.is_dir())


def _tail_of(path: Path) -> tuple[dict[str, str] | None, str | None]:
    """Read what a terminated attempt left in its last line.

    Args:
        path: The attempt file.

    Returns:
        The declared coordinates and the terminal status, either of which is None
        if the attempt never terminated, or terminated without declaring.

    """
    try:
        record = json.loads(tail_line(path))
    except ValueError:
        return None, None
    if not isinstance(record, dict) or record.get("kind") != "status":
        return None, None
    declared = record.get("coords")
    coords = (
        {str(k): str(v) for k, v in declared.items()}
        if isinstance(declared, dict)
        else None
    )
    status = record.get("status")
    return coords, str(status) if status is not None else None


def _scan_coords(path: Path) -> dict[str, str]:
    """Recover the coordinates of an attempt that never declared them, by reading it.

    Only an interrupted attempt lands here, so this touches one file rather than
    the corpus.

    Args:
        path: The attempt file.

    Returns:
        The coordinates it uses.

    """
    rows = duckdb.sql(f"""
        SELECT DISTINCT k, json_type(json, '$."' || k || '"') AS t
        FROM (
            SELECT json, unnest(json_keys(json)) AS k
            FROM read_ndjson_objects('{_quote(path)}', ignore_errors=true)
            WHERE json_extract_string(json, '$.kind') = 'metric'
        )
        WHERE json_type(json, '$."' || k || '"') NOT IN ('NULL')
    """).fetchall()  # noqa: S608 -- the only interpolation is the escaped path of one file
    found: dict[str, str] = {}
    for key, kind in rows:
        if key in _METRIC_CORE:
            continue
        mapped = _JSON_TYPES.get(kind, "DOUBLE")
        seen = found.setdefault(key, mapped)
        if seen != mapped:
            found[key] = _widen(seen, mapped, key, path.name)
    return found


def _widen(seen: str, kind: str, key: str, where: str) -> str:
    """Reconcile two types seen for one coordinate.

    An integer and a real number are one column: a coordinate written 0 in one run
    and 0.5 in the next is one axis, and refusing to read that corpus would be
    refusing to read a sweep. Anything else is two axes wearing one name, and there
    is no frame that holds both.

    Args:
        seen: The type recorded so far.
        kind: The type just found.
        key: The coordinate name.
        where: What to name in the error message.

    Returns:
        The type that holds both.

    Raises:
        CorpusError: The two types are not both numeric.

    """
    if seen in _NUMERIC and kind in _NUMERIC:
        return "DOUBLE"
    msg = (
        f"coordinate {key!r} is {seen} elsewhere and {kind} in {where}; it has one type"
    )
    raise CorpusError(msg)


def _merge(into: dict[str, str], found: dict[str, str], where: str) -> None:
    """Merge one attempt's coordinates into the corpus-wide set.

    Args:
        into: The set so far.
        found: The coordinates of one attempt.
        where: What to name in the error message.

    """
    for key, kind in found.items():
        seen = into.setdefault(key, kind)
        if seen != kind:
            into[key] = _widen(seen, kind, key, where)


def info(root: Path | str, rid: str) -> RunInfo:
    """Read the tail of every attempt file of one run.

    Args:
        root: The corpus root.
        rid: The run id.

    Returns:
        What those tails say: the attempt files, the coordinates, and the status.

    """
    paths = attempt_paths(run_dir(Path(root), rid))
    coords: dict[str, str] = {}
    undeclared: list[Path] = []
    status = INCOMPLETE
    for path in paths:
        declared, terminal = _tail_of(path)
        if declared is None:
            undeclared.append(path)
        else:
            _merge(coords, declared, path.name)
        if terminal == FINISHED:
            status = FINISHED
        elif terminal == FAILED and status != FINISHED:
            status = FAILED
    return RunInfo(rid, paths, coords, undeclared, status)


def scan(root: Path | str) -> list[RunInfo]:
    """Walk the corpus once, reading the tail of every attempt file.

    Everything the reader needs to know before it touches a record - the
    coordinate columns to pin, which runs finished, and whether any run was ever
    resumed - comes from this one walk.

    Args:
        root: The corpus root.

    Returns:
        One entry per run, in run id order.

    """
    root = Path(root)
    return [info(root, rid) for rid in _run_ids(root)]


def coords_of(runs: Iterable[RunInfo]) -> dict[str, str]:
    """Collect the coordinate columns of a set of runs.

    An attempt that terminated declared its coordinates, so it costs nothing. An
    attempt that was interrupted has to be read, and it is read only if its run is
    one the caller will actually look at. On a preemptible cluster the incomplete
    runs are the many, and ``load`` excludes them, so it must not pay to scan them.

    Args:
        runs: The runs to cover.

    Returns:
        A mapping from coordinate name to DuckDB type.

    """
    coords: dict[str, str] = {}
    for run in runs:
        _merge(coords, run.coords, run.rid)
        for path in run.undeclared:
            _merge(coords, _scan_coords(path), path.name)
    return coords


def discover(root: Path | str) -> dict[str, str]:
    """Discover the coordinate keys of a corpus and their types.

    Args:
        root: The corpus root.

    Returns:
        A mapping from coordinate name to DuckDB type.

    """
    return coords_of(scan(root))


def status_of(root: Path | str, rid: str) -> str:
    """Classify one run, without walking the rest of the corpus.

    Args:
        root: The corpus root.
        rid: The run id.

    Returns:
        ``finished``, ``failed``, or ``incomplete``.

    """
    return info(root, rid).status


def census(root: Path | str) -> dict[str, int]:
    """Count the runs of a corpus by status.

    Args:
        root: The corpus root.

    Returns:
        Counts keyed by ``runs``, ``finished``, ``failed`` and ``incomplete``.

    """
    counts = {"runs": 0, FINISHED: 0, FAILED: 0, INCOMPLETE: 0}
    for run in scan(root):
        counts["runs"] += 1
        counts[run.status] += 1
    return counts


def _columns_clause(coords: dict[str, str]) -> str:
    """Render the pinned column specification for ``read_json``.

    Args:
        coords: The coordinate types.

    Returns:
        The ``columns={...}`` literal.

    """
    fields = {**CORE, **coords}
    return "{" + ",".join(f"'{k}':'{v}'" for k, v in fields.items()) + "}"


def _metrics_sql(coords: dict[str, str], *, resumed: bool, guard: bool) -> str:
    """Render the ``metrics`` view.

    Deduplication is skipped when no run has more than one attempt, because the
    window provably selects every row and at a million records it is the most
    expensive thing the reader does. The uniqueness guard is skipped only when the
    caller checks the materialised frame instead, which finds the same duplicates
    for the same reason.

    Args:
        coords: The coordinate columns, which are part of a metric's key.
        resumed: Whether any run has more than one attempt.
        guard: Whether the view itself must reject a duplicate key.

    Returns:
        The SQL of the view.

    """
    if not resumed and not guard:
        return """
            SELECT p.* EXCLUDE (kind, ref, status), c.* EXCLUDE (run)
            FROM pinned p JOIN runs c USING (run) WHERE p.kind = 'metric'
        """
    keys = ", ".join(["name", "step", "split", *coords])
    windows, drop, where = [], [], []
    if resumed:
        windows.append(
            f"row_number() OVER (PARTITION BY p.run, {keys} "
            "ORDER BY p.attempt DESC) AS r"
        )
        drop.append("r")
        where.append("r = 1")
    if guard:
        windows.append(f"count(*) OVER (PARTITION BY p.run, p.attempt, {keys}) AS dups")
        drop.append("dups")
        where.append("(dups = 1 OR error('duplicate metric key within one attempt'))")
    return f"""
        SELECT * EXCLUDE ({", ".join(drop)}) FROM (
            SELECT p.* EXCLUDE (kind, ref, status), c.* EXCLUDE (run),
            {", ".join(windows)}
            FROM pinned p JOIN runs c USING (run) WHERE p.kind = 'metric'
        ) WHERE {" AND ".join(where)}
    """  # noqa: S608 -- interpolates only column names the writer validated as identifiers


def _views(
    con: duckdb.DuckDBPyConnection,
    root: Path,
    coords: dict[str, str],
    *,
    resumed: bool,
    guard: bool,
    events: bool,
) -> None:
    """Define the ``pinned``, ``events``, ``runs`` and ``metrics`` views over a corpus.

    Two reads of the same files, because they answer different questions.
    ``pinned`` fixes the schema and feeds the tidy frame, which is the hot path and
    the one that has to stay fast. ``events`` infers the schema and so carries
    everything the pinned read drops: the provenance on a start record, the fields
    on an event, the shape and dtype on a blob.

    DuckDB binds a view when it is created, not when it is queried, so the
    inference is paid by whoever defines the view whether or not anything reads it.
    It is therefore defined only when the caller's SQL names it, which it must do
    to select from it.

    Args:
        con: The connection to define them on.
        root: The corpus root.
        coords: The coordinate types to pin.
        resumed: Whether any run has more than one attempt.
        guard: Whether the metrics view must itself reject a duplicate key.
        events: Whether to define the full-fidelity events view.

    """
    glob = _events_glob(root)
    run_from = "regexp_extract(filename, '([^/]+)/events\\.\\d+\\.jsonl$', 1) AS run"
    attempt_from = (
        "CAST(regexp_extract(filename, 'events\\.(\\d+)\\.jsonl$', 1) AS INT) "
        "AS attempt"
    )
    con.execute(f"""
        CREATE VIEW pinned AS
        SELECT {run_from}, {attempt_from}, * EXCLUDE (filename)
        FROM read_json('{glob}', filename := true, format := 'newline_delimited',
                       ignore_errors := true, columns := {_columns_clause(coords)})
        WHERE kind IS NOT NULL;

        CREATE VIEW runs AS
        SELECT regexp_extract(filename, '([^/]+)/run\\.json$', 1) AS run,
               * EXCLUDE (filename)
        FROM read_json_auto('{_quote(root / "runs" / "*" / "run.json")}',
                            filename := true, union_by_name := true);

        CREATE VIEW metrics AS {_metrics_sql(coords, resumed=resumed, guard=guard)};
    """)  # noqa: S608 -- interpolates an escaped path and column names the writer validated
    if events:
        con.execute(f"""
            CREATE VIEW events AS
            SELECT {run_from}, {attempt_from}, * EXCLUDE (filename)
            FROM read_json_auto('{glob}', filename := true,
                                format := 'newline_delimited',
                                union_by_name := true, ignore_errors := true);
        """)  # noqa: S608 -- interpolates an escaped path


def _check_duplicates(frame: pl.DataFrame, coords: dict[str, str]) -> None:
    """Raise if one attempt logged the same metric key twice.

    Keeping one of them silently would corrupt a mean, and it always means the
    training loop logged the same thing twice.

    Args:
        frame: The loaded frame.
        coords: The coordinate columns, which are part of the key.

    Raises:
        CorpusError: An attempt holds a duplicate metric key.

    """
    if frame.is_empty():
        return
    keys = ["run", "attempt", "name", "step", "split", *coords]
    duplicated = frame.filter(frame.select(keys).is_duplicated())
    if duplicated.height:
        rows = duplicated.select("run", "attempt", "name", "step").head(5).rows()
        msg = (
            f"duplicate metric keys within one attempt (run, attempt, name, step): "
            f"{rows}"
        )
        raise CorpusError(msg)


def load(
    root: Path | str,
    *,
    names: Sequence[str] | None = None,
    coords: dict[str, str] | None = None,
    require_finished: bool = True,
) -> pl.DataFrame:
    """Load the tidy metric frame of a corpus, with config columns joined.

    Args:
        root: The corpus root.
        names: Metric names to keep. All of them by default.
        coords: Coordinate types to pin. Discovered by default.
        require_finished: Exclude runs that never recorded a ``finished`` status.

    Returns:
        One row per surviving metric record: run, attempt, step, split, name,
        value, nonfinite, the coordinates, and every config key.

    """
    root = Path(root)
    if not _has_runs(root):
        return pl.DataFrame()

    runs = scan(root)
    included = (
        [run for run in runs if run.status == FINISHED] if require_finished else runs
    )
    if require_finished and len(included) < len(runs):
        logger.warning(
            "load: excluding %d of %d runs with no finished status "
            "(see logmint.census)",
            len(runs) - len(included),
            len(runs),
        )
    found = coords_of(included) if coords is None else coords

    where, params = [], []
    if names is not None:
        params.append(list(names))
        where.append(f"list_contains(${len(params)}, name)")
    if require_finished:
        params.append([run.rid for run in included])
        where.append(f"list_contains(${len(params)}, run)")
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    resumed = any(len(run.paths) > 1 for run in included)
    con = duckdb.connect()
    try:
        # When nothing was resumed the frame itself is checked below, which is
        # free; when something was resumed the duplicate is hidden by the
        # deduplication, so SQL must catch it.
        _views(con, root, found, resumed=resumed, guard=resumed, events=False)
        frame = con.execute(
            f"SELECT * FROM metrics {clause}",  # noqa: S608 -- the clause holds only $n placeholders
            params,
        ).pl()
    finally:
        con.close()

    if not resumed:
        _check_duplicates(frame, found)
    return frame


def query(root: Path | str, sql: str) -> pl.DataFrame:
    """Run SQL against a corpus.

    The ``events``, ``runs`` and ``metrics`` views are defined before it runs.

    Args:
        root: The corpus root.
        sql: The query.

    Returns:
        The result.

    """
    root = Path(root)
    runs = scan(root)
    con = duckdb.connect()
    try:
        _views(
            con,
            root,
            coords_of(runs),
            resumed=any(len(run.paths) > 1 for run in runs),
            guard=True,
            # SQL that selects from the events view has to name it, and defining
            # it costs a schema inference over every record in the corpus, so SQL
            # that does not name it does not pay.
            events="events" in sql.lower(),
        )
        return con.execute(sql).pl()
    finally:
        con.close()


def _check_one_row_per_repeat(
    frame: pl.DataFrame, by: list[str], x: str, over: str
) -> None:
    """Check that the repeats are the only thing varying within a group.

    A group holding more than one row per repeat means the frame carries a
    dimension the caller did not name: a second config key the sweep varied, or a
    split that was never filtered out. The mean would then run over more values
    than ``n`` reports, and the error bar would be too narrow by exactly the factor
    nobody would notice.

    Args:
        frame: The tidy frame.
        by: The grouping columns.
        x: The column the curve runs along.
        over: The column the repeats vary along.

    Raises:
        CorpusError: Some group holds more than one row per repeat.

    """
    key = [*by, x, over]
    if not frame.select(key).is_duplicated().any():
        return
    candidates = [c for c in frame.columns if c not in {*key, *_DERIVED}]
    spread = frame.group_by(key).agg([
        pl.col(c).n_unique().alias(c) for c in candidates
    ])
    varying = [c for c in candidates if (spread[c] > 1).any()]
    msg = (
        f"a ({', '.join(key)}) group holds more than one row, so the frame "
        f"carries a dimension you have not named: rows within a group differ in "
        f"{varying}. Put those in by=, or filter them out. Averaging over them "
        "would divide the spread of more values than n."
    )
    raise CorpusError(msg)


def aggregate(
    frame: pl.DataFrame,
    *,
    over: str,
    x: str,
    by: str | Sequence[str] | None = None,
    allow_ragged: bool = False,
    allow_nonfinite: bool = False,
) -> pl.DataFrame:
    """Aggregate a tidy frame over repeats, producing what a figure needs.

    The standard error is ``s / sqrt(n)`` with ``s`` the sample standard deviation.

    Args:
        frame: A tidy frame from ``load``.
        over: The column the repeats vary along, typically ``seed``.
        x: The column the curve runs along, typically ``step``.
        by: One column name or several, to keep separate. Typically the method.
        allow_ragged: Permit ``n`` to vary along ``x`` within a group.
        allow_nonfinite: Drop the records that diverged instead of refusing to
            average them.

    Returns:
        One row per group and x, with columns ``y``, ``yerr`` and ``n``.

    Raises:
        CorpusError: The frame mixes metric names, lacks a column, carries an
            unnamed dimension, holds records that diverged, or has an ``n`` that
            varies along ``x``.

    """
    by = [by] if isinstance(by, str) else list(by or [])
    missing = [c for c in [over, x, *by] if c not in frame.columns]
    if missing:
        msg = f"frame has no column(s) {missing}; it has {frame.columns}"
        raise CorpusError(msg)
    if "name" not in by and frame["name"].n_unique() > 1:
        mixed = sorted(frame["name"].unique())
        msg = (
            f"frame mixes metric names {mixed}; filter with load(names=[...]) "
            "or put 'name' in by"
        )
        raise CorpusError(msg)

    _check_one_row_per_repeat(frame, by, x, over)

    diverged = frame.filter(pl.col("value").is_null())
    if diverged.height and not allow_nonfinite:
        tags = sorted(diverged["nonfinite"].unique())
        msg = (
            f"{diverged.height} of {frame.height} records diverged ({tags}). A "
            f"null is skipped by the mean but still counted by n, so the error bar "
            f"would come out too narrow. Pass allow_nonfinite=True to drop them, "
            f"which will make n vary along {x!r} and say so."
        )
        raise CorpusError(msg)
    frame = frame.filter(pl.col("value").is_not_null())

    grouped = (
        frame
        .group_by([*by, x])
        .agg(
            pl.col("value").mean().alias("y"),
            pl.col("value").std(ddof=1).alias("s"),
            pl.col(over).n_unique().alias("n"),
        )
        .with_columns((pl.col("s") / pl.col("n").sqrt()).alias("yerr"))
        .sort([*by, x])
    )

    if not allow_ragged:
        ragged = (
            (grouped.group_by(by).agg(pl.col("n").n_unique())["n"] > 1).any()
            if by
            else grouped["n"].n_unique() > 1
        )
        if ragged:
            msg = (
                f"n varies along {x!r}: a band whose n changes from point to "
                "point is not comparable across the curve. Pass "
                "allow_ragged=True to accept it."
            )
            raise CorpusError(msg)
    return grouped.select([*by, x, "y", "yerr", "n"])
