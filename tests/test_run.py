"""The writer: attempts, records, and resume (spec section 3)."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

import logmint
from logmint._identity import Config
from tests.conftest import base_config, finished_run


def _lines(root: Path, rid: str, attempt: int = 0) -> list[dict[str, object]]:
    """Parse the records of one attempt file.

    Args:
        root: The corpus root.
        rid: The run id.
        attempt: The attempt index.

    Returns:
        The records that parse.

    """
    path = root / "runs" / rid / f"events.{attempt:02d}.jsonl"
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def test_an_attempt_opens_with_a_start_record(root: Path) -> None:
    """Provenance is per attempt, so it opens the file, not run.json."""
    rid = finished_run(root)
    first = _lines(root, rid)[0]
    assert first["kind"] == "start"
    assert set(first) >= {
        "time",
        "git",
        "dirty",
        "lock",
        "host",
        "gpus",
        "slurm_job_id",
    }
    assert "git" not in json.loads(
        (root / "runs" / rid / "run.json").read_text(encoding="utf-8")
    )


def test_status_is_the_last_line(root: Path) -> None:
    """The terminal record separates a converged run from one that died early."""
    rid = finished_run(root)
    assert _lines(root, rid)[-1] == {
        "kind": "status",
        "status": "finished",
        "wall_s": pytest.approx(0.0, abs=5.0),
        "coords": {},
    }


def test_the_status_record_declares_the_coordinates_used(root: Path) -> None:
    """The declaration is what lets the reader discover the schema from a tail read."""
    config = base_config(seed=16)
    with logmint.init(root, config) as run:
        run.metric("acc", 0.5, step=1, split="forget")
        run.metric("barrier", 0.3, step=1, t=0.5, against="retrain", layer=3)
    status = _lines(root, logmint.run_id(config))[-1]
    assert status["coords"] == {"against": "VARCHAR", "layer": "BIGINT", "t": "DOUBLE"}


def _log_two_types_of_one_coordinate(run: logmint.Run) -> None:
    """Log one coordinate first as a number and then as a string.

    Args:
        run: The open run.

    """
    run.metric("barrier", 0.1, step=1, t=0.5)
    run.metric("barrier", 0.2, step=2, t="half")


def test_a_coordinate_that_changes_type_is_refused(root: Path) -> None:
    """A column has one type, so the writer catches the change at the call site."""
    with (
        logmint.init(root, base_config(seed=17)) as run,
        pytest.raises(logmint.RecordError, match="one type"),
    ):
        _log_two_types_of_one_coordinate(run)


def _crash_inside_a_run(root: Path, config: Config) -> None:
    """Raise from inside a run context, as a training loop would.

    Args:
        root: The corpus root.
        config: The run config.

    Raises:
        ZeroDivisionError: Always.

    """
    with logmint.init(root, config) as run:
        run.metric("acc", 0.5, step=1)
        raise ZeroDivisionError


def test_a_crashing_block_records_a_failure(root: Path) -> None:
    """A run that raises is labelled, not merely absent."""
    config = base_config(seed=9)
    with pytest.raises(ZeroDivisionError):
        _crash_inside_a_run(root, config)
    last = _lines(root, logmint.run_id(config))[-1]
    assert last["kind"] == "status"
    assert last["status"] == "failed"
    assert last["exc"] == "ZeroDivisionError"


def test_attempts_take_the_lowest_free_index(root: Path) -> None:
    """Two attempts of one config land in separate files rather than interleaving."""
    config = base_config(seed=1)
    first = logmint.init(root, config)
    second = logmint.init(root, config)
    assert (first.attempt, second.attempt) == (0, 1)
    first.finish()
    second.finish()
    directory = root / "runs" / logmint.run_id(config)
    assert sorted(p.name for p in directory.glob("events.*.jsonl")) == [
        "events.00.jsonl",
        "events.01.jsonl",
    ]


def test_concurrent_writers_never_share_a_file(root: Path) -> None:
    """O_EXCL is the mutual exclusion, so no lock goes stale under preemption."""
    config = base_config(seed=2)
    runs = [logmint.init(root, config) for _ in range(4)]
    assert sorted(run.attempt for run in runs) == [0, 1, 2, 3]
    for run in runs:
        run.finish()


def test_done_is_false_until_a_run_finishes(root: Path) -> None:
    """The guard a sweep script uses for idempotent resubmission."""
    config = base_config(seed=4)
    run = logmint.init(root, config)
    assert not logmint.done(root, config)
    run.finish()
    assert logmint.done(root, config)


def test_reopening_a_finished_run_is_refused(root: Path) -> None:
    """Accidentally rerunning finished work costs GPU hours, so it raises by default."""
    config = base_config(seed=5)
    logmint.init(root, config).finish()
    with pytest.raises(logmint.AlreadyFinishedError, match="already recorded"):
        logmint.init(root, config)
    logmint.init(root, config, allow_rerun=True).finish()


def test_a_non_finite_value_is_null_with_a_tag(root: Path) -> None:
    """RFC 8259 has no NaN token, so a divergence is a null plus a tag, still JSON."""
    config = base_config(seed=6)
    with logmint.init(root, config) as run:
        run.metric("loss", float("nan"), step=1)
        run.metric("loss", float("inf"), step=2)
        run.metric("loss", float("-inf"), step=3)
    records = [r for r in _lines(root, logmint.run_id(config)) if r["kind"] == "metric"]
    assert [r["value"] for r in records] == [None, None, None]
    assert [r["nonfinite"] for r in records] == ["nan", "+inf", "-inf"]


def test_a_non_finite_coordinate_is_refused(root: Path) -> None:
    """A coordinate indexes a result, so it cannot be a gap."""
    with (
        logmint.init(root, base_config(seed=7)) as run,
        pytest.raises(logmint.RecordError, match="non-finite"),
    ):
        run.metric("barrier", 0.3, step=1, t=float("nan"))


def test_a_boolean_is_not_a_measurement(root: Path) -> None:
    """Value is a real number; a boolean is neither measurement nor coordinate."""
    with (
        logmint.init(root, base_config(seed=8)) as run,
        pytest.raises(logmint.RecordError, match="real number"),
    ):
        run.metric("converged", True, step=1)


def test_a_reserved_coordinate_name_is_refused(root: Path) -> None:
    """Coordinates may not shadow the core columns."""
    with (
        logmint.init(root, base_config(seed=10)) as run,
        pytest.raises(logmint.RecordError, match="reserved"),
    ):
        run.metric("acc", 0.5, step=1, attempt=2)


def test_a_non_scalar_coordinate_is_refused(root: Path) -> None:
    """Every record is a flat row, so coordinates are scalars."""
    with (
        logmint.init(root, base_config(seed=11)) as run,
        pytest.raises(logmint.RecordError, match="must be a scalar"),
    ):
        run.metric("acc", 0.5, step=1, layers=[1, 2])  # ty: ignore[invalid-argument-type]


def test_writing_after_the_status_record_is_refused(root: Path) -> None:
    """The stream is closed once it is terminated."""
    run = logmint.init(root, base_config(seed=12))
    run.finish()
    with pytest.raises(logmint.RecordError, match="closed"):
        run.metric("acc", 0.5, step=1)


def test_last_blob_is_available_before_init(root: Path) -> None:
    """A resuming job resolves its checkpoint before opening the new attempt."""
    config = base_config(seed=13)
    with logmint.init(root, config) as run:
        run.blob("checkpoint", np.arange(4), step=100)
        run.blob("checkpoint", np.arange(8), step=200)

    previous = logmint.last_blob(root, config, "checkpoint")
    assert previous is not None
    assert previous.step == 200
    np.testing.assert_array_equal(logmint.get(root, previous.ref), np.arange(8))

    with logmint.init(
        root, config, resumed_from=previous.step, allow_rerun=True
    ) as run:
        assert run.attempt == 1
    start = _lines(root, logmint.run_id(config), attempt=1)[0]
    assert start["resumed_from"] == 200


def test_last_blob_ignores_the_current_attempt(root: Path) -> None:
    """A run resuming from itself would be a cycle."""
    config = base_config(seed=14)
    with logmint.init(root, config) as run:
        run.blob("checkpoint", np.arange(2), step=1)
    with logmint.init(root, config, allow_rerun=True) as run:
        run.blob("checkpoint", np.arange(3), step=2)
        found = run.last_blob("checkpoint")
    assert found is not None
    assert found.step == 1


def test_events_keep_prose_out_of_the_results_stream(root: Path) -> None:
    """A runtime fact worth querying is a record, not a line of English."""
    config = base_config(seed=15)
    with logmint.init(root, config) as run:
        run.event("grad_overflow", step=3, level="warning")
    record = next(
        r for r in _lines(root, logmint.run_id(config)) if r["kind"] == "event"
    )
    assert record["name"] == "grad_overflow"
    assert record["level"] == "warning"


def test_a_coordinate_may_not_shadow_a_config_key(root: Path) -> None:
    """The frame joins config onto record columns, so the two namespaces are one."""
    with (
        logmint.init(root, base_config(seed=18)) as run,
        pytest.raises(logmint.RecordError, match="also config keys"),
    ):
        run.metric("sensitivity", 0.3, step=100, lr=0.05)


def test_a_metric_coordinate_may_not_take_over_the_record(root: Path) -> None:
    """A key that lands on a field the record carries would rewrite the record."""
    with (
        logmint.init(root, base_config(seed=19)) as run,
        pytest.raises(logmint.RecordError, match="would overwrite"),
    ):
        run.metric("acc", 0.5, step=1, kind="oops")


def test_blob_metadata_may_not_take_over_the_record(root: Path) -> None:
    """Rewriting a blob record's ref points it away from the bytes it just wrote,
    which makes those bytes unreachable and hands them to the collector.
    """  # noqa: D205
    with (
        logmint.init(root, base_config(seed=20)) as run,
        pytest.raises(logmint.RecordError, match="would overwrite"),
    ):
        run.blob("checkpoint", np.zeros(2), step=1, ref="blob:" + "0" * 64)


def test_an_event_may_not_declare_itself_a_metric(root: Path) -> None:
    """An event that rewrites its kind enters the metric frame and corrupts a mean."""
    with (
        logmint.init(root, base_config(seed=21)) as run,
        pytest.raises(logmint.RecordError, match="would overwrite"),
    ):
        run.event("grad_overflow", step=1, kind="metric")


def test_blob_metadata_may_not_shadow_a_recorded_field(root: Path) -> None:
    """format, bytes, shape and dtype belong to the record, not to the caller."""
    with (
        logmint.init(root, base_config(seed=22)) as run,
        pytest.raises(logmint.RecordError, match="would overwrite"),
    ):
        run.blob("checkpoint", np.zeros(2), step=1, dtype="lies")


def test_a_step_is_an_integer_not_a_boolean(root: Path) -> None:
    """The step column is BIGINT, and True is not a step."""
    with (
        logmint.init(root, base_config(seed=23)) as run,
        pytest.raises(logmint.RecordError, match="step must be an integer"),
    ):
        run.metric("acc", 0.5, step=True)


def test_a_split_is_a_string(root: Path) -> None:
    """The split column is VARCHAR."""
    with (
        logmint.init(root, base_config(seed=24)) as run,
        pytest.raises(logmint.RecordError, match="split must be a string"),
    ):
        run.metric("acc", 0.5, step=1, split=3)  # ty: ignore[invalid-argument-type]


def test_a_metric_name_is_a_non_empty_string(root: Path) -> None:
    """A nameless metric cannot be selected; one named 123 lies about its column."""
    with (
        logmint.init(root, base_config(seed=25)) as run,
        pytest.raises(logmint.RecordError, match="non-empty string"),
    ):
        run.metric("", 0.5, step=1)


@pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded:DeprecationWarning"
)
def test_a_forked_process_may_not_write_through_the_inherited_handle(
    root: Path,
) -> None:
    """A fork inherits the descriptor, so two processes interleave in one file."""
    run = logmint.init(root, base_config(seed=26))
    run.metric("acc", 0.5, step=1)

    read_fd, write_fd = os.pipe()
    if (pid := os.fork()) == 0:  # pragma: no cover -- runs only in the child
        os.close(read_fd)
        try:
            run.metric("acc", 0.9, step=2)
            os.write(write_fd, b"wrote")
        except logmint.RecordError:
            os.write(write_fd, b"raised")
        os._exit(0)

    os.close(write_fd)
    outcome = os.read(read_fd, 16)
    os.close(read_fd)
    os.waitpid(pid, 0)
    run.finish()

    assert outcome == b"raised"
    steps = [r["step"] for r in _lines(root, run.run_id) if r["kind"] == "metric"]
    assert steps == [1]


def test_an_abandoned_run_releases_its_descriptor_but_stays_incomplete(
    root: Path,
) -> None:
    """Closing the file is not finishing the run. A driver that opens runs and
    moves on would otherwise hold every descriptor until the process exits.
    """  # noqa: D205
    config = base_config(seed=27)
    run = logmint.init(root, config)
    run.metric("acc", 0.5, step=1)
    run.close()

    assert not logmint.done(root, config)
    assert logmint.census(root)["incomplete"] == 1
    with pytest.raises(logmint.RecordError, match="closed"):
        run.metric("acc", 0.6, step=2)
    run.close()  # idempotent


def test_closing_twice_and_finishing_after_close_are_both_safe(root: Path) -> None:
    """A terminated run is closed, and closing it again is not an error."""
    with logmint.init(root, base_config(seed=28)) as run:
        run.metric("acc", 0.5, step=1)
    run.close()
    assert logmint.done(root, base_config(seed=28))


def test_a_run_written_from_another_process_is_refused_in_process(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check the fork test proves across a real fork, where coverage can see it."""
    with logmint.init(root, base_config(seed=29)) as run:
        monkeypatch.setattr(run, "_pid", -1)
        with pytest.raises(logmint.RecordError, match="inherits the file descriptor"):
            run.metric("acc", 0.5, step=1)
        monkeypatch.undo()


def test_a_coordinate_that_is_not_an_identifier_is_refused(root: Path) -> None:
    """A coordinate is a column name, and a name with a space in it needs quoting."""
    with (
        logmint.init(root, base_config(seed=30)) as run,
        pytest.raises(logmint.RecordError, match="not an identifier"),
    ):
        run.metric("acc", 0.5, step=1, **{"my coord": 1})  # ty: ignore[invalid-argument-type]


def test_closing_inside_the_context_manager_is_safe(root: Path) -> None:
    """The exit path finds the run already closed and leaves it alone."""
    config = base_config(seed=31)
    with logmint.init(root, config) as run:
        run.metric("acc", 0.5, step=1)
        run.close()
    assert not logmint.done(root, config)


def test_last_blob_before_the_run_has_ever_started(root: Path) -> None:
    """The first submission asks this, and there is no run directory yet."""
    assert logmint.last_blob(root, base_config(seed=32), "checkpoint") is None


def test_last_blob_skips_a_truncated_line(root: Path) -> None:
    """A kill during a blob record leaves half of it, and a resume still has to work."""
    config = base_config(seed=33)
    run = logmint.init(root, config)
    ref = run.blob("checkpoint", np.zeros(4), step=100)
    run.close()
    path = root / "runs" / logmint.run_id(config) / "events.00.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"kind":"blob","step":200,"name":"checkpoint","ref":"blob:00')

    found = logmint.last_blob(root, config, "checkpoint")
    assert found is not None
    assert (found.step, found.ref) == (100, ref)


def test_a_numpy_scalar_is_a_number(root: Path) -> None:
    """np.float64 subclasses float and np.float32 does not, so a writer that tests
    for float accepts a metric from a float64 array and rejects the same metric
    from a float32 one. Every training loop produces the second.
    """  # noqa: D205
    with logmint.init(root, base_config(seed=34)) as run:
        run.metric("acc", np.float32(0.5), step=np.int64(100), split="forget")  # ty: ignore[invalid-argument-type]
        run.metric("acc", np.float64(0.6), step=np.int32(200), split="forget")  # ty: ignore[invalid-argument-type]
        run.metric("acc", np.zeros(()) + 0.7, step=300, split="forget")  # ty: ignore[invalid-argument-type]

    frame = logmint.load(root, names=["acc"])
    assert frame["value"].to_list() == [0.5, 0.6, 0.7]
    assert frame["step"].to_list() == [100, 200, 300]


def test_an_array_of_several_numbers_is_not_a_metric(root: Path) -> None:
    """It is a blob, and the message says which."""
    with (
        logmint.init(root, base_config(seed=35)) as run,
        pytest.raises(logmint.RecordError, match="must be a real number"),
    ):
        run.metric("acc", np.zeros(3), step=1)  # ty: ignore[invalid-argument-type]


def test_an_integer_coordinate_stays_an_integer(root: Path) -> None:
    """A layer index is not 3.0."""
    with logmint.init(root, base_config(seed=36)) as run:
        run.metric("sensitivity", 0.3, step=1, layer=np.int64(3), tied=np.True_)  # ty: ignore[invalid-argument-type]

    frame = logmint.load(root, names=["sensitivity"])
    assert frame["layer"].to_list() == [3]
    assert frame["tied"].to_list() == [True]
    assert logmint.discover(root) == {"layer": "BIGINT", "tied": "BOOLEAN"}


def test_a_coordinate_written_as_an_integer_and_then_a_real_is_one_column(
    root: Path,
) -> None:
    """A coordinate written 0, then 0.5, is one axis. Refusing that refuses a sweep."""
    with logmint.init(root, base_config(seed=37)) as run:
        run.metric("barrier", 0.1, step=1, t=0)
        run.metric("barrier", 0.2, step=1, t=0.5)
        run.metric("barrier", 0.3, step=1, t=1)

    assert logmint.discover(root) == {"t": "DOUBLE"}
    assert logmint.load(root, names=["barrier"])["t"].to_list() == [0.0, 0.5, 1.0]


def test_a_coordinate_written_as_a_number_and_then_a_string_is_two_columns(
    root: Path,
) -> None:
    """Widening a number is one column; a number and a name are two axes, one name."""
    with logmint.init(root, base_config(seed=38)) as run:
        run.metric("barrier", 0.1, step=1, t=0.5)
        with pytest.raises(logmint.RecordError, match="a column has one type"):
            run.metric("barrier", 0.2, step=1, t="half")


def test_a_runtime_that_cannot_be_serialised_leaves_no_attempt_behind(
    root: Path,
) -> None:
    """The runtime dict is the caller's and can hold anything. An attempt that
    never recorded its start never began, so it must not sit in the corpus looking
    like one that did.
    """  # noqa: D205
    config = base_config(seed=39)
    with pytest.raises(logmint.RecordError, match="JSON cannot represent"):
        logmint.init(root, config, runtime={"model": object()})

    # The config is registered, because two processes may race to write it. The
    # attempt is not.
    assert not (root / "runs" / logmint.run_id(config) / "events.00.jsonl").exists()
    assert not logmint.done(root, config)
    assert logmint.census(root) == {
        "runs": 1,
        "finished": 0,
        "failed": 0,
        "incomplete": 1,
    }

    with logmint.init(root, config, runtime={"workers": 8}) as run:
        run.metric("acc", 0.5, step=1)
    assert run.attempt == 0
