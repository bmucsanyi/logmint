"""The reader and the aggregation that feeds a figure (spec sections 6 and 7)."""

import logging
from pathlib import Path

import duckdb
import numpy as np
import pytest

import logmint
from logmint._identity import Config
from tests.conftest import base_config, finished_run


def _corpus(root: Path) -> None:
    """Write two methods by three seeds, with a coordinate on one metric only.

    Args:
        root: The corpus root.

    """
    for method in ("scrub", "npo"):
        for seed in range(3):
            config = base_config(method=method, seed=seed)
            with logmint.init(root, config) as run:
                for step in (100, 200):
                    run.metric("acc", 0.5 + 0.1 * seed, step=step, split="forget")
                    run.metric("acc", 0.9, step=step, split="retain")
                run.metric("barrier", 0.3, step=200, t=0.5, against="reference")


def test_coordinates_are_discovered_from_metric_records_only(root: Path) -> None:
    """A coordinate is any key on a metric record beyond the core columns."""
    _corpus(root)
    assert logmint.discover(root) == {"against": "VARCHAR", "t": "DOUBLE"}


def test_records_are_ragged_and_the_frame_null_fills(root: Path) -> None:
    """A metric without a coordinate gets a null; no schema is needed up front."""
    _corpus(root)
    frame = logmint.load(root)
    accuracy = frame.filter(name="acc")
    barrier = frame.filter(name="barrier")
    assert accuracy["t"].null_count() == accuracy.height
    assert barrier["t"].to_list() == [0.5] * 6
    assert barrier["split"].null_count() == barrier.height


def test_config_columns_join_onto_every_record(root: Path) -> None:
    """Semantics live in the config, so a plot filters on method, not an opaque id."""
    _corpus(root)
    frame = logmint.load(root, names=["acc"])
    assert set(frame["method"].unique()) == {"scrub", "npo"}
    assert frame.filter(method="scrub", split="forget").height == 6


def test_a_new_coordinate_does_not_invalidate_older_runs(root: Path) -> None:
    """Adding a coordinate mid-project is a new key on new lines, not a schema edit."""
    finished_run(root, seed=0)
    with logmint.init(root, base_config(seed=1)) as run:
        run.metric("acc", 0.4, step=200, split="forget", layer=3)
    frame = logmint.load(root, names=["acc"]).sort("seed", "step")
    assert frame["layer"].to_list() == [None, None, 3.0]


def test_incomplete_runs_are_excluded_and_counted(
    root: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A run that never finished never enters a figure, and never disappears quietly."""
    finished_run(root, seed=0)
    logmint.init(root, base_config(seed=1)).metric("acc", 0.9, step=1)

    with caplog.at_level(logging.WARNING, logger="logmint"):
        frame = logmint.load(root, names=["acc"])
    assert frame["seed"].unique().to_list() == [0]
    assert "excluding 1 of 2 runs" in caplog.text
    assert logmint.census(root) == {
        "runs": 2,
        "finished": 1,
        "failed": 0,
        "incomplete": 1,
    }


def _crash_inside_a_run(root: Path, config: Config) -> None:
    """Raise from inside a run context.

    Args:
        root: The corpus root.
        config: The run config.

    Raises:
        RuntimeError: Always.

    """
    with logmint.init(root, config) as run:
        run.metric("acc", 0.1, step=1)
        raise RuntimeError


def test_a_failed_run_is_counted_separately(root: Path) -> None:
    """A labelled failure is not the same thing as a run that vanished."""
    with pytest.raises(RuntimeError):
        _crash_inside_a_run(root, base_config(seed=3))
    assert logmint.census(root) == {
        "runs": 1,
        "finished": 0,
        "failed": 1,
        "incomplete": 0,
    }


def test_a_duplicate_key_within_one_attempt_raises(root: Path) -> None:
    """Silently keeping one of them would corrupt a mean."""
    with logmint.init(root, base_config(seed=4)) as run:
        run.metric("acc", 0.5, step=1, split="forget")
        run.metric("acc", 0.6, step=1, split="forget")
    with pytest.raises(logmint.CorpusError, match="duplicate metric keys"):
        logmint.load(root)


def test_the_same_key_at_different_coordinates_is_not_a_duplicate(root: Path) -> None:
    """Coordinates are part of the key, which is what makes a curve a curve."""
    with logmint.init(root, base_config(seed=5)) as run:
        run.metric("barrier", 0.1, step=1, t=0.25)
        run.metric("barrier", 0.2, step=1, t=0.75)
    assert logmint.load(root).height == 2


def test_a_divergence_survives_the_round_trip(root: Path) -> None:
    """The null leaves a gap in the curve; the tag says which kind."""
    with logmint.init(root, base_config(seed=6)) as run:
        run.metric("loss", 1.0, step=1)
        run.metric("loss", float("nan"), step=2)
    frame = logmint.load(root, names=["loss"]).sort("step")
    assert frame["value"].to_list() == [1.0, None]
    assert frame["nonfinite"].to_list() == [None, "nan"]


def test_aggregate_produces_what_a_figure_needs(root: Path) -> None:
    """Mean, standard error, and the count that says if the error bar is meaningful."""
    _corpus(root)
    frame = logmint.load(root, names=["acc"]).filter(split="forget")
    table = logmint.aggregate(frame, over="seed", x="step", by=["method"])

    assert table.columns == ["method", "step", "y", "yerr", "n"]
    assert table.height == 4
    assert table["n"].to_list() == [3, 3, 3, 3]
    row = table.filter(method="scrub", step=100)
    assert row["y"][0] == pytest.approx(0.6)
    assert row["yerr"][0] == pytest.approx(np.std([0.5, 0.6, 0.7], ddof=1) / np.sqrt(3))


def test_by_takes_one_column_or_several(root: Path) -> None:
    """A bare column name is not a sequence of characters."""
    _corpus(root)
    frame = logmint.load(root, names=["acc"]).filter(split="forget")
    one = logmint.aggregate(frame, over="seed", x="step", by="method")
    many = logmint.aggregate(frame, over="seed", x="step", by=["method"])
    assert one.equals(many)


def test_aggregate_refuses_a_band_whose_n_varies(root: Path) -> None:
    """A seed that dies at step 40k silently widens the band from that point on."""
    for seed in range(3):
        config = base_config(seed=seed)
        with logmint.init(root, config) as run:
            run.metric("acc", 0.5, step=100, split="forget")
            if seed < 2:
                run.metric("acc", 0.4, step=200, split="forget")
    frame = logmint.load(root, names=["acc"])
    with pytest.raises(logmint.CorpusError, match="n varies"):
        logmint.aggregate(frame, over="seed", x="step", by=["method"])

    table = logmint.aggregate(
        frame, over="seed", x="step", by=["method"], allow_ragged=True
    )
    assert table["n"].to_list() == [3, 2]


def test_aggregate_refuses_to_mix_metric_names(root: Path) -> None:
    """Averaging accuracy together with loss is not an error SQL would catch."""
    _corpus(root)
    with pytest.raises(logmint.CorpusError, match="mixes metric names"):
        logmint.aggregate(logmint.load(root), over="seed", x="step", by=["method"])


def test_aggregate_reports_a_missing_column(root: Path) -> None:
    """Grouping over a config key that does not exist is a mistake worth naming."""
    _corpus(root)
    frame = logmint.load(root, names=["acc"])
    with pytest.raises(logmint.CorpusError, match="no column"):
        logmint.aggregate(frame, over="replicate", x="step")


def test_query_exposes_three_views(root: Path) -> None:
    """The corpus answers SQL without an index, a server, or an export step."""
    _corpus(root)
    counts = logmint.query(
        root,
        "SELECT (SELECT count(*) FROM runs) AS runs, "
        "(SELECT count(*) FROM events WHERE kind='status') AS statuses, "
        "(SELECT count(*) FROM metrics) AS metrics",
    )
    assert counts["runs"][0] == 6
    assert counts["statuses"][0] == 6
    assert counts["metrics"][0] == 30


def test_load_on_an_empty_corpus_is_empty(root: Path) -> None:
    """A corpus with no runs is not an error."""
    (root / "runs").mkdir(parents=True)
    assert logmint.load(root).is_empty()
    assert logmint.census(root) == {
        "runs": 0,
        "finished": 0,
        "failed": 0,
        "incomplete": 0,
    }


def test_aggregate_refuses_to_pool_over_an_unnamed_dimension(root: Path) -> None:
    """The sweep varied lr; the plot groups only by method. Averaging over lr
    would divide the spread of six values by the square root of three, and nothing
    would look wrong.
    """  # noqa: D205
    for method in ("scrub", "npo"):
        for lr, bump in ((1e-3, 0.3), (1e-4, 0.0)):
            for seed in range(3):
                with logmint.init(
                    root, {"method": method, "lr": lr, "seed": seed}
                ) as run:
                    run.metric("acc", 0.5 + bump, step=100, split="forget")

    frame = logmint.load(root, names=["acc"])
    with pytest.raises(logmint.CorpusError, match=r"differ in \['lr'\]"):
        logmint.aggregate(frame, over="seed", x="step", by="method")

    table = logmint.aggregate(frame, over="seed", x="step", by=["method", "lr"])
    assert table["n"].to_list() == [3, 3, 3, 3]


def test_aggregate_refuses_to_pool_over_an_unfiltered_split(root: Path) -> None:
    """The same guard catches the split you forgot to filter out."""
    _corpus(root)
    frame = logmint.load(root, names=["acc"])
    with pytest.raises(logmint.CorpusError, match=r"differ in \['split'\]"):
        logmint.aggregate(frame, over="seed", x="step", by="method")


def test_aggregate_refuses_to_average_away_a_divergence(root: Path) -> None:
    """A null is skipped by the mean but counted by n, so the band comes out narrow."""
    for seed in range(3):
        with logmint.init(root, base_config(seed=seed)) as run:
            value = float("nan") if seed == 2 else 0.5 + 0.1 * seed
            run.metric("acc", value, step=100, split="forget")

    frame = logmint.load(root, names=["acc"])
    with pytest.raises(logmint.CorpusError, match="diverged"):
        logmint.aggregate(frame, over="seed", x="step", by="method")

    table = logmint.aggregate(
        frame, over="seed", x="step", by="method", allow_nonfinite=True
    )
    assert table["n"].to_list() == [2]
    assert table["y"][0] == pytest.approx(0.55)
    assert table["yerr"][0] == pytest.approx(np.std([0.5, 0.6], ddof=1) / np.sqrt(2))


def test_dropping_a_divergence_makes_n_vary_and_say_so(root: Path) -> None:
    """The two guards compose: a seed diverging leaves a band whose n changes."""
    for seed in range(3):
        with logmint.init(root, base_config(seed=seed)) as run:
            run.metric("acc", 0.5, step=100, split="forget")
            run.metric(
                "acc", float("nan") if seed == 2 else 0.4, step=200, split="forget"
            )

    frame = logmint.load(root, names=["acc"])
    with pytest.raises(logmint.CorpusError, match="n varies"):
        logmint.aggregate(
            frame, over="seed", x="step", by="method", allow_nonfinite=True
        )

    table = logmint.aggregate(
        frame,
        over="seed",
        x="step",
        by="method",
        allow_nonfinite=True,
        allow_ragged=True,
    )
    assert table["n"].to_list() == [3, 2]


def test_the_events_view_carries_what_the_pinned_read_drops(root: Path) -> None:
    """A start record's provenance, an event's fields, a blob's shape: all queryable."""
    with logmint.init(root, base_config(), runtime={"workers": 8}) as run:
        run.metric("acc", 0.5, step=1, split="forget")
        run.event("grad_overflow", step=1, level="warning", scale=65536.0)
        run.blob("eigvals", np.ones(7, dtype=np.float32), step=1)

    found = logmint.query(
        root,
        "SELECT (SELECT host FROM events WHERE kind='start') AS host, "
        "(SELECT level FROM events WHERE kind='event') AS level, "
        "(SELECT scale FROM events WHERE kind='event') AS scale, "
        "(SELECT dtype FROM events WHERE kind='blob') AS dtype, "
        "(SELECT len(shape) FROM events WHERE kind='blob') AS rank",
    )
    assert found["level"][0] == "warning"
    assert found["scale"][0] == pytest.approx(65536.0)
    assert found["dtype"][0] == "float32"
    assert found["rank"][0] == 1
    assert found["host"][0]


def test_query_refuses_a_duplicate_key_too(root: Path) -> None:
    """The guard is in the view, so it catches whoever reaches the metrics, not load."""
    with logmint.init(root, base_config()) as run:
        run.metric("acc", 0.5, step=1, split="forget")
        run.metric("acc", 0.6, step=1, split="forget")
    with pytest.raises(duckdb.Error, match="duplicate metric key"):
        logmint.query(root, "SELECT avg(value) FROM metrics")


def test_a_duplicate_key_is_caught_after_a_resume_too(root: Path) -> None:
    """Deduplication hides a within-attempt duplicate, so SQL catches it."""
    config = base_config()
    with logmint.init(root, config) as run:
        run.metric("acc", 0.5, step=1, split="forget")
        run.metric("acc", 0.6, step=1, split="forget")
    with logmint.init(root, config, allow_rerun=True) as run:
        run.metric("acc", 0.7, step=2, split="forget")
    with pytest.raises(duckdb.Error, match="duplicate metric key"):
        logmint.load(root)


def test_incomplete_runs_are_never_read_for_their_coordinates(root: Path) -> None:
    """On a preemptible cluster the incomplete runs are the many, and load
    excludes them, so it must not pay to read them. An attempt that never
    terminated declared nothing, so the only way to learn its coordinates is to
    scan it; if load had, its coordinate would be a column here.
    """  # noqa: D205
    finished_run(root, seed=0)
    crashed = logmint.init(root, base_config(seed=1))
    crashed.metric("acc", 0.5, step=1, t=0.25)

    frame = logmint.load(root, names=["acc"])
    assert frame.height == 2
    assert "t" not in frame.columns

    assert logmint.discover(root) == {"t": "DOUBLE"}
    assert "t" in logmint.load(root, require_finished=False).columns


def test_a_coordinate_with_two_types_across_the_corpus_is_refused(root: Path) -> None:
    """A column has one type; two runs disagreeing make an unreadable corpus."""
    with logmint.init(root, base_config(seed=0)) as run:
        run.metric("barrier", 0.1, step=1, t=0.5)
    with logmint.init(root, base_config(seed=1)) as run:
        run.metric("barrier", 0.2, step=1, t="half")

    with pytest.raises(logmint.CorpusError, match="it has one type"):
        logmint.load(root)


def test_status_of_classifies_one_run_without_walking_the_corpus(root: Path) -> None:
    """The CLI asks this per run, so it must not be linear in the size of the corpus."""
    finished = finished_run(root, seed=0)
    crashed = logmint.init(root, base_config(seed=1))
    crashed.metric("acc", 0.5, step=1)
    crashed.close()

    assert logmint.status_of(root, finished) == "finished"
    assert logmint.status_of(root, logmint.run_id(base_config(seed=1))) == "incomplete"


def test_an_empty_root_is_an_empty_corpus(root: Path) -> None:
    """Nothing has been written yet, and the reader says so instead of failing."""
    assert logmint.census(root) == {
        "runs": 0,
        "finished": 0,
        "failed": 0,
        "incomplete": 0,
    }
    assert logmint.discover(root) == {}
    assert logmint.load(root).is_empty()


def test_an_interrupted_attempt_widens_its_own_coordinate(root: Path) -> None:
    """The reader has to scan an attempt that never declared, and reach the same
    answer the writer would have declared: 0 and 0.5 are one axis.
    """  # noqa: D205
    crashed = logmint.init(root, base_config(seed=2))
    crashed.metric("barrier", 0.1, step=1, t=0)
    crashed.metric("barrier", 0.2, step=1, t=0.5)
    crashed.close()

    assert logmint.discover(root) == {"t": "DOUBLE"}
    assert logmint.load(root, require_finished=False)["t"].to_list() == [0.0, 0.5]


def test_an_interrupted_attempt_with_a_broken_coordinate_is_refused(root: Path) -> None:
    """Two axes wearing one name, found by the scan rather than the declaration."""
    crashed = logmint.init(root, base_config(seed=3))
    crashed.metric("barrier", 0.1, step=1, t=0.5)
    crashed.close()
    path = root / "runs" / logmint.run_id(base_config(seed=3)) / "events.00.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            '{"kind":"metric","step":1,"name":"barrier","value":0.2,"t":"half"}\n'
        )

    with pytest.raises(logmint.CorpusError, match="it has one type"):
        logmint.discover(root)
