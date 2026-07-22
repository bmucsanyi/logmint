# logmint

Bálint's opinionated logging stack. Append-only run logs, content-addressed blobs, and a tidy
frame that goes straight into `figmint`.

```python
import logmint

# a non-scalar input exists before the run, so it is part of the run's identity
subset = logmint.put("results", forget_indices)  # -> "blob:9f2c..."

config = {"method": "scrub", "lr": 1e-4, "seed": 0, "subset": subset}
if logmint.done("results", config):  # idempotent resubmission: a stat
    raise SystemExit

with logmint.init("results", config) as run:  # run_id = sha256(canonical(config))[:16]
    for step in steps:
        run.metric("acc", acc, step=step, split="forget")
        run.metric("barrier", b, step=step, t=t)  # any extra key is a coordinate
        run.blob("checkpoint", path, step=step)
```

```python
frame = logmint.load("results", names=["acc"]).filter(
    split="forget"
)  # finished runs, config joined
table = logmint.aggregate(
    frame, over="seed", x="step", by="method"
)  # y, yerr = s/sqrt(n), n
figmint.export_table("acc.tsv", x=table["step"], y=table["y"], yerr=table["yerr"])
```

`aggregate` raises rather than hand you a wrong error bar. If a `(by, x, over)` group holds more
than one row, the frame carries a dimension you have not named - a second config key the sweep
varied, or a split you forgot to filter - and the mean would run over more values than `n` reports.
If a seed diverged, its `null` is skipped by the mean but still counted by `n`. If `n` varies along
`x`, the band is not comparable from point to point. Each of those is a wrong number in a figure
that nothing else would catch.

```sh
logmint ls                       # every run, its status, its config
logmint query "SELECT ..."       # SQL over metrics / runs / events
logmint verify                   # exits nonzero; run it before you make a figure
logmint gc --dry-run             # unreachable blobs, outside the grace period
logmint schema                   # SCHEMA.md, the file an agent reads first
```

Three views. `metrics` is the tidy frame, deduplicated across attempts, with the config joined on.
`runs` is one row per run. `events` is every record of every kind with every field on it - the
provenance on a `start`, the fields on an `event`, the shape and dtype on a `blob` - so
`SELECT run, step, level FROM events WHERE name = 'grad_overflow'` is a question you can ask.

## Why the format is what it is

A preempted writer costs one line, not the run: a SIGKILL can only cut the line being written,
and that line is always the last one in its file. Parquet cannot offer this - its footer holds
the schema, so a killed writer leaves a file that reads back as zero rows.

One file per attempt, created with `O_CREAT | O_EXCL` at the lowest free index. Mutual exclusion
is a byproduct of creating the file, so there is no lock to go stale when a job is preempted.

The run id is the hash of the config, so semantics live in the config and never in a name table.
Two runs of the same config are the same run; a resubmission is a stat.

Records are ragged by design. A metric carries `name` and `value`; everything else on the line is
a coordinate. A new coordinate is a new key on new lines, and every earlier file still reads.

The reader never scans the corpus to learn its schema. A terminating attempt declares its
coordinates in its `status` record, so discovery is a tail read of each attempt file - 0.19 s
across 2000 runs, and it does not grow with the number of records. Only an attempt that was
interrupted before it could declare has to be read, and only if its run is one you are loading.

## Numbers

A 2000-run factorial sweep, $10^6$ metric records, 102 MB on disk:

| | |
|---|---|
| write | 19 us/record (line-buffered: a SIGKILL costs one line, not one buffer) |
| `scan` / `census` | 0.19 s |
| `load` (whole joined frame, $10^6$ rows) | 1.7 s |
| `aggregate` | 0.25 s |
| `query` (aggregates in SQL) | 1.2 s |
| `query` against `events` (full fidelity) | 2.6 s |
| `verify` (parses every line) | 4.0 s |
| *for comparison:* DuckDB inferring the schema, just to `count(*)` | 2.2 s |

## Install

```sh
pip install -e ".[dev]"
pytest && ruff check . && ruff format --check .
```

`ruff` runs with `select = ["ALL"]`. Tests cover 100% of the package.
