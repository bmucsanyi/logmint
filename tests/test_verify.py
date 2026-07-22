"""The invariants, and the generated schema (spec sections 9 and 12)."""

import hashlib
import json
from pathlib import Path

import numpy as np

import logmint
from logmint import _blobs
from logmint._identity import RESERVED
from logmint._read import CORE
from tests.conftest import base_config, finished_run


def _errors(root: Path, *, blobs: bool = False) -> list[str]:
    """Collect the error messages verify reports.

    Args:
        root: The corpus root.
        blobs: Re-hash blob contents.

    Returns:
        The messages, warnings excluded.

    """
    return [p.message for p in logmint.verify(root, blobs=blobs) if p.level == "error"]


def test_a_healthy_corpus_verifies(root: Path) -> None:
    """Every invariant holds on a corpus the writer produced."""
    with logmint.init(root, base_config()) as run:
        run.metric("acc", 0.5, step=1)
        run.blob("eigvals", np.ones(3), step=1)
    assert logmint.verify(root) == []


def test_verify_catches_a_hand_edited_config(root: Path) -> None:
    """Invariant 1: the directory name is the hash of the config inside it."""
    rid = finished_run(root)
    path = root / "runs" / rid / "run.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["lr"] = 0.02
    path.write_text(logmint.canonical(config), encoding="utf-8")
    assert _errors(root) == ["directory name is not the hash of run.json"]


def test_verify_catches_a_non_canonical_config(root: Path) -> None:
    """Invariant 1: a pretty-printed config no longer hashes to its directory."""
    rid = finished_run(root)
    path = root / "runs" / rid / "run.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    assert _errors(root) == ["run.json is not in canonical form"]


def test_verify_catches_a_dangling_reference(root: Path) -> None:
    """Invariant 2: every reference resolves."""
    with logmint.init(root, base_config()) as run:
        ref = run.blob("eigvals", np.ones(3), step=1)
    _blobs.path_for(root, ref).unlink()
    assert _errors(root) == [f"dangling reference: {ref}"]


def test_verify_catches_a_blob_that_lies_about_its_content(root: Path) -> None:
    """Invariant 2 under --blobs: the name is the hash of the bytes, and checked."""
    with logmint.init(root, base_config()) as run:
        ref = run.blob("eigvals", np.ones(3), step=1)
    _blobs.path_for(root, ref).write_bytes(b"not what the name says")
    assert _errors(root) == []
    assert _errors(root, blobs=True) == [f"blob content does not match its name: {ref}"]


def test_verify_catches_a_status_record_before_the_end(root: Path) -> None:
    """Invariant 3: the terminal record is terminal."""
    rid = finished_run(root)
    path = root / "runs" / rid / "events.00.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"kind":"metric","step":300,"name":"acc","value":0.1}\n')
    assert _errors(root) == ["events.00.jsonl has a status record before the end"]


def test_verify_catches_a_duplicate_metric_key(root: Path) -> None:
    """Invariant 4: an attempt logs a key once."""
    with logmint.init(root, base_config()) as run:
        run.metric("acc", 0.5, step=1, split="forget")
        run.metric("acc", 0.6, step=1, split="forget")
    assert len(_errors(root)) == 1
    assert "duplicate metric key" in _errors(root)[0]


def test_verify_catches_a_gap_in_the_attempt_indices(root: Path) -> None:
    """Invariant 5: attempts are contiguous from zero."""
    rid = finished_run(root)
    directory = root / "runs" / rid
    (directory / "events.00.jsonl").rename(directory / "events.02.jsonl")
    assert _errors(root) == ["attempt indices are not contiguous from 0: [2]"]


def test_verify_warns_when_attempts_disagree_on_the_commit(root: Path) -> None:
    """Invariant 6: resuming across an unrelated fix is legitimate; across a
    change to the loss it is not, and only the author can tell which.
    """  # noqa: D205
    config = base_config()
    logmint.init(root, config).finish()
    directory = root / "runs" / logmint.run_id(config)
    for attempt, commit in enumerate(["cafebabe", "deadbeef"]):
        (directory / f"events.{attempt:02d}.jsonl").write_text(
            f'{{"kind":"start","time":1,"git":"{commit}"}}\n'
            '{"kind":"status","status":"finished","wall_s":1,"coords":{}}\n',
            encoding="utf-8",
        )
    problems = logmint.verify(root)
    assert [p.level for p in problems] == ["warning"]
    assert "disagree on the commit" in problems[0].message


def test_verify_catches_a_declaration_that_lies(root: Path) -> None:
    """Invariant 7: the reader trusts the declaration, so the declaration is checked."""
    config = base_config()
    with logmint.init(root, config) as run:
        run.metric("barrier", 0.5, step=1, t=0.25)
    path = root / "runs" / logmint.run_id(config) / "events.00.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[-1] = json.dumps({
        "kind": "status",
        "status": "finished",
        "wall_s": 1,
        "coords": {},
    })
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert _errors(root) == ["events.00.jsonl declares coordinates [] but uses ['t']"]


def test_the_generated_schema_cannot_drift(root: Path) -> None:
    """SCHEMA.md is rendered from the constants the writer and reader use."""
    path = logmint.write_schema(root)
    text = path.read_text(encoding="utf-8")
    for column in CORE:
        assert f"| `{column}` |" in text
    for name in RESERVED:
        assert f"`{name}`" in text
    assert "blob:[0-9a-f]{64}" in text


def test_verify_reports_an_invalid_config_instead_of_crashing(root: Path) -> None:
    """A hand-written or migrated corpus holds anything; diagnosing that is the job."""
    text = json.dumps(
        {"nested": {"a": 2}, "step": 1}, sort_keys=True, separators=(",", ":")
    )
    rid = hashlib.sha256(text.encode()).hexdigest()[:16]
    directory = root / "runs" / rid
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(text, encoding="utf-8")

    problems = _errors(root)
    assert len(problems) == 1
    assert "not a valid config" in problems[0]


def test_verify_catches_a_coordinate_that_shadows_a_config_key(root: Path) -> None:
    """The frame joins the config onto the record, so two columns are named `lr`."""
    config = base_config()
    with logmint.init(root, config) as run:
        run.metric("sensitivity", 0.3, step=1)
    path = root / "runs" / logmint.run_id(config) / "events.00.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[-1] = json.dumps({
        "kind": "status",
        "status": "finished",
        "wall_s": 1,
        "coords": {"lr": "DOUBLE"},
    })
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert any("as coordinates" in message for message in _errors(root))


def test_verify_catches_a_run_directory_with_no_config(root: Path) -> None:
    """A directory that is not a run is a directory that should not be there."""
    (root / "runs" / ("a" * 16)).mkdir(parents=True)
    assert _errors(root) == ["no run.json"]


def test_verify_catches_a_config_that_does_not_parse(root: Path) -> None:
    """Half a config is not a config."""
    directory = root / "runs" / ("b" * 16)
    directory.mkdir(parents=True)
    (directory / "run.json").write_text('{"method":"scr', encoding="utf-8")
    assert len(_errors(root)) == 1
    assert "does not parse" in _errors(root)[0]


def test_verify_catches_a_terminated_attempt_that_declared_nothing(root: Path) -> None:
    """The reader learns the schema from the declaration; one without it is a hole."""
    rid = finished_run(root)
    path = root / "runs" / rid / "events.00.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[-1] = json.dumps({"kind": "status", "status": "finished", "wall_s": 1})
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert _errors(root) == [
        "events.00.jsonl terminated without declaring its coordinates"
    ]


def test_verify_reads_a_corpus_whose_last_line_was_cut(root: Path) -> None:
    """The corruption a crash can produce is not a corpus error and stops nothing."""
    config = base_config()
    run = logmint.init(root, config)
    run.metric("acc", 0.5, step=1)
    run.close()
    path = root / "runs" / logmint.run_id(config) / "events.00.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"kind":"metric","step":2,"na')

    assert logmint.verify(root) == []


def test_verify_catches_two_status_records_in_one_attempt(root: Path) -> None:
    """An attempt terminates once. Twice means something appended after the end."""
    rid = finished_run(root)
    path = root / "runs" / rid / "events.00.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({
                "kind": "status",
                "status": "finished",
                "wall_s": 1,
                "coords": {},
            })
            + "\n"
        )

    assert any("2 status records" in message for message in _errors(root))
