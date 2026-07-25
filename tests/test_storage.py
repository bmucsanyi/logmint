"""Behavioral tests for authenticated publications and Arrow restarts."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pytest
from numpy.testing import assert_array_equal

from logmint import arrow_store, storage


@dataclass(frozen=True, slots=True)
class _Payload:
    step: int
    label: str


_FORMAT = "test-program-store-v1"
_IDENTITY = {"implementation": "a" * 40, "spec": "b" * 64}
_SCHEMA = pa.schema((pa.field("value", pa.int64(), nullable=False),))


def _batch(values: npt.ArrayLike) -> pa.RecordBatch:
    return pa.record_batch([pa.array(values)], schema=_SCHEMA)


def _stream(root: Path, name: str, rows: int) -> arrow_store.ArrowStream:
    return arrow_store.ArrowStream(
        root,
        f"test-{name}-v1",
        _IDENTITY,
        _SCHEMA,
        rows,
    )


def _values(
    writer: arrow_store.ArrowStreamWriter,
    checkpoint: str,
) -> tuple[int, ...]:
    return tuple(
        value
        for batch in writer.batches(checkpoint)
        for value in batch.column(0).to_pylist()
    )


def _append(
    writer: arrow_store.ArrowStreamWriter,
    *rows: npt.ArrayLike,
) -> None:
    writer.append(tuple(_batch(values) for values in rows))


def test_two_slot_journal_promotes_recovers_and_rejects_stale_writer(
    tmp_path: Path,
) -> None:
    path = tmp_path / "progress.json"
    first = storage.replace_journal(path, _FORMAT, _IDENTITY, -1, _Payload(1, "first"))
    second = storage.replace_journal(
        path,
        _FORMAT,
        _IDENTITY,
        first.generation,
        _Payload(2, "second"),
    )

    assert (first.generation, second.generation) == (0, 1)
    assert storage.load_journal(path, _FORMAT, _IDENTITY, _Payload) == second

    with pytest.raises(ValueError, match="changed before promotion"):
        storage.replace_journal(
            path,
            _FORMAT,
            _IDENTITY,
            first.generation,
            _Payload(3, "stale"),
        )

    newest = path.with_name(f".{path.name}.generation-{second.generation % 2}")
    newest.write_bytes(b"{}\n")
    assert storage.load_journal(path, _FORMAT, _IDENTITY, _Payload) == first


def test_journal_rejects_symlink_slot(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    slot = path.with_name(f".{path.name}.generation-0")
    slot.symlink_to(tmp_path / "missing")

    with pytest.raises(ValueError, match="not a regular file"):
        storage.load_journal(path, _FORMAT, _IDENTITY, _Payload)


def test_array_bundle_authenticates_values_and_rejects_changed_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "arrays"
    counts = np.asarray([2, 3, 5, 7], dtype=np.int64)
    scores = np.asarray([[0.25, -1.5], [3.0, 4.5]], dtype=np.float32)
    manifest = storage.publish_arrays(
        root,
        "test-arrays-v1",
        _IDENTITY,
        {"counts": counts, "scores": scores},
    )
    loaded = storage.load_array_subset(
        root,
        "test-arrays-v1",
        _IDENTITY,
        ("scores",),
    )
    assert manifest.identity == tuple(sorted(_IDENTITY.items()))
    assert dict(manifest.records) == {"arrays": 2}
    assert tuple(loaded) == ("scores",)
    assert_array_equal(loaded["scores"], scores)
    assert not loaded["scores"].flags.writeable

    changed_path = root / "array-000.npy"
    changed = bytearray(changed_path.read_bytes())
    changed[-1] ^= 1
    changed_path.write_bytes(changed)

    with pytest.raises(storage.StorageError, match="inventory differs"):
        storage.load_arrays(root, "test-arrays-v1", _IDENTITY)


def test_arrow_stream_discards_interrupted_shard_and_replays_exactly(
    tmp_path: Path,
) -> None:
    rows = 65_538
    root = tmp_path / "stream"
    stream = _stream(root, "arrow", rows)
    initial = "4" * 64
    final = "5" * 64
    writer = stream.open(initial)
    staged = writer.pending / ".staged.arrow"
    staged.write_bytes(b"interrupted Arrow IPC bytes")
    interrupted_shard = writer.pending / "shard-000000.arrow"
    interrupted_shard.write_bytes(b"interrupted Arrow shard")
    resumed = stream.open(initial)
    assert not staged.exists()
    assert not interrupted_shard.exists()

    batches = tuple(
        _batch(np.arange(start, min(start + 2_048, rows)))
        for start in range(0, rows, 2_048)
    )
    resumed.append(batches)
    resumed.checkpoint(final)
    resumed.finalize(final)
    manifest = storage.load_manifest(root / "manifest.json")
    assert dict(manifest.records) == {"batches": 33, "rows": rows, "shards": 2}
    sealed = stream.open(final)
    observed = np.concatenate(
        tuple(batch.column(0).to_numpy() for batch in sealed.batches(final))
    )
    assert_array_equal(observed, np.arange(rows))
    baseline = _stream(tmp_path / "baseline", "arrow", rows)
    baseline_writer = baseline.open(initial)
    baseline_writer.append(batches)
    baseline_writer.checkpoint(final)
    baseline_writer.finalize(final)

    for index in range(2):
        name = f"shard-{index:06d}.arrow"
        assert (root / name).read_bytes() == (baseline.root / name).read_bytes()

    (root / "manifest.json").unlink()
    root.rename(stream.pending_path)
    recovered = stream.open(final)
    assert _values(recovered, final) == tuple(range(rows))


def test_arrow_stream_rolls_back_and_replays_retained_checkpoint(
    tmp_path: Path,
) -> None:
    stream = _stream(tmp_path / "stream", "rollback", 8)
    initial = "6" * 64
    retained = "7" * 64
    final = "8" * 64
    writer = stream.open(initial)
    _append(writer, [0, 1], [2, 3])
    writer.checkpoint(retained)
    writer.retain(retained)
    _append(writer, [4, 5], [6, 7])
    writer.checkpoint(final)

    recovered = stream.open(retained)
    assert recovered.committed_rows == 4
    assert _values(recovered, retained) == (0, 1, 2, 3)
    _append(recovered, [4, 5], [6, 7])
    recovered.checkpoint(final)
    recovered.retain(final)
    recovered.finalize(final)
    assert _values(stream.open(final), final) == tuple(range(8))


def test_arrow_stream_rejects_changed_published_bytes(tmp_path: Path) -> None:
    stream = _stream(tmp_path / "stream", "tamper", 4)
    initial = "9" * 64
    final = "a" * 64
    writer = stream.open(initial)
    _append(writer, [0, 1, 2, 3])
    writer.checkpoint(final)
    writer.retain(final)
    writer.finalize(final)
    shard = stream.root / "shard-000000.arrow"
    changed = bytearray(shard.read_bytes())
    changed[len(changed) // 2] ^= 1
    shard.write_bytes(changed)

    with pytest.raises(storage.StorageError, match="inventory differs"):
        stream.open(final)
