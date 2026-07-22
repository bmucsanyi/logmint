"""The content-addressed blob store (spec section 4)."""

from pathlib import Path

import numpy as np
import pytest

import logmint
from logmint import _blobs
from tests.conftest import base_config


def test_put_returns_a_reference_and_get_round_trips_an_array(root: Path) -> None:
    """An array survives a write and a read."""
    array = np.arange(10, dtype=np.float32)
    ref = logmint.put(root, array)
    assert ref.startswith("blob:")
    np.testing.assert_array_equal(logmint.get(root, ref), array)


def test_bytes_round_trip(root: Path) -> None:
    """Raw bytes come back as raw bytes."""
    ref = logmint.put(root, b"\x00\x01checkpoint")
    assert logmint.get(root, ref) == b"\x00\x01checkpoint"


def test_a_file_on_disk_can_be_stored_without_being_loaded(
    root: Path, tmp_path: Path
) -> None:
    """A checkpoint from another library is streamed in, not loaded into memory."""
    source = tmp_path / "model.pt"
    source.write_bytes(b"weights" * 1000)
    ref = logmint.put(root, source)
    assert logmint.get_path(root, ref).read_bytes() == source.read_bytes()


def test_identical_bytes_are_stored_once(root: Path) -> None:
    """The name is the content, so a shared checkpoint is stored once."""
    first = logmint.put(root, np.zeros(4))
    second = logmint.put(root, np.zeros(4))
    assert first == second
    assert len(_blobs.stored(root)) == 1


def test_path_is_a_pure_function_of_the_reference(root: Path) -> None:
    """A reference resolves without knowing the format, so its path has no suffix."""
    ref = logmint.put(root, np.zeros(2))
    digest = ref.removeprefix("blob:")
    assert _blobs.path_for(root, ref) == root / "blobs" / digest[:2] / digest[2:]


def test_no_temporary_file_survives_a_write(root: Path) -> None:
    """The staging directory is empty once the write commits."""
    logmint.put(root, np.zeros(3))
    assert list((root / "blobs" / "tmp").iterdir()) == []


def test_malformed_reference_is_refused(root: Path) -> None:
    """A reference that is not blob:<64 hex> never becomes a path."""
    with pytest.raises(logmint.BlobError, match="malformed"):
        _blobs.path_for(root, "blob:nope")


def test_dangling_reference_is_refused(root: Path) -> None:
    """Resolving a reference to nothing is an error, not an empty result."""
    with pytest.raises(logmint.BlobError, match="does not resolve"):
        logmint.get(root, "blob:" + "0" * 64)


def test_an_input_blob_is_part_of_the_run_identity(root: Path) -> None:
    """Two runs differing only in a non-scalar input are different runs."""
    one = logmint.put(root, np.array([0, 1, 2]))
    other = logmint.put(root, np.array([3, 4, 5]))
    assert logmint.run_id(base_config(subset=one)) != logmint.run_id(
        base_config(subset=other)
    )


def test_reachability_covers_inputs_and_outputs(root: Path) -> None:
    """A blob is reachable from a config value or a blob record, and nowhere else."""
    subset = logmint.put(root, np.array([1, 2, 3]))
    with logmint.init(root, base_config(subset=subset)) as run:
        output = run.blob("eigvals", np.ones(5), step=1)
    orphan = logmint.put(root, b"nobody references this")

    assert logmint.reachable(root) == {subset, output}
    assert orphan in _blobs.stored(root)


def test_gc_removes_only_unreachable_blobs(root: Path) -> None:
    """Collection keeps every referenced blob and deletes the rest."""
    subset = logmint.put(root, np.array([1, 2, 3]))
    with logmint.init(root, base_config(subset=subset)) as run:
        output = run.blob("eigvals", np.ones(5), step=1)
    orphan = logmint.put(root, b"orphan")

    assert logmint.gc(root, dry_run=True, grace_s=0) == [orphan]
    assert _blobs.stored(root) == {subset, output, orphan}

    assert logmint.gc(root, grace_s=0) == [orphan]
    assert _blobs.stored(root) == {subset, output}
    np.testing.assert_array_equal(logmint.get(root, output), np.ones(5))


def test_blob_record_carries_shape_and_dtype(root: Path) -> None:
    """An output blob records enough metadata to be legible without loading it."""
    with logmint.init(root, base_config()) as run:
        run.blob("eigvals", np.ones(7, dtype=np.float32), step=1)
    frame = logmint.query(root, "SELECT * FROM events WHERE kind = 'blob'")
    assert frame.height == 1
    assert frame["ref"][0].startswith("blob:")


def test_gc_reclaims_a_staging_file_a_killed_writer_left_behind(root: Path) -> None:
    """A put that dies between opening and renaming its staging file leaves it."""
    logmint.put(root, np.zeros(3))
    abandoned = root / "blobs" / "tmp" / "deadbeef"
    abandoned.write_bytes(b"half a checkpoint")
    assert _blobs.staged(root) == [abandoned]

    logmint.gc(root, grace_s=0)
    assert _blobs.staged(root) == []


def test_gc_does_not_leave_empty_directories(root: Path) -> None:
    """A fan-out directory whose last blob was collected goes with it."""
    ref = logmint.put(root, b"orphan")
    fanout = _blobs.path_for(root, ref).parent
    logmint.gc(root, grace_s=0)
    assert not fanout.exists()


def test_put_on_a_missing_file_is_a_logmint_error(root: Path) -> None:
    """Not an OSError from three frames down."""
    with pytest.raises(logmint.BlobError, match="not a file"):
        logmint.put(root, Path("/nonexistent/checkpoint.pt"))


def test_a_large_blob_is_not_read_into_memory_to_identify_it(root: Path) -> None:
    """The magic bytes are sniffed from the handle; get_path never reads the file."""
    ref = logmint.put(root, np.arange(1000, dtype=np.int64))
    array = logmint.get(root, ref)
    assert isinstance(array, np.ndarray)
    np.testing.assert_array_equal(array, np.arange(1000))
    assert logmint.get_path(root, ref).stat().st_size > 8000


def test_an_uppercase_reference_is_refused(root: Path) -> None:
    """The reference notation is lowercase hex, and grep is the reachability rule."""
    with pytest.raises(logmint.BlobError, match="lowercase hex"):
        _blobs.path_for(root, "blob:" + "A" * 64)


def test_an_unsupported_type_is_refused(root: Path) -> None:
    """A string is not a blob. Wrap a filename in Path if that is what it is."""
    with pytest.raises(logmint.BlobError, match="cannot store an object of type str"):
        logmint.put(root, "checkpoints/model.pt")  # ty: ignore[invalid-argument-type]


def test_staged_on_an_empty_store_is_empty(root: Path) -> None:
    """A corpus with no blobs has nothing to reclaim."""
    assert _blobs.staged(root) == []
    assert logmint.gc(root, grace_s=0) == []


def test_a_blob_is_not_collected_before_its_reference_is_written(root: Path) -> None:
    """A blob is written before the record that references it, and an input blob
    before the config that names it. A collector running against a live corpus
    must not delete either.
    """  # noqa: D205
    ref = logmint.put(root, np.arange(4))
    assert logmint.gc(root) == []
    assert logmint.get(root, ref) is not None

    config = {**base_config(), "subset": ref}
    with logmint.init(root, config) as run:
        run.metric("acc", 0.5, step=1)
    assert logmint.gc(root, grace_s=0) == []


def test_a_blob_unreachable_past_the_grace_period_is_collected(root: Path) -> None:
    """Once nothing can come to reference it, it is garbage."""
    ref = logmint.put(root, np.arange(4))
    assert logmint.gc(root, grace_s=0) == [ref]
    with pytest.raises(logmint.BlobError, match="does not resolve"):
        logmint.get(root, ref)
