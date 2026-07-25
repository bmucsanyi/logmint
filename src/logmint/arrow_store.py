"""Restartable authenticated Arrow streams."""

import os
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, Never

import pyarrow as pa
from blake3 import blake3
from pyarrow import ipc

from logmint import storage
from logmint.storage import _identity, _records, _seal, _stored_file, _sync

__all__ = ["ArrowStream", "ArrowStreamWriter"]

_BATCH_ROWS = 2_048
_SHARD_ROWS = 65_536
_STATE_FORMAT = "tda-arrow-stream-state-v6"
_STATE_NAME = "progress.json"
_STAGED_NAME = ".staged.arrow"


def _raise(error: type[Exception], message: str) -> Never:
    raise error(message)


_invalid = partial(_raise, ValueError)
_fail = partial(_raise, storage.StorageError)


@dataclass(frozen=True, slots=True)
class _Checkpoint:
    files: tuple[storage.StoredFile, ...]
    rows: int
    batch_count: int
    identity: str


@dataclass(frozen=True, slots=True)
class _State:
    current: _Checkpoint
    predecessor: _Checkpoint | None


@dataclass(slots=True)
class _Stage:
    stream: BinaryIO
    writer: ipc.RecordBatchFileWriter
    rows: int
    batches: int

    @classmethod
    def create(cls, path: Path, schema: pa.Schema) -> "_Stage":
        stream = path.open("xb")

        return cls(stream, ipc.new_file(stream, schema), 0, 0)

    def append(self, records: tuple[pa.RecordBatch, ...]) -> None:
        for batch in records:
            self.writer.write_batch(batch)
            self.rows += batch.num_rows
            self.batches += 1

    def close(self) -> None:
        self.writer.close()
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()


@dataclass(frozen=True, slots=True)
class ArrowStream:
    """One immutable Arrow stream and its restart state."""

    root: Path
    format_name: str
    identity: Mapping[str, str]
    schema: pa.Schema
    rows: int
    _journal_identity: Mapping[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and freeze the stream description."""
        valid_format = (
            bool(self.format_name)
            and self.format_name.isascii()
            and "\x00" not in self.format_name
        )
        valid_schema = bool(self.schema.names) and len(set(self.schema.names)) == len(
            self.schema.names
        )
        valid_rows = self.rows > 0

        if (
            not self.root.is_absolute()
            or not valid_format
            or not valid_schema
            or not valid_rows
        ):
            _invalid("Arrow stream specification is invalid")

        identity = MappingProxyType(dict(self.identity))
        object.__setattr__(self, "identity", identity)
        value = {
            "format": self.format_name,
            "identity": _identity(identity),
            "rows": self.rows,
            "schema_blake3": blake3(self.schema.serialize().to_pybytes()).hexdigest(),
        }
        digest = blake3(storage.canonical_json(value)).hexdigest()
        object.__setattr__(
            self,
            "_journal_identity",
            MappingProxyType({"stream_blake3": digest}),
        )

    @property
    def pending_path(self) -> Path:
        """Mutable directory beside the final root."""
        return self.root.with_name(f"{self.root.name}.partial")

    def load_state(self, root: Path) -> storage.JournalValue[_State]:
        """Return the newest authenticated stream state."""
        return storage.load_journal(
            root / _STATE_NAME,
            _STATE_FORMAT,
            self._journal_identity,
            _State,
        )

    def promote_state(
        self,
        root: Path,
        generation: int,
        state: _State,
    ) -> storage.JournalValue[_State]:
        """Publish the next authenticated stream state.

        Returns:
            The promoted journal value.
        """
        return storage.replace_journal(
            root / _STATE_NAME,
            _STATE_FORMAT,
            self._journal_identity,
            generation,
            state,
        )

    def _validate_checkpoint(
        self,
        checkpoint: _Checkpoint,
        statistics: tuple[tuple[int, int], ...],
        *,
        sealed: bool,
    ) -> None:
        storage.Digest.require_hex(checkpoint.identity, 64, "checkpoint")

        if (
            checkpoint.rows not in range(self.rows + 1)
            or checkpoint.batch_count < 0
            or len(statistics) != len(checkpoint.files)
            or sum(value[0] for value in statistics) != checkpoint.rows
            or sum(value[1] for value in statistics) != checkpoint.batch_count
        ):
            _fail("Arrow checkpoint values are inconsistent")

        prefix = "shard" if sealed else "segment"

        if any(
            descriptor.path != f"{prefix}-{index:06d}.arrow"
            for index, descriptor in enumerate(checkpoint.files)
        ):
            _fail("Arrow file order is inconsistent")

        if sealed:
            expected = (checkpoint.rows + _SHARD_ROWS - 1) // _SHARD_ROWS

            if len(checkpoint.files) != expected or any(
                rows
                != (
                    checkpoint.rows - _SHARD_ROWS * index
                    if index == expected - 1
                    else _SHARD_ROWS
                )
                for index, (rows, _) in enumerate(statistics)
            ):
                _fail("Arrow shard row count differs")

    def _verify_state(self, root: Path, state: _State, *, sealed: bool) -> None:
        latest = state.current
        predecessor = state.predecessor

        if sealed and predecessor is not None:
            _fail("sealed Arrow stream retains rollback state")

        if predecessor is not None and (
            predecessor.rows >= latest.rows
            or predecessor.batch_count >= latest.batch_count
            or predecessor.identity == latest.identity
            or predecessor.files != latest.files[: len(predecessor.files)]
        ):
            _fail("Arrow checkpoint history is inconsistent")

        statistics = []

        for descriptor in latest.files:
            path = root / descriptor.path

            if _stored_file(root, path) != descriptor:
                _fail(f"Arrow file differs from its descriptor: {descriptor.path}")

            rows = 0
            batches = 0

            for batch in _arrow_file(path, self.schema):
                rows += batch.num_rows
                batches += 1

            statistics.append((rows, batches))

        values = tuple(statistics)
        self._validate_checkpoint(latest, values, sealed=sealed)

        if predecessor is not None:
            self._validate_checkpoint(
                predecessor,
                values[: len(predecessor.files)],
                sealed=False,
            )

    def open(self, checkpoint_identity: str) -> "ArrowStreamWriter":
        """Open mutable state or verify a published stream.

        Returns:
            The recovered writer bound to the requested checkpoint.
        """
        final = storage.publication_root_exists(self.root)
        partial = storage.publication_root_exists(self.pending_path)

        if final and partial:
            _fail("Arrow stream has final and partial directories")

        if final:
            manifest = storage.verify_directory(
                self.root,
                self.format_name,
                self.identity,
            )
            journal = self.load_state(self.root)
            self._verify_state(self.root, journal.value, sealed=True)
            writer = ArrowStreamWriter(
                self,
                journal,
                manifest_blake3=manifest.blake3,
            )
            latest = journal.value.current

            if latest.identity != checkpoint_identity or latest.rows != self.rows:
                _fail("sealed Arrow stream differs from its final checkpoint")

            return writer

        if partial and (self.pending_path / "manifest.json").exists():
            storage.verify_directory(
                self.pending_path,
                self.format_name,
                self.identity,
            )
            self.pending_path.rename(self.root)
            _sync(self.root.parent)

            return self.open(checkpoint_identity)

        if partial:
            journal = self.load_state(self.pending_path)
            latest = journal.value.current
            sealing = bool(latest.files) and latest.files[0].path.startswith("shard-")
            self._verify_state(self.pending_path, journal.value, sealed=sealing)

            if sealing:
                if latest.identity != checkpoint_identity:
                    _fail("finalizing Arrow stream differs from its checkpoint")

                journal, manifest = _seal_stream(self, journal)

                return ArrowStreamWriter(
                    self,
                    journal,
                    manifest_blake3=manifest.blake3,
                )

            _remove_unreferenced(self.pending_path, journal.value)
            writer = ArrowStreamWriter(self, journal)
            writer.rollback(checkpoint_identity)

            return writer

        storage.Digest.require_hex(checkpoint_identity, 64, "checkpoint")
        self.pending_path.parent.mkdir(parents=True, exist_ok=True)
        self.pending_path.mkdir()
        checkpoint = _Checkpoint((), 0, 0, checkpoint_identity)
        journal = self.promote_state(
            self.pending_path,
            -1,
            _State(checkpoint, None),
        )
        _sync(self.pending_path)

        return ArrowStreamWriter(self, journal)


def _arrow_file(path: Path, schema: pa.Schema) -> Iterator[pa.RecordBatch]:
    if path.is_symlink() or not path.is_file():
        _fail(f"Arrow file is not regular: {path}")

    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)

        if not reader.schema.equals(schema, check_metadata=True):
            _fail(f"Arrow file schema differs: {path}")

        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)

            if batch.num_rows not in range(1, _BATCH_ROWS + 1):
                _fail(f"Arrow record-batch row count differs: {path}")

            yield batch


def _partition(
    batches: Iterable[pa.RecordBatch],
    rows: int,
) -> Iterator[tuple[pa.RecordBatch, ...]]:
    selected = []
    selected_rows = 0

    for batch in batches:
        offset = 0

        while offset < batch.num_rows:
            count = min(rows - selected_rows, batch.num_rows - offset)
            selected.append(batch.slice(offset, count))
            selected_rows += count
            offset += count

            if selected_rows == rows:
                yield tuple(selected)
                selected = []
                selected_rows = 0

    if selected:
        yield tuple(selected)


def _remove_unreferenced(root: Path, state: _State) -> None:
    referenced = {
        *(descriptor.path for descriptor in state.current.files),
        ".progress.json.generation-0",
        ".progress.json.generation-1",
    }
    removed = False

    for path in root.iterdir():
        if path.name in referenced or path.name == "manifest.json":
            continue

        recognized = (
            path.name == _STAGED_NAME
            or path.name.startswith("segment-")
            or path.name.startswith("shard-")
        )

        if not recognized or path.is_symlink() or not path.is_file():
            _fail(f"Arrow stream contains an unexpected file: {path}")

        path.unlink()
        removed = True

    if removed:
        _sync(root)


def _seal_stream(
    stream: ArrowStream,
    journal: storage.JournalValue[_State],
) -> tuple[storage.JournalValue[_State], storage.DirectoryManifest]:
    state = journal.value
    latest = state.current

    if state.predecessor is not None or latest.rows != stream.rows:
        _fail("Arrow final state is inconsistent")

    promoted = stream.promote_state(
        stream.pending_path,
        journal.generation,
        state,
    )
    _remove_unreferenced(stream.pending_path, state)
    manifest = _seal(
        stream.root,
        stream.pending_path,
        stream.format_name,
        _identity(stream.identity),
        _records({
            "batches": latest.batch_count,
            "rows": stream.rows,
            "shards": len(latest.files),
        }),
    )

    return promoted, manifest


@dataclass(slots=True)
class ArrowStreamWriter:
    """Authenticated segment writer with checkpoint rollback."""

    stream: ArrowStream
    _journal: storage.JournalValue[_State]
    _stage: _Stage | None = None
    manifest_blake3: str | None = None

    @property
    def pending(self) -> Path:
        """Partial stream directory."""
        return self.stream.pending_path

    @property
    def rows(self) -> int:
        """Current staged and committed row count."""
        staged_rows = 0 if self._stage is None else self._stage.rows

        return self.committed_rows + staged_rows

    @property
    def committed_rows(self) -> int:
        """Current committed row count."""
        return self._journal.value.current.rows

    @property
    def checkpoint_identity(self) -> str:
        """Identity of the current committed prefix."""
        return self._journal.value.current.identity

    def _promote(self, state: _State) -> None:
        self._journal = self.stream.promote_state(
            self.pending,
            self._journal.generation,
            state,
        )

    def _retain_checkpoint(self, checkpoint: _Checkpoint) -> None:
        retained = _State(checkpoint, None)
        self._promote(retained)
        _remove_unreferenced(self.pending, retained)

    def _discard_stage(self) -> None:
        if self._stage is not None:
            self._stage.close()
            self._stage = None

        path = self.pending / _STAGED_NAME

        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                _fail(f"staged Arrow file is not regular: {path}")

            path.unlink()
            _sync(self.pending)

    def append(
        self,
        batches: Iterable[pa.RecordBatch],
    ) -> None:
        """Stage complete batches until the next checkpoint."""
        records = tuple(batches)

        if not records:
            _invalid("Arrow transaction cannot be empty")

        if any(
            batch.num_rows not in range(1, _BATCH_ROWS + 1)
            or not batch.schema.equals(self.stream.schema, check_metadata=True)
            for batch in records
        ):
            _invalid("Arrow batch has invalid rows or schema")

        final_rows = self.rows + sum(batch.num_rows for batch in records)

        if final_rows > self.stream.rows:
            _invalid("Arrow append exceeds the registered row count")

        if self._stage is None:
            self._stage = _Stage.create(
                self.pending / _STAGED_NAME,
                self.stream.schema,
            )

        self._stage.append(records)

    def checkpoint(self, identity: str) -> None:
        """Commit the staged prefix as one immutable segment."""
        storage.Digest.require_hex(identity, 64, "checkpoint")

        if self._stage is None or self._stage.rows == 0:
            _invalid("Arrow checkpoint requires staged rows")

        state = self._journal.value

        if state.predecessor is not None:
            _invalid("Arrow checkpoint requires retaining its predecessor")

        stage = self._stage
        stage.close()
        self._stage = None
        previous = state.current
        path = self.pending / f"segment-{len(previous.files):06d}.arrow"

        if path.exists() or path.is_symlink():
            _fail(f"Arrow segment already exists: {path}")

        (self.pending / _STAGED_NAME).rename(path)
        _sync(self.pending)
        checkpoint = _Checkpoint(
            (*previous.files, _stored_file(self.pending, path)),
            previous.rows + stage.rows,
            previous.batch_count + stage.batches,
            identity,
        )
        self._promote(_State(checkpoint, previous))

    def rollback(self, identity: str) -> None:
        """Roll back staged and committed rows to one retained checkpoint."""
        storage.Digest.require_hex(identity, 64, "checkpoint")
        self._discard_stage()
        state = self._journal.value

        if state.current.identity == identity:
            return

        checkpoint = state.predecessor

        if checkpoint is None or checkpoint.identity != identity:
            _fail("Arrow checkpoint identity is absent")

        self._retain_checkpoint(checkpoint)

    def retain(self, identity: str) -> None:
        """Discard rollback state older than the active checkpoint."""
        latest = self._journal.value.current

        if latest.identity != identity:
            _fail("Arrow retained checkpoint is not active")

        if self._journal.value.predecessor is None:
            return

        self._retain_checkpoint(latest)

    def batches(self, checkpoint_identity: str) -> Iterator[pa.RecordBatch]:
        """Yield one authenticated committed prefix."""
        root = self.stream.root if self.manifest_blake3 is not None else self.pending
        checkpoint = self._journal.value.current

        if checkpoint.identity != checkpoint_identity:
            _fail("Arrow checkpoint is not the active prefix")

        for descriptor in checkpoint.files:
            yield from _arrow_file(root / descriptor.path, self.stream.schema)

    def finalize(
        self,
        checkpoint_identity: str,
    ) -> None:
        """Compact and publish all committed rows as fixed-size Arrow shards."""
        latest = self._journal.value.current

        if (
            self._stage is not None
            or latest.rows != self.stream.rows
            or latest.identity != checkpoint_identity
        ):
            _invalid("Arrow stream is incomplete")

        shards = []
        rows = 0
        batches = 0

        for index, records in enumerate(
            _partition(
                (
                    batch
                    for descriptor in latest.files
                    for batch in _arrow_file(
                        self.pending / descriptor.path,
                        self.stream.schema,
                    )
                ),
                _SHARD_ROWS,
            )
        ):
            path = self.pending / f"shard-{index:06d}.arrow"
            shard = _Stage.create(path, self.stream.schema)

            try:
                shard.append(records)
            finally:
                shard.close()

            rows += shard.rows
            batches += shard.batches
            shards.append(_stored_file(self.pending, path))

        if rows != latest.rows:
            _fail("Arrow final compaction changed the committed row count")

        sealed = replace(latest, files=tuple(shards), batch_count=batches)
        final_state = _State(sealed, None)
        self._promote(final_state)
        self._journal, manifest = _seal_stream(
            self.stream,
            self._journal,
        )
        self.manifest_blake3 = manifest.blake3
