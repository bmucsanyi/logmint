"""Authenticated filesystem publication, typed journals, and numeric arrays."""

import hashlib
import json
import os
import resource
import shutil
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from functools import partial
from itertools import chain, starmap
from pathlib import Path, PurePosixPath
from types import MappingProxyType, UnionType
from typing import (
    Any,
    NamedTuple,
    Never,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import numpy as np
import numpy.typing as npt
from blake3 import blake3

__all__ = [
    "CanonicalJSON",
    "Digest",
    "DirectoryManifest",
    "JournalValue",
    "StorageError",
    "StoredFile",
    "canonical_json",
    "file_sha256",
    "host_peak_bytes",
    "immutable_array",
    "load_array_subset",
    "load_arrays",
    "load_journal",
    "load_manifest",
    "named_array_blake3",
    "publication_root_exists",
    "publish_arrays",
    "publish_directory",
    "read_json",
    "read_numpy",
    "read_numpy_exact",
    "replace_journal",
    "tree_bytes",
    "verify_directory",
    "write_bytes",
    "write_json",
    "write_numpy",
]

_MANIFEST = "manifest.json"
_BLOCK_BYTES = 8 * 1_024 * 1_024
_HEX_LENGTH = 64
_JOURNAL_SLOTS = 2
_PAIR_FIELDS = 2
type _Builder = Callable[[Path], Mapping[str, int]]
type ResultArray = npt.NDArray[np.generic]


class StorageError(RuntimeError):
    """Invalid, incomplete, or mismatched stored content."""


@dataclass(frozen=True, slots=True)
class StoredFile:
    """One regular file sealed by byte count and BLAKE3."""

    path: str
    size: int
    blake3: str

    def __post_init__(self) -> None:
        """Validate one canonical manifest entry."""
        relative = PurePosixPath(self.path)

        if not self.path or self.path == _MANIFEST or "\x00" in self.path:
            _invalid("publication file entry is invalid")

        if (
            relative.is_absolute()
            or relative.as_posix() != self.path
            or ".." in relative.parts
        ):
            _invalid("publication file entry is invalid")

        if self.size < 0:
            _invalid("publication file entry is invalid")

        Digest.require_hex(self.blake3, _HEX_LENGTH, "file BLAKE3")


@dataclass(frozen=True, slots=True)
class DirectoryManifest:
    """Exact identity and file inventory of one published directory."""

    format: str
    identity: tuple[tuple[str, str], ...]
    records: tuple[tuple[str, int], ...]
    files: tuple[StoredFile, ...]
    blake3: str


class JournalValue[T](NamedTuple):
    """Newest authenticated mutable-journal generation."""

    generation: int
    value: T


def _fail(error: type[Exception], message: str) -> Never:
    raise error(message)


_invalid = partial(_fail, ValueError)
_corrupt = partial(_fail, StorageError)
_wrong_type = partial(_fail, TypeError)


def _sync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)

    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _identity(value: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    valid = bool(value) and all(
        bool(key) and bool(item) and "\x00" not in key + item
        for key, item in value.items()
    )

    if not valid:
        _invalid("publication identity requires nonempty NUL-free strings")

    return tuple(sorted(value.items()))


def _records(value: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    valid = all(
        bool(key) and "\x00" not in key and item >= 0 for key, item in value.items()
    )

    if not valid:
        _invalid("publication records require nonnegative integer counts")

    return tuple(sorted(value.items()))


def canonical_json(value: object) -> bytes:
    """Return finite, sorted, newline-terminated JSON bytes."""
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def named_array_blake3(values: Sequence[tuple[str, np.ndarray]]) -> str:
    """Return the BLAKE3 identity of an ordered sequence of named arrays."""
    digest = blake3()

    for name, value in values:
        array = np.ascontiguousarray(value)
        encoded = name.encode("ascii")
        digest.update(len(encoded).to_bytes(8, byteorder="little"))
        digest.update(encoded)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(array).cast("B"))

    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree_bytes(root: Path) -> int:
    """Return regular-file bytes below a symlink-free directory."""
    if root.is_symlink() or not root.is_dir():
        _invalid("tree root must be an existing directory")

    paths = tuple(root.rglob("*"))

    if any(path.is_symlink() for path in paths):
        _invalid("tree contains a symlink")

    return sum(path.stat().st_size for path in paths if path.is_file())


def write_bytes(path: Path, content: bytes) -> None:
    """Create and synchronize one new file."""
    try:
        with path.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        _corrupt(f"refusing to replace an existing file: {path}")


def write_json(path: Path, value: object) -> None:
    """Create one canonical JSON file."""
    write_bytes(path, canonical_json(value))


def read_json(path: Path) -> object:
    """Return one canonical JSON value."""
    encoded = path.read_bytes()
    value = json.loads(encoded)

    if encoded != canonical_json(value):
        _corrupt(f"JSON file is not canonical: {path}")

    return value


def _decode_value(value: object, annotation: object) -> object:
    origin = get_origin(annotation)

    if annotation in {str, int, bool}:
        expected = cast("type[Any]", annotation)

        if type(value) is not expected:
            _wrong_type(f"canonical JSON field must be {expected.__name__}")

        return value

    if origin is UnionType:
        arguments = get_args(annotation)
        concrete = tuple(item for item in arguments if item is not type(None))

        if len(arguments) != _PAIR_FIELDS or len(concrete) != 1:
            _wrong_type("canonical JSON supports only optional union fields")

        return None if value is None else _decode_value(value, concrete[0])

    if origin is tuple:
        if type(value) is not list:
            _wrong_type("canonical JSON tuple field must be a list")

        arguments = get_args(annotation)

        if arguments and arguments[-1] is Ellipsis:
            arguments = (arguments[0],) * len(value)

        if len(value) != len(arguments):
            _wrong_type("canonical JSON tuple field has the wrong length")

        return tuple(starmap(_decode_value, zip(value, arguments, strict=True)))

    if isinstance(annotation, type) and is_dataclass(annotation):
        if type(value) is not dict:
            _wrong_type(f"canonical JSON {annotation.__name__} must be an object")

        item = cast("dict[str, object]", value)
        names = tuple(field.name for field in fields(cast("Any", annotation)))

        if set(item) != set(names):
            _wrong_type(f"canonical JSON {annotation.__name__} fields differ")

        hints = get_type_hints(annotation)

        return annotation(*(_decode_value(item[name], hints[name]) for name in names))

    return _wrong_type(f"unsupported canonical JSON field type: {annotation}")


class CanonicalJSON:
    """Strict canonical JSON operations."""

    @staticmethod
    def decode_dataclass[T](value: object, expected: type[T]) -> T:
        """Return one strictly decoded nested dataclass value."""
        if not isinstance(expected, type) or not is_dataclass(expected):
            _wrong_type("canonical JSON target must be a dataclass type")

        return cast("T", _decode_value(value, expected))

    @staticmethod
    def read_object(path: Path) -> dict[str, object]:
        """Return one canonical JSON object."""
        if path.is_symlink() or not path.is_file():
            _corrupt(f"canonical JSON must be a regular file: {path}")

        value = read_json(path)

        if type(value) is not dict:
            _corrupt(f"canonical JSON value must be an object: {path}")

        return cast("dict[str, object]", value)


class Digest:
    """Canonical content-digest operations."""

    @staticmethod
    def require_hex(value: str, length: int, name: str) -> None:
        """Require exact lowercase hexadecimal content."""
        if len(value) != length or any(
            character not in "0123456789abcdef" for character in value
        ):
            _invalid(f"{name} must contain {length} lowercase hexadecimal digits")

    @staticmethod
    def framed_blake3(domain: bytes, values: Iterable[bytes]) -> str:
        """Return a length-framed BLAKE3 digest."""
        if not domain:
            _invalid("framed BLAKE3 requires a nonempty byte domain")

        digest = blake3()

        for value in chain((domain,), values):
            digest.update(len(value).to_bytes(8, "little"))
            digest.update(value)

        return digest.hexdigest()


def _stored_identity(identity: Mapping[str, str]) -> dict[str, str]:
    stored = dict(_identity(identity))

    if any(not key.isascii() or not value.isascii() for key, value in stored.items()):
        _invalid("stored JSON identity fields must be ASCII")

    return stored


def _document(
    format_name: str,
    identity: Mapping[str, str],
    value: object,
    generation: int | None = None,
) -> dict[str, object]:
    if not format_name or not format_name.isascii() or "\x00" in format_name:
        _invalid("stored JSON format is invalid")

    if not is_dataclass(value) or isinstance(value, type):
        _wrong_type("stored JSON payload must be a dataclass instance")

    bound: dict[str, object] = {
        "format": format_name,
        "identity": _stored_identity(identity),
        "payload": asdict(cast("Any", value)),
    }

    if generation is not None:
        bound["generation"] = generation

    return {**bound, "blake3": blake3(canonical_json(bound)).hexdigest()}


def _journal_path(path: Path, slot: int) -> Path:
    return path.with_name(f".{path.name}.generation-{slot}")


def _journal_slot[T](
    path: Path,
    format_name: str,
    identity: Mapping[str, str],
    value_type: type[T],
) -> JournalValue[T] | None:
    if path.is_symlink():
        _invalid(f"journal slot is not a regular file: {path}")

    if not path.exists():
        return None

    if not path.is_file():
        _invalid(f"journal slot is not a regular file: {path}")

    try:
        document = CanonicalJSON.read_object(path)
        generation = document["generation"]
        value = CanonicalJSON.decode_dataclass(document["payload"], value_type)
        expected = _document(format_name, identity, value, cast("int", generation))
    except (
        KeyError,
        StorageError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None

    if (
        type(generation) is not int
        or generation < 0
        or canonical_json(document) != canonical_json(expected)
    ):
        return None

    return JournalValue(generation, value)


def load_journal[T](
    path: Path,
    format_name: str,
    identity: Mapping[str, str],
    value_type: type[T],
) -> JournalValue[T]:
    """Return the newest authenticated mutable-journal generation."""
    if not format_name or not format_name.isascii() or "\x00" in format_name:
        _invalid("stored JSON format is invalid")

    _stored_identity(identity)

    if not isinstance(value_type, type) or not is_dataclass(cast("Any", value_type)):
        _wrong_type("stored JSON target must be a dataclass type")

    values = tuple(
        value
        for slot in range(_JOURNAL_SLOTS)
        if (
            value := _journal_slot(
                _journal_path(path, slot), format_name, identity, value_type
            )
        )
        is not None
    )
    generations = tuple(sorted(value.generation for value in values))

    if (
        not values
        or len(set(generations)) != len(generations)
        or (len(generations) == _JOURNAL_SLOTS and generations[1] != generations[0] + 1)
    ):
        _invalid("mutable journal has no unique authenticated generation")

    return max(values, key=lambda value: value.generation)


def replace_journal[T](
    path: Path,
    format_name: str,
    identity: Mapping[str, str],
    expected_generation: int,
    value: T,
) -> JournalValue[T]:
    """Return one atomically promoted journal generation."""
    _document(format_name, identity, value)
    current = (
        None
        if expected_generation == -1
        else load_journal(path, format_name, identity, type(value))
    )

    if expected_generation < -1 or (
        current is not None and current.generation != expected_generation
    ):
        _invalid("mutable journal generation changed before promotion")

    slots = tuple(_journal_path(path, slot) for slot in range(_JOURNAL_SLOTS))

    if expected_generation == -1 and any(
        slot.exists() or slot.is_symlink() for slot in slots
    ):
        _invalid("mutable journal already has durable state")

    generation = expected_generation + 1
    document = _document(format_name, identity, value, generation)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = slots[generation % _JOURNAL_SLOTS]

    if target.exists() or target.is_symlink():
        previous = _journal_slot(target, format_name, identity, type(value))

        if previous is not None and previous.generation != generation - 2:
            _invalid("mutable journal target generation is inconsistent")

        target.unlink()
        _sync(path.parent)

    write_json(target, document)
    _sync(path.parent)
    promoted = load_journal(path, format_name, identity, type(value))

    if promoted.generation != generation or promoted.value != value:
        _invalid("mutable journal promotion did not authenticate")

    return promoted


def _stored_file(root: Path, path: Path) -> StoredFile:
    if path.is_symlink() or not path.is_file():
        _corrupt(f"published file is not regular: {path}")

    digest = blake3()

    with path.open("rb") as stream:
        while block := stream.read(_BLOCK_BYTES):
            digest.update(block)

    return StoredFile(
        path.relative_to(root).as_posix(),
        path.stat().st_size,
        digest.hexdigest(),
    )


def _inventory(root: Path, *, synchronize: bool = False) -> tuple[StoredFile, ...]:
    result = []

    for path in sorted(root.rglob("*")):
        if path == root / _MANIFEST:
            continue

        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            _corrupt(f"published entries must be regular: {path}")

        if path.is_dir():
            if not any(path.iterdir()):
                _corrupt(f"published directories cannot be empty: {path}")

            continue

        if synchronize:
            _sync(path)

        result.append(_stored_file(root, path))

    return tuple(result)


def write_numpy(path: Path, values: np.ndarray) -> None:
    """Create one non-object NumPy array file."""
    if values.dtype.hasobject:
        _wrong_type("object arrays cannot be published")

    if path.exists() or path.is_symlink():
        _corrupt(f"refusing to replace an existing file: {path}")

    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def read_numpy(path: Path) -> np.ndarray:
    """Return one read-only memory-mapped numeric array."""
    values = np.load(path, allow_pickle=False, mmap_mode="r")

    if values.dtype.hasobject:
        _wrong_type(f"stored array has object dtype: {path}")

    return values


def read_numpy_exact(
    path: Path,
    dtype: np.dtype[np.generic],
    shape: tuple[int, ...],
) -> np.ndarray:
    """Return one memory-mapped array with exact physical metadata."""
    values = read_numpy(path)

    if values.dtype != dtype or values.shape != shape or values.flags.writeable:
        _corrupt(f"array dtype, shape, or mutability differs: {path}")

    return values


def immutable_array(value: npt.ArrayLike, name: str) -> ResultArray:
    """Return one finite, read-only, named result array.

    Raises:
        FloatingPointError: If an inexact value is nonfinite.
    """
    result = np.asarray(value)

    if (
        not name
        or not name.isascii()
        or "\x00" in name
        or result.ndim == 0
        or result.dtype.hasobject
    ):
        _invalid(f"invalid result array: {name}")

    if np.issubdtype(result.dtype, np.inexact) and not np.all(np.isfinite(result)):
        message = f"result array contains a nonfinite value: {name}"
        raise FloatingPointError(message)

    result.flags.writeable = False

    return result


def publish_arrays(
    root: Path,
    format_name: str,
    identity: Mapping[str, str],
    arrays: Mapping[str, npt.ArrayLike],
) -> DirectoryManifest:
    """Return the manifest after publishing a canonical named-array bundle."""
    names = tuple(arrays)

    if not names or names != tuple(sorted(names)):
        _invalid("array bundle names must be nonempty, unique, and sorted")

    values = tuple(immutable_array(arrays[name], name) for name in names)

    def build(destination: Path) -> dict[str, int]:
        write_json(destination / "arrays.json", list(names))

        for index, value in enumerate(values):
            write_numpy(destination / f"array-{index:03d}.npy", value)

        return {"arrays": len(names)}

    return publish_directory(root, format_name, identity, build)


def _load_arrays(
    root: Path,
    format_name: str,
    identity: Mapping[str, str],
    selected: frozenset[str] | None,
) -> Mapping[str, ResultArray]:
    verify_directory(root, format_name, identity)
    value = read_json(root / "arrays.json")

    if (
        type(value) is not list
        or not value
        or any(type(name) is not str for name in value)
    ):
        _wrong_type("array bundle names must be a nonempty string list")

    names = tuple(cast("list[str]", value))

    if names != tuple(sorted(names)) or len(set(names)) != len(names):
        _invalid("array bundle names must be unique and sorted")

    if selected is not None and (not selected or not selected.issubset(names)):
        _invalid("array subset differs from the authenticated inventory")

    return MappingProxyType({
        name: immutable_array(read_numpy(root / f"array-{index:03d}.npy"), name)
        for index, name in enumerate(names)
        if selected is None or name in selected
    })


def load_arrays(
    root: Path,
    format_name: str,
    identity: Mapping[str, str],
) -> Mapping[str, ResultArray]:
    """Return immutable arrays from one authenticated bundle."""
    return _load_arrays(root, format_name, identity, None)


def load_array_subset(
    root: Path,
    format_name: str,
    identity: Mapping[str, str],
    selected: Iterable[str],
) -> Mapping[str, ResultArray]:
    """Return selected memory-mapped arrays after authenticating their bundle."""
    return _load_arrays(root, format_name, identity, frozenset(selected))


def host_peak_bytes() -> int:
    """Return platform-normalized process peak resident memory."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if sys.platform.startswith("linux"):
        return value * 1_024

    if sys.platform == "darwin":
        return value

    return _fail(ValueError, f"unsupported ru_maxrss platform: {sys.platform}")


def load_manifest(path: Path) -> DirectoryManifest:
    """Return one parsed and authenticated directory manifest."""
    value = CanonicalJSON.read_object(path)

    if set(value) != {"blake3", "files", "format", "identity", "records"}:
        _corrupt(f"manifest fields are invalid: {path}")

    try:
        manifest = CanonicalJSON.decode_dataclass(value, DirectoryManifest)
    except (KeyError, TypeError, ValueError):
        _corrupt("manifest payload fields are invalid")

    payload = asdict(manifest)
    registered = payload.pop("blake3")

    if blake3(canonical_json(payload)).hexdigest() != registered:
        _corrupt(f"manifest checksum differs: {path}")

    if not manifest.format or "\x00" in manifest.format:
        _corrupt("manifest format is invalid")

    if manifest.identity != _identity(
        dict(manifest.identity)
    ) or manifest.records != _records(dict(manifest.records)):
        _corrupt("manifest identity or records are invalid")

    if manifest.files != tuple(
        sorted(manifest.files, key=lambda item: item.path)
    ) or len({item.path for item in manifest.files}) != len(manifest.files):
        _corrupt("manifest inventory is invalid")

    return manifest


def publication_root_exists(root: Path) -> bool:
    """Return whether a publication root is a real directory."""
    if root.is_symlink():
        _corrupt(f"publication root must be a real directory: {root}")

    if not root.exists():
        return False

    if not root.is_dir():
        _corrupt(f"publication root must be a real directory: {root}")

    return True


def verify_directory(
    directory: Path,
    format_name: str,
    identity: Mapping[str, str],
) -> DirectoryManifest:
    """Return a directory verified against its identity and bytes."""
    if not publication_root_exists(directory):
        _corrupt(f"published directory is missing: {directory}")

    manifest = load_manifest(directory / _MANIFEST)

    if manifest.format != format_name or manifest.identity != _identity(identity):
        _corrupt(f"published directory belongs to another identity: {directory}")

    if _inventory(directory) != manifest.files:
        _corrupt(f"published file inventory differs: {directory}")

    return manifest


def _seal(
    destination: Path,
    pending: Path,
    format_name: str,
    identity: tuple[tuple[str, str], ...],
    records: tuple[tuple[str, int], ...],
) -> DirectoryManifest:
    if not publication_root_exists(pending):
        _corrupt(f"publication partial must remain a real directory: {pending}")

    stored = _inventory(pending, synchronize=True)
    payload = {
        "files": tuple(asdict(item) for item in stored),
        "format": format_name,
        "identity": identity,
        "records": records,
    }
    write_json(
        pending / _MANIFEST,
        {**payload, "blake3": blake3(canonical_json(payload)).hexdigest()},
    )
    manifest = load_manifest(pending / _MANIFEST)
    directories = {path.parent for path in pending.rglob("*") if path.is_file()}

    for directory in sorted(
        directories, key=lambda item: len(item.parts), reverse=True
    ):
        _sync(directory)

    pending.rename(destination)
    _sync(destination.parent)

    return manifest


def publish_directory(
    destination: Path,
    format_name: str,
    identity: Mapping[str, str],
    build: _Builder,
) -> DirectoryManifest:
    """Return one atomically built or recovered directory publication."""
    if not destination.is_absolute() or not format_name or "\x00" in format_name:
        _invalid("publication destination or format is invalid")

    pending = destination.with_name(f"{destination.name}.partial")

    if publication_root_exists(destination):
        manifest = verify_directory(destination, format_name, identity)

        if publication_root_exists(pending):
            duplicate = verify_directory(pending, format_name, identity)

            if duplicate != manifest:
                _corrupt("finalized publication and stale partial differ")

            shutil.rmtree(pending)
            _sync(destination.parent)

        return manifest

    if publication_root_exists(pending):
        manifest = verify_directory(pending, format_name, identity)
        pending.rename(destination)
        _sync(destination.parent)

        return manifest

    pending.mkdir()

    return _seal(
        destination,
        pending,
        format_name,
        _identity(identity),
        _records(build(pending)),
    )
