"""Content-addressed blob store (spec section 4).

Every non-scalar lives here, named by the sha256 of its bytes, and is referenced
from exactly one place: a value in ``run.json`` (an input blob) or a ``blob``
record in an event stream (an output blob). The path carries no extension, so a
reference resolves without knowing the format.
"""

import contextlib
import hashlib
import io
import os
import re
import time
import uuid
from pathlib import Path

import numpy as np

from logmint._errors import BlobError

GRACE_S = 3600.0
"""How long a blob must have been unreachable and untouched before it can be
collected."""

PREFIX = "blob:"
REF_RE = re.compile(r"blob:[0-9a-f]{64}")
_HEX_CHARS = 64
_FANOUT = 2
_CHUNK = 1 << 20
_NPY_MAGIC = b"\x93NUMPY"

Blobbable = np.ndarray | bytes | Path


def blobs_dir(root: Path) -> Path:
    """Return the blob store directory of a corpus.

    Args:
        root: The corpus root.

    Returns:
        The ``blobs/`` directory, which may be a symlink to scratch.

    """
    return root / "blobs"


def digest_of(ref: str) -> str:
    """Extract the hex digest from a reference.

    Args:
        ref: A reference of the form ``blob:<64 hex chars>``.

    Returns:
        The hex digest.

    Raises:
        BlobError: The reference is malformed.

    """
    if not ref.startswith(PREFIX) or len(ref) != len(PREFIX) + _HEX_CHARS:
        msg = f"malformed blob reference: {ref!r}"
        raise BlobError(msg)
    digest = ref.removeprefix(PREFIX)
    if not REF_RE.fullmatch(ref):
        msg = f"blob reference is not lowercase hex: {ref!r}"
        raise BlobError(msg)
    return digest


def path_for(root: Path, ref: str) -> Path:
    """Map a reference to its path, which is a pure function of the reference.

    Args:
        root: The corpus root.
        ref: The blob reference.

    Returns:
        The path the blob occupies, whether or not it exists.

    """
    digest = digest_of(ref)
    return blobs_dir(root) / digest[:_FANOUT] / digest[_FANOUT:]


def _encode(obj: Blobbable) -> tuple[bytes | None, Path | None, str]:
    """Reduce a blobbable object to bytes or a source path, plus its format tag.

    Args:
        obj: An array, raw bytes, or a path to a file already on disk.

    Returns:
        A triple of in-memory bytes (or None), a source path (or None), and the
        format tag.

    Raises:
        BlobError: The object is of an unsupported type.

    """
    if isinstance(obj, np.ndarray):
        buffer = io.BytesIO()
        np.save(buffer, obj, allow_pickle=False)
        return buffer.getvalue(), None, "npy"
    if isinstance(obj, bytes):
        return obj, None, "bin"
    if isinstance(obj, Path):
        return None, obj, obj.suffix.removeprefix(".") or "bin"
    msg = f"cannot store an object of type {type(obj).__name__} as a blob"
    raise BlobError(msg)


def _commit(root: Path, tmp: Path, digest: str) -> str:
    """Move a fully written temporary file into the store, deduplicating by content.

    Args:
        root: The corpus root.
        tmp: The temporary file holding the bytes.
        digest: The sha256 of those bytes.

    Returns:
        The blob reference.

    """
    ref = f"{PREFIX}{digest}"
    final = path_for(root, ref)
    if final.exists():
        tmp.unlink()
        return ref
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, final)  # noqa: PTH105 -- atomic rename; Path.rename is not on POSIX overwrite
    return ref


def put(root: Path, obj: Blobbable) -> str:
    """Write an object into the blob store and return its reference.

    Takes no run: an input blob has to exist before ``init`` can hash a config
    that names it. Writing is atomic (temporary file, fsync, rename), so a torn
    blob is never visible under its final name. Bytes already in the store are not
    rewritten, so a blob shared by fifty runs is stored once.

    Args:
        root: The corpus root.
        obj: An array, raw bytes, or a path to a file already on disk.

    Returns:
        The reference ``blob:<sha256>``.

    Raises:
        BlobError: The object is an unsupported type, or a given path is not a file.

    """
    data, source, _ = _encode(obj)
    if source is not None and not source.is_file():
        msg = f"cannot store {source} as a blob: it is not a file"
        raise BlobError(msg)
    tmp_dir = blobs_dir(root) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / uuid.uuid4().hex
    hasher = hashlib.sha256()

    with tmp.open("wb") as handle:
        if data is not None:
            hasher.update(data)
            handle.write(data)
        elif source is not None:
            with source.open("rb") as src:
                while chunk := src.read(_CHUNK):
                    hasher.update(chunk)
                    handle.write(chunk)
        handle.flush()
        os.fsync(handle.fileno())

    return _commit(root, tmp, hasher.hexdigest())


def format_of(obj: Blobbable) -> str:
    """Return the format tag logmint records for an object.

    Args:
        obj: An array, raw bytes, or a path to a file already on disk.

    Returns:
        The format tag, such as ``npy`` or ``pt``.

    """
    return _encode(obj)[2]


def get_path(root: Path, ref: str) -> Path:
    """Resolve a reference to an existing path.

    Args:
        root: The corpus root.
        ref: The blob reference.

    Returns:
        The path holding the bytes.

    Raises:
        BlobError: The reference does not resolve to a file.

    """
    path = path_for(root, ref)
    if not path.is_file():
        msg = f"blob reference does not resolve: {ref}"
        raise BlobError(msg)
    return path


def get(root: Path, ref: str) -> np.ndarray | bytes:
    """Load a blob, decoding NumPy payloads by their magic bytes.

    Args:
        root: The corpus root.
        ref: The blob reference.

    Returns:
        An array if the bytes are a ``.npy`` payload, otherwise the raw bytes.

    """
    path = get_path(root, ref)
    with path.open("rb") as handle:
        if handle.read(len(_NPY_MAGIC)) == _NPY_MAGIC:
            handle.seek(0)
            return np.load(handle, allow_pickle=False)
    return path.read_bytes()


def refs_in(text: str) -> set[str]:
    """Find every blob reference in a piece of text.

    Args:
        text: Any text, typically the contents of a ``run.json`` or an event stream.

    Returns:
        The set of references it mentions.

    """
    return set(REF_RE.findall(text))


def reachable(root: Path) -> set[str]:
    """Compute the reachable set: every reference named by any run.

    The rule needs no parser, which is why it is stated as a regular expression.
    The shell equivalent is ``grep -rhoE 'blob:[0-9a-f]{64}' runs/ | sort -u``.

    Args:
        root: The corpus root.

    Returns:
        The set of referenced blobs.

    """
    found: set[str] = set()
    runs = root / "runs"
    if not runs.is_dir():
        return found
    for path in runs.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl"}:
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    found |= refs_in(line)
    return found


def stored(root: Path) -> set[str]:
    """List every blob in the store.

    Args:
        root: The corpus root.

    Returns:
        The set of stored references, ignoring the temporary directory.

    """
    blobs = blobs_dir(root)
    if not blobs.is_dir():
        return set()
    found: set[str] = set()
    for path in blobs.iterdir():
        if not path.is_dir() or len(path.name) != _FANOUT:
            continue
        found |= {
            f"{PREFIX}{path.name}{child.name}"
            for child in path.iterdir()
            if child.is_file() and len(path.name + child.name) == _HEX_CHARS
        }
    return found


def staged(root: Path) -> list[Path]:
    """List the staging files a killed writer left behind.

    A ``put`` that is killed between opening its staging file and renaming it
    leaves that file in place. It is unreferenced by construction, since a
    reference is only ever the hash of a blob that arrived, so it is safe to
    remove and nothing else will.

    Args:
        root: The corpus root.

    Returns:
        The abandoned staging files.

    """
    tmp = blobs_dir(root) / "tmp"
    return sorted(p for p in tmp.iterdir() if p.is_file()) if tmp.is_dir() else []


def collect(
    root: Path, *, dry_run: bool = False, grace_s: float = GRACE_S
) -> list[str]:
    """Delete unreachable blobs and abandoned staging files.

    A blob is written before the record that references it, and an input blob is
    written before the config that names it - which the caller may not assemble
    for another minute. In both windows the blob is on disk and unreachable, and a
    collector running against a live corpus would delete data that is about to be
    referenced. So a blob is only collected once it has been unreachable *and*
    untouched for ``grace_s``, which is the same reason git keeps unreachable
    objects around for two weeks before pruning them.

    Args:
        root: The corpus root.
        dry_run: Report what would be deleted without deleting it.
        grace_s: Leave anything modified more recently than this alone.

    Returns:
        The sorted references that are (or would be) removed.

    """
    cutoff = time.time() - grace_s
    unreachable = [
        ref
        for ref in sorted(stored(root) - reachable(root))
        if path_for(root, ref).stat().st_mtime < cutoff
    ]
    if dry_run:
        return unreachable
    for ref in unreachable:
        path = path_for(root, ref)
        path.unlink(missing_ok=True)
        with contextlib.suppress(
            OSError
        ):  # the fan-out directory may still hold other blobs
            path.parent.rmdir()
    for path in staged(root):
        if path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)
    return unreachable
