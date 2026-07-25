"""logmint: append-only run logs, content-addressed blobs, and a tidy frame for figmint.

Write:

    ref = logmint.put(root, subset_indices)  # an input blob, before any run exists
    with logmint.init(root, {..., "subset": ref}) as run:
        run.metric("acc", 0.42, step=200, split="forget")
        run.blob("checkpoint", model_path, step=200)

Read:

    frame = logmint.load(root, names=["acc"])
    table = logmint.aggregate(frame, over="seed", x="step", by=["method"])
"""

from logmint import arrow_store as arrow_store
from logmint import storage as storage
from logmint._blobs import collect as gc
from logmint._blobs import get, get_path, put, reachable
from logmint._errors import (
    AlreadyFinishedError,
    BlobError,
    CollisionError,
    ConfigError,
    CorpusError,
    LogmintError,
    RecordError,
)
from logmint._identity import RESERVED, canonical, run_id
from logmint._read import (
    aggregate,
    census,
    coords_of,
    discover,
    info,
    load,
    query,
    scan,
    status_of,
)
from logmint._run import BlobRecord, Run, done, init, last_blob
from logmint._schema import write as write_schema
from logmint._verify import Problem, verify

__all__ = [
    "RESERVED",
    "AlreadyFinishedError",
    "BlobError",
    "BlobRecord",
    "CollisionError",
    "ConfigError",
    "CorpusError",
    "LogmintError",
    "Problem",
    "RecordError",
    "Run",
    "aggregate",
    "arrow_store",
    "canonical",
    "census",
    "coords_of",
    "discover",
    "done",
    "gc",
    "get",
    "get_path",
    "info",
    "init",
    "last_blob",
    "load",
    "put",
    "query",
    "reachable",
    "run_id",
    "scan",
    "status_of",
    "storage",
    "verify",
    "write_schema",
]
