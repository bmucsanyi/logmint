"""Config canonicalisation and run identity (spec section 2).

The directory name of a run is the hash of the file inside it, which makes
``sha256(canonical(run.json))[:RUN_ID_CHARS] == basename(dir)`` a checkable invariant.

A config is flat and its keys are identifiers, because every key becomes a
column of the read frame. Nesting would arrive in the frame as a struct, and a
key that needs quoting would arrive as a column nobody can group by without
ceremony.
"""

import hashlib
import json
import re
from collections.abc import Mapping

from logmint._errors import ConfigError

type JSONScalar = str | int | float | bool | None
type JSONValue = JSONScalar | list[JSONScalar]
type Config = Mapping[str, JSONValue]

RUN_ID_CHARS = 16
"""64 bits. With n runs the collision probability is about n^2 / 2^65, and
the real guard is that ``init`` compares an existing ``run.json``
byte-for-byte and raises rather than merging."""

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
"""Config keys and coordinate names are identifiers, so they can be columns
without quoting."""

RESERVED: frozenset[str] = frozenset({
    "attempt",
    "kind",
    "name",
    "nonfinite",
    "ref",
    "run",
    "split",
    "status",
    "step",
    "time",
    "value",
})
"""Names a config key may not take: they are columns of the read frame."""


def check_name(name: str, kind: str) -> None:
    """Check that a name can serve as a column of the read frame.

    Args:
        name: The candidate name.
        kind: What the name is, for the error message.

    Raises:
        ConfigError: The name is reserved or is not an identifier.

    """
    if not isinstance(name, str) or not IDENT.fullmatch(name):
        msg = (
            f"{kind} {name!r} is not an identifier, so it cannot be a column of "
            "the read frame"
        )
        raise ConfigError(msg)
    if name in RESERVED:
        msg = f"{kind} {name!r} is a reserved name of a read-frame column"
        raise ConfigError(msg)


def validate(config: Config) -> None:
    """Check that a config can serve as a run identity.

    Args:
        config: The config to check.

    Raises:
        ConfigError: The config is not a flat dict of identifiers mapping to JSON
            scalars or lists of them, or it holds a non-finite number.

    """
    if not isinstance(config, dict):
        msg = f"config must be a dict, got {type(config).__name__}"
        raise ConfigError(msg)

    for key, value in config.items():
        check_name(key, "config key")
        _check_value(key, value)

    try:
        canonical(config)
    except (
        ValueError
    ) as exc:  # json raises ValueError on a non-finite number under allow_nan=False
        msg = (
            f"config holds a non-finite number, which has no JSON representation: {exc}"
        )
        raise ConfigError(msg) from exc
    except TypeError as exc:
        msg = f"config holds a value JSON cannot represent: {exc}"
        raise ConfigError(msg) from exc


def _check_value(key: str, value: object) -> None:
    """Check that a config value can be a column of the read frame.

    Args:
        key: The config key, for the error message.
        value: Its value.

    Raises:
        ConfigError: The value is a mapping, or a sequence holding one, either of
            which arrives in the frame as a struct rather than a column.

    """
    if isinstance(value, dict):
        msg = (
            f"config key {key!r} is nested. A config is flat, because every key "
            "becomes a column of the read frame, and a nested value becomes a "
            "struct; flatten it before init"
        )
        raise ConfigError(msg)
    if isinstance(value, list) and not all(
        isinstance(item, str | int | float | bool | None) for item in value
    ):
        msg = (
            f"config key {key!r} holds a list of non-scalars, which becomes a "
            "column of structs; a config value is a scalar or a list of scalars"
        )
        raise ConfigError(msg)


def canonical(config: Config) -> str:
    """Serialise a config to its canonical form.

    Sorted keys, no whitespace, no non-finite numbers. Types are part of
    identity: ``0`` and ``0.0`` canonicalise differently, as do a missing key
    and ``null``.

    Args:
        config: The config to serialise.

    Returns:
        The canonical JSON text.

    """
    return json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def run_id(config: Config) -> str:
    """Compute the run id of a config.

    Args:
        config: The config to identify.

    Returns:
        The first ``RUN_ID_CHARS`` hex characters of the sha256 of the canonical form.

    """
    validate(config)
    digest = hashlib.sha256(canonical(config).encode("utf-8")).hexdigest()
    return digest[:RUN_ID_CHARS]
