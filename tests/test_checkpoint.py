"""Independent checks for authenticated JAX checkpoints."""

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from logmint import storage
from logmint.checkpoint import (
    restore_checkpoint,
    restore_checkpoint_state,
    save_checkpoint,
)


class _State(NamedTuple):
    values: dict[str, jax.Array]
    step: jax.Array


def _state(step: int, offset: int) -> _State:
    return _State(
        {
            "matrix": jnp.arange(offset, offset + 6, dtype=jnp.float32).reshape(2, 3),
            "vector": jnp.arange(offset + 6, offset + 9, dtype=jnp.int32),
        },
        jnp.asarray(step, dtype=jnp.uint32),
    )


def _assert_state(actual: object, expected: _State) -> None:
    assert isinstance(actual, _State)
    assert_array_equal(actual.step, expected.step)

    for name in expected.values:
        assert_array_equal(actual.values[name], expected.values[name])


def test_checkpoint_restores_targeted_and_target_free_trees(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    source = _state(7, 3)
    manifest = save_checkpoint(root, source, "training-run")
    target = _state(7, 100)
    targeted = restore_checkpoint(root, target, "training-run", 7)
    target_free = restore_checkpoint_state(root, "training-run", 7)

    _assert_state(targeted, source)
    _assert_state(target_free, source)
    assert dict(manifest.records) == {"step": 7}


def test_checkpoint_rejects_identity_step_layout_and_bytes(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    source = _state(5, 2)
    save_checkpoint(root, source, "training-run")

    with pytest.raises(storage.StorageError):
        restore_checkpoint(root, _state(5, 0), "other-run", 5)

    with pytest.raises(ValueError, match="step differs"):
        restore_checkpoint(root, _state(4, 0), "training-run", 5)

    wrong_layout = _State(
        {"other": jnp.zeros(6)},
        jnp.asarray(5, dtype=jnp.uint32),
    )

    with pytest.raises(ValueError, match="tree differs"):
        restore_checkpoint(root, wrong_layout, "training-run", 5)

    definition = root / "tree-definition.json"
    definition.write_bytes(definition.read_bytes() + b" ")

    with pytest.raises(storage.StorageError):
        restore_checkpoint_state(root, "training-run", 5)
