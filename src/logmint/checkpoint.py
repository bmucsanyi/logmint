"""Authenticated Orbax checkpoints for exact JAX PyTree restarts."""

import ast
from collections.abc import Mapping
from importlib import import_module
from operator import attrgetter
from pathlib import Path
from typing import Never, Protocol, cast

import jax
import numpy as np
from orbax import checkpoint as ocp

from logmint import storage

__all__ = [
    "CHECKPOINT_FORMAT",
    "restore_checkpoint",
    "restore_checkpoint_state",
    "save_checkpoint",
]

CHECKPOINT_FORMAT = "logmint-jax-checkpoint-v1"
_TREE_DEFINITION = "tree-definition.json"


class _Stepped(Protocol):
    @property
    def step(self) -> jax.Array: ...


def _invalid(message: str) -> Never:
    raise ValueError(message)


def _integer_step(value: jax.Array) -> int:
    host = np.asarray(jax.device_get(value))

    if host.shape or not np.issubdtype(host.dtype, np.integer):
        _invalid("checkpoint state step must be an integer scalar")

    return int(host)


def _identity(name: str, step: int) -> dict[str, str]:
    return {"checkpoint": name, "step": str(step)}


def _tree_data(definition: jax.tree_util.PyTreeDef) -> dict[str, object]:
    node = definition.node_data()

    if node is None:
        return {"type": "", "auxiliary": "None", "children": ()}

    node_type, auxiliary = node

    return {
        "type": f"{node_type.__module__}:{node_type.__qualname__}",
        "auxiliary": repr(auxiliary),
        "children": tuple(_tree_data(child) for child in definition.children()),
    }


def _resolve_type(name: str) -> type:
    if name == "builtins:NoneType":
        return type(None)

    module_name, separator, qualified_name = name.partition(":")

    if separator != ":" or not module_name or not qualified_name:
        _invalid("checkpoint tree type name is invalid")

    return cast("type", attrgetter(qualified_name)(import_module(module_name)))


def _tree_definition(data: Mapping[str, object]) -> jax.tree_util.PyTreeDef:
    name = cast("str", data["type"])
    node = (
        None
        if not name
        else (_resolve_type(name), ast.literal_eval(cast("str", data["auxiliary"])))
    )

    return jax.tree_util.PyTreeDef.from_node_data_and_children(
        jax.tree_util.default_registry,
        node,
        tuple(
            _tree_definition(child)
            for child in cast("list[Mapping[str, object]]", data["children"])
        ),
    )


def save_checkpoint(
    directory: Path,
    state: _Stepped,
    identity: str,
) -> storage.DirectoryManifest:
    """Return the manifest after publishing an identity-bound checkpoint."""
    step = _integer_step(state.step)

    def build(root: Path) -> Mapping[str, int]:
        storage.write_json(
            root / _TREE_DEFINITION,
            _tree_data(jax.tree.structure(state)),
        )

        with ocp.StandardCheckpointer() as checkpointer:
            checkpointer.save(root / "state", tuple(jax.tree.leaves(state)))

        return {"step": step}

    return storage.publish_directory(
        directory,
        CHECKPOINT_FORMAT,
        _identity(identity, step),
        build,
    )


def _verified_definition(
    directory: Path,
    identity: str,
    expected_step: int,
) -> jax.tree_util.PyTreeDef:
    storage.verify_directory(
        directory,
        CHECKPOINT_FORMAT,
        _identity(identity, expected_step),
    )

    return _tree_definition(
        cast(
            "Mapping[str, object]",
            storage.read_json(directory / _TREE_DEFINITION),
        )
    )


def restore_checkpoint[StateT: _Stepped](
    directory: Path,
    target_state: StateT,
    identity: str,
    expected_step: int,
) -> StateT:
    """Return one identity- and layout-checked restored checkpoint."""
    if expected_step < 0 or _integer_step(target_state.step) != expected_step:
        _invalid("target checkpoint step differs from expected_step")

    definition = _verified_definition(directory, identity, expected_step)

    if definition != jax.tree.structure(target_state):
        _invalid("target checkpoint tree differs from the stored state")

    with ocp.StandardCheckpointer() as checkpointer:
        leaves = checkpointer.restore(
            directory / "state",
            tuple(jax.tree.leaves(target_state)),
            strict=True,
        )

    return cast("StateT", definition.unflatten(leaves))


def restore_checkpoint_state(
    directory: Path,
    identity: str,
    expected_step: int,
) -> _Stepped:
    """Return an authenticated checkpoint without an in-memory target."""
    if expected_step < 0:
        _invalid("checkpoint step must be nonnegative")

    definition = _verified_definition(directory, identity, expected_step)

    with ocp.StandardCheckpointer() as checkpointer:
        leaves = checkpointer.restore(directory / "state")

    return cast("_Stepped", definition.unflatten(leaves))
