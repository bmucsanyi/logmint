"""Config canonicalisation and run identity (spec section 2)."""

from pathlib import Path

import pytest

import logmint


def test_canonical_is_sorted_and_tight() -> None:
    """The canonical form sorts keys and carries no whitespace."""
    assert logmint.canonical({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_run_id_is_deterministic_and_order_free() -> None:
    """Key order does not change the identity."""
    assert logmint.run_id({"a": 1, "b": 2}) == logmint.run_id({"b": 2, "a": 1})


def test_types_are_part_of_identity() -> None:
    """An int and a float that compare equal are different configs."""
    assert logmint.run_id({"seed": 0}) != logmint.run_id({"seed": 0.0})


def test_missing_key_differs_from_null() -> None:
    """Absence and null are different configs."""
    assert logmint.run_id({"a": 1}) != logmint.run_id({"a": 1, "b": None})


def test_run_id_length() -> None:
    """The run id is 16 hex characters."""
    assert len(logmint.run_id({"a": 1})) == 16


def test_reserved_config_key_is_rejected() -> None:
    """A config key that would shadow a read-frame column is refused."""
    with pytest.raises(logmint.ConfigError, match="reserved"):
        logmint.run_id({"step": 1})


def test_a_nested_config_is_rejected() -> None:
    """A config is flat, because every key becomes a column of the read frame."""
    with pytest.raises(logmint.ConfigError, match="flat"):
        logmint.run_id({"opt": {"lr": 0.1}})  # ty: ignore[invalid-argument-type]


def test_a_config_key_that_is_not_an_identifier_is_rejected() -> None:
    """A column nobody can group by without quoting is not a column worth having."""
    with pytest.raises(logmint.ConfigError, match="identifier"):
        logmint.run_id({"learning rate": 0.1})


def test_non_finite_config_value_is_rejected() -> None:
    """A config holding a NaN has no JSON representation."""
    with pytest.raises(logmint.ConfigError, match="non-finite"):
        logmint.run_id({"lr": float("nan")})


def test_unserialisable_config_value_is_rejected() -> None:
    """A config holding an arbitrary object is refused at the call site."""
    with pytest.raises(logmint.ConfigError, match="cannot represent"):
        logmint.run_id({"model": object()})  # ty: ignore[invalid-argument-type]


def test_directory_name_is_the_hash_of_the_file_inside_it(root: Path) -> None:
    """The invariant that makes a hand-edited config detectable."""
    config = {"method": "npo", "seed": 3}
    with logmint.init(root, config) as run:
        run.metric("acc", 1.0)
    written = (root / "runs" / logmint.run_id(config) / "run.json").read_text(
        encoding="utf-8"
    )
    assert written == logmint.canonical(config)


def test_collision_raises_rather_than_merging(root: Path) -> None:
    """A directory holding a different config is an error, never a merge."""
    config = {"method": "npo", "seed": 3}
    logmint.init(root, config).finish()
    directory = root / "runs" / logmint.run_id(config)
    (directory / "run.json").write_text('{"method":"other"}', encoding="utf-8")
    with pytest.raises(logmint.CollisionError, match="collision"):
        logmint.init(root, config, allow_rerun=True)


def test_a_config_that_is_not_a_dict_is_rejected() -> None:
    """A run is identified by a mapping, not by a list of pairs."""
    with pytest.raises(logmint.ConfigError, match="must be a dict"):
        logmint.run_id([("seed", 0)])  # ty: ignore[invalid-argument-type]


def test_a_list_of_non_scalars_is_rejected() -> None:
    """It would arrive in the read frame as a column of structs."""
    with pytest.raises(logmint.ConfigError, match="list of non-scalars"):
        logmint.run_id({"schedule": [{"step": 1, "lr": 0.1}]})  # ty: ignore[invalid-argument-type]


def test_a_list_of_scalars_is_a_fine_config_value() -> None:
    """A list of layer indices is a scalar sequence and stays one column."""
    assert logmint.canonical({"layers": [3, 7, 11]}) == '{"layers":[3,7,11]}'
