"""Shared fixtures."""

from pathlib import Path

import pytest

import logmint
from logmint._identity import JSONValue


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Provide an empty corpus root.

    Args:
        tmp_path: The pytest temporary directory.

    Returns:
        The corpus root.

    """
    return tmp_path / "corpus"


def base_config(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Build a config for a test run.

    Args:
        **overrides: Keys to override.

    Returns:
        The config.

    """
    return {"method": "scrub", "dataset": "cifar10", "lr": 0.01, "seed": 0, **overrides}


def finished_run(root: Path, **overrides: JSONValue) -> str:
    """Write one finished run holding two accuracy points.

    Args:
        root: The corpus root.
        **overrides: Config keys to override.

    Returns:
        The run id.

    """
    config = base_config(**overrides)
    with logmint.init(root, config) as run:
        run.metric("acc", 0.9, step=100, split="forget")
        run.metric("acc", 0.4, step=200, split="forget")
    return logmint.run_id(config)
