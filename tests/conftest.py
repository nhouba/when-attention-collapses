"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from _helpers import make_fake_fuss


@pytest.fixture(scope="session")
def fake_fuss(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A FUSS-layout ``ssdata`` directory with a few tiny random mixtures."""
    return make_fake_fuss(tmp_path_factory.mktemp("fuss") / "ssdata")
