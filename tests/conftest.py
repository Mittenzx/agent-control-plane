"""Pytest fixtures for the control plane test suite.

We force an isolated temp data dir for every test so SQLite persistence state
from one test never leaks into another (which would otherwise break assertions
that count tasks/projects).
"""

import pytest

from control_plane.config.manager import ControlPlaneConfig


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Point ControlPlane's default data dir at a fresh temp directory."""
    original_init = ControlPlaneConfig.__init__

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("data_dir", str(tmp_path))
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(ControlPlaneConfig, "__init__", __init__)
    return tmp_path
