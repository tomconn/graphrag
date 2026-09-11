"""Shared setup for agent tests: make the `app` package importable."""
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = REPO_ROOT / "agent"

if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

SCHEMA_PATH = REPO_ROOT / "schema" / "graph_schema.yaml"
os.environ.setdefault("SCHEMA_FILE", str(SCHEMA_PATH))


@pytest.fixture(scope="session")
def schema() -> dict:
    import yaml

    with open(SCHEMA_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)