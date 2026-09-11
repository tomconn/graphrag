"""Shared setup for ingest tests: make `pipeline` importable and pin the schema."""
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INGEST_DIR = REPO_ROOT / "ingest"

if str(INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(INGEST_DIR))

SCHEMA_PATH = REPO_ROOT / "schema" / "graph_schema.yaml"
os.environ.setdefault("SCHEMA_FILE", str(SCHEMA_PATH))


@pytest.fixture()
def schema_path() -> str:
    return str(SCHEMA_PATH)