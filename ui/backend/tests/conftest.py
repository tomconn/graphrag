"""Shared setup for ui/backend tests.

ui/backend/app.py reads DATA_DIR at import time, so the fixture data dir is
created and the env var set BEFORE the module is imported (once per session,
under a unique module name so it cannot clash with the agent's `app` package).
"""
import importlib.util
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_APP = REPO_ROOT / "ui" / "backend" / "app.py"

FIXTURE_MD = """# Sample Standard

Preamble text about the standard.

## Scope and Application

This standard applies to all APRA-regulated entities.

### Operational Risk

Operational risk content line one.

```python
# looks_like_a_heading = True
code_line = 1
```

Still operational risk content after the fence.

### Change Management

Change management content.

## Information Security

Security content.

## Overview

Overview content.
"""


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("ui_data")
    (root / "regulatory").mkdir()
    (root / "regulatory" / "sample-standard.md").write_text(FIXTURE_MD)
    (root / "architecture" / "nested").mkdir(parents=True)
    (root / "architecture" / "nested" / "deep.md").write_text(
        "# Deep Doc\n\nNested document body.\n")
    # a file OUTSIDE the data dir, used as the traversal target
    secret = root.parent / "secret-outside-data.txt"
    secret.write_text("top secret\n")
    return root


@pytest.fixture(scope="session")
def ui_app(data_dir):
    old = os.environ.get("DATA_DIR")
    os.environ["DATA_DIR"] = str(data_dir)
    try:
        spec = importlib.util.spec_from_file_location("ui_backend_app", BACKEND_APP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if old is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = old
    return module


@pytest.fixture()
def client(ui_app):
    from fastapi.testclient import TestClient

    return TestClient(ui_app.app)