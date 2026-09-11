"""Unit tests for ui/backend/app.py: /api/source path-traversal guard and
section extraction, plus /api/health with a stubbed agent dependency.
"""

_HTTP_ERROR = type("FakeHTTPError", (Exception,), {})


# --------------------------------------------------------------- path guard

def test_valid_nested_path_is_served(client):
    resp = client.get("/api/source", params={"path": "regulatory/sample-standard.md"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["path"] == "regulatory/sample-standard.md"
    assert body["content"].startswith("# Sample Standard")


def test_valid_deeply_nested_path_is_served(client):
    resp = client.get("/api/source",
                      params={"path": "architecture/nested/deep.md"})
    assert resp.status_code == 200
    assert "Nested document body." in resp.json()["content"]


def test_optional_leading_data_prefix_is_accepted(client):
    with_prefix = client.get("/api/source",
                             params={"path": "data/regulatory/sample-standard.md"})
    without_prefix = client.get("/api/source",
                                params={"path": "regulatory/sample-standard.md"})
    assert with_prefix.status_code == 200
    assert with_prefix.json()["content"] == without_prefix.json()["content"]


def test_parent_traversal_is_rejected(client):
    resp = client.get("/api/source", params={"path": "../secret-outside-data.txt"})
    assert resp.status_code == 400
    assert "escapes DATA_DIR" in resp.json()["detail"]


def test_deep_traversal_is_rejected(client):
    resp = client.get("/api/source",
                      params={"path": "regulatory/../../secret-outside-data.txt"})
    assert resp.status_code == 400


def test_dotdot_behind_the_data_prefix_is_rejected(client):
    """Pinned actual behavior: the "data/" prefix is stripped BEFORE the
    traversal guard, so "data/../regulatory/x.md" becomes "../regulatory/x.md"
    — an escape attempt — and is rejected with 400 even though the literal
    path never leaves the data tree when read unstripped."""
    resp = client.get("/api/source",
                      params={"path": "data/../regulatory/sample-standard.md"})
    assert resp.status_code == 400


def test_missing_file_is_404(client):
    resp = client.get("/api/source", params={"path": "regulatory/nope.md"})
    assert resp.status_code == 404


def test_empty_path_is_404_not_400(client):
    # "" resolves to DATA_DIR itself: the guard lets it through (it equals
    # DATA_DIR) and the is_file() check then 404s
    resp = client.get("/api/source", params={"path": ""})
    assert resp.status_code == 404


# ------------------------------------------------------- section extraction

def test_section_extraction_full_heading_path(client):
    resp = client.get("/api/source", params={
        "path": "regulatory/sample-standard.md",
        "section": "Scope and Application > Operational Risk"})
    assert resp.status_code == 200
    content = resp.json()["content"]
    assert content.startswith("### Operational Risk")
    assert "Operational risk content line one." in content


def test_section_stops_at_equal_or_higher_heading(client):
    resp = client.get("/api/source", params={
        "path": "regulatory/sample-standard.md",
        "section": "Scope and Application"})
    content = resp.json()["content"]
    # includes both H3 subsections ...
    assert "Operational risk content line one." in content
    assert "Change management content." in content
    # ... and stops before the next H2
    assert "Information Security" not in content
    assert content.startswith("## Scope and Application")


def test_section_heading_path_includes_the_doc_title(client):
    resp = client.get("/api/source", params={
        "path": "regulatory/sample-standard.md",
        "section": "Sample Standard > Information Security"})
    assert resp.status_code == 200
    content = resp.json()["content"]
    assert content.startswith("## Information Security")
    assert "Security content." in content


def test_section_extraction_ignores_hash_in_code_fence(client):
    resp = client.get("/api/source", params={
        "path": "regulatory/sample-standard.md",
        "section": "Scope and Application > Operational Risk"})
    content = resp.json()["content"]
    # the fenced '# looks_like_a_heading' line does not end the section ...
    assert "# looks_like_a_heading = True" in content
    assert "code_line = 1" in content
    assert "Still operational risk content after the fence." in content
    # ... and the section ends before the next H3
    assert "Change Management" not in content


def test_unknown_section_is_404(client):
    resp = client.get("/api/source", params={
        "path": "regulatory/sample-standard.md",
        "section": "Scope and Application > Nonexistent"})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "section not found"


def test_section_on_whole_file_without_that_heading_is_404(client):
    resp = client.get("/api/source", params={
        "path": "architecture/nested/deep.md", "section": "Missing > Section"})
    assert resp.status_code == 404


# ------------------------------------------------------------------- /api/health

class _FakeResponse:
    status_code = 200


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url):
        assert url.endswith("/health")
        return _FakeResponse()


def _fake_httpx(client_class):
    """A stand-in for the httpx module (the app resolves `httpx.X` from its
    own module namespace, so patching the attribute keeps the real httpx
    untouched for everything else)."""
    return type("M", (), {
        "AsyncClient": client_class,
        "HTTPError": type("HTTPError", (Exception,), {}),
    })


def test_health_reports_agent_ok(ui_app, client, monkeypatch):
    monkeypatch.setattr(ui_app, "httpx", _fake_httpx(_FakeAsyncClient))
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "agent": "ok"}


def test_health_reports_agent_unreachable_on_os_error(ui_app, client, monkeypatch):
    class ExplodingClient(_FakeAsyncClient):
        async def get(self, url):
            raise OSError("connection refused")

    monkeypatch.setattr(ui_app, "httpx", _fake_httpx(ExplodingClient))
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "agent": "unreachable"}


def test_health_swallows_httpx_errors(ui_app, client, monkeypatch):
    class FailingClient(_FakeAsyncClient):
        async def get(self, url):
            raise _HTTP_ERROR("boom")

    monkeypatch.setattr(
        ui_app, "httpx",
        type("M", (), {"AsyncClient": FailingClient, "HTTPError": _HTTP_ERROR}))
    resp = client.get("/api/health")
    assert resp.json()["agent"] == "unreachable"