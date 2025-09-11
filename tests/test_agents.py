"""Agent tests (no external calls)."""

import os
import pytest
from fastapi.testclient import TestClient

from hydroflow.api.app import create_app


def test_agents_summarize_missing_key():
    app = create_app()
    client = TestClient(app)
    if os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY set; skipping negative test")
    r = client.post("/agents/summarize", json={"metrics": {"a": 1}})
    assert r.status_code == 400
    assert "OPENAI_API_KEY" in r.json()["detail"]


