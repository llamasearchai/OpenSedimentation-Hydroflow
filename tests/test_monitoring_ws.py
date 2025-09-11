"""WebSocket monitoring tests."""

import json
from fastapi.testclient import TestClient

from hydroflow.api.app import create_app


def test_http_stream_endpoint():
    app = create_app()
    client = TestClient(app)
    r = client.get("/monitor/stream", params={"sensor_id": "s-1", "n": 3})
    assert r.status_code == 200
    data = r.json()
    assert "readings" in data and len(data["readings"]) == 3


def test_websocket_monitor_stream():
    app = create_app()
    client = TestClient(app)
    with client.websocket_connect("/ws/monitor") as ws:
        messages = []
        try:
            while True:
                msg = ws.receive()
                if "text" in msg:
                    messages.append(json.loads(msg["text"]))
                elif "bytes" in msg:
                    messages.append(json.loads(msg["bytes"].decode("utf-8")))
                else:
                    break
        except Exception:
            pass
    assert len(messages) >= 1
    assert "sensor_id" in messages[0]

