"""API tests using FastAPI TestClient."""

import numpy as np
from fastapi.testclient import TestClient

from hydroflow.api.app import create_app


def test_health_and_config():
    app = create_app()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    r = client.get("/config")
    assert r.status_code == 200
    assert "environment" in r.json()

    r = client.get("/version")
    assert r.status_code == 200
    assert "version" in r.json()


def test_analyze_bathymetry_endpoint():
    app = create_app()
    client = TestClient(app)
    # Simple grid of points
    x = np.linspace(0, 10, 5)
    y = np.linspace(0, 10, 5)
    xx, yy = np.meshgrid(x, y)
    zz = -5 - 0.1 * (xx + yy)
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).tolist()
    r = client.post("/analyze/bathymetry", json={"points": pts, "method": "idw", "resolution": 2.0})
    assert r.status_code == 200
    body = r.json()
    assert "shape" in body and len(body["shape"]) == 2
    assert "metrics" in body


def test_analyze_sediment_endpoint():
    app = create_app()
    client = TestClient(app)
    r = client.post(
        "/analyze/sediment",
        json={
            "velocity": [1.0, 1.2, 1.5],
            "depth": [2.0, 2.5, 3.0],
            "d50": 0.5,
            "d90": 1.0,
        },
    )
    assert r.status_code == 200
    data = r.json()
    for key in ["mean_bedload", "max_bedload", "mean_suspended", "total_transport"]:
        assert key in data


def test_remediate_dredging_endpoint():
    app = create_app()
    client = TestClient(app)
    bathymetry = (np.ones((10, 10)) * -2.0).tolist()
    r = client.post(
        "/remediate/dredging",
        json={"bathymetry": bathymetry, "target_depth": 3.0, "max_volume": 100.0},
    )
    assert r.status_code == 200
    body = r.json()
    assert "areas" in body
    assert "total_volume" in body


def test_analyze_vegetation_endpoint():
    app = create_app()
    client = TestClient(app)
    # Create simple 4-band imagery (4, H, W)
    arr = np.zeros((4, 10, 10))
    arr[3, 3:6, 3:6] = 0.8  # nir high patch
    arr[0, 3:6, 3:6] = 0.2  # red lower
    imagery = arr.tolist()
    r = client.post("/analyze/vegetation", json={"imagery": imagery})
    assert r.status_code == 200
    body = r.json()
    assert "metrics" in body
