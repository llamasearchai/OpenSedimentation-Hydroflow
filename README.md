Open-Sedimentation & Hydroflow
==============================

Open-Sedimentation & Hydroflow is a production-grade, modular hydrological analysis platform. It combines bathymetric processing, sediment transport modeling, vegetation analytics, real‑time monitoring, automated remediation planning, reporting, a FastAPI server, a clean CLI, Dockerized deployment, CI workflow, and optional OpenAI Agents SDK integration.

Contents
--------
- Overview
- Features
- Architecture
- Quickstart
- Command Line Interface
- API Endpoints
- Monitoring & WebSocket Streaming
- Remediation Planning
- Vegetation Analysis
- Reporting
- Agents (OpenAI integration)
- Configuration
- Docker & Compose
- Development (tests, linting, typing)
- Versioning & Releases
- Security & Privacy
- Roadmap

Overview
--------
This repository delivers a complete, tested codebase for hydrological and environmental management workflows:
- Numeric analysis with NumPy/SciPy/xarray and scikit‑learn.
- FastAPI app for HTTP/WebSocket access to analysis operations.
- CLI for fast local runs and scripting.
- Optional OpenAI Agents SDK to summarize metrics for non‑technical and technical consumers.

Features
--------
- Bathymetric surface creation (IDW, spline, triangulation, and Gaussian process) and feature detection.
- Sediment transport modeling (Meyer‑Peter & Müller, Van Rijn, Einstein), Exner solver, deposition prediction.
- Vegetation indices and classification with lightweight NDVI/EVI/SAVI logic.
- Real‑time monitoring (HTTP stream + WebSocket) with generated sensor readings for demo/testing.
- Remediation/dredging optimization with volume accounting and prioritization.
- HTML reporting via `ReportGenerator`.
- FastAPI with JSON REST + WebSocket endpoints.
- Docker image for reproducible deployment.
- CI workflow to run tests on push/PR.

Architecture
------------
```
hydroflow/
  api/            # FastAPI app & routes
  analysis/       # Bathymetry, sediment, vegetation modules
  agents/         # OpenAI agent wrapper (optional)
  cli/            # Click-based CLI
  core/           # Config & exceptions
  monitoring/     # Real-time monitor utilities
  remediation/    # Dredging optimizer
  utils/          # GIS helpers & reporting
tests/            # Unit & API/WebSocket tests
docker/           # Dockerfile and compose file
```

Quickstart
----------
1) Create environment and install
```
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e .[dev]
```

2) Verify
```
pytest -q
hydroflow info --check
```

3) Run API
```
hydroflow api serve --host 0.0.0.0 --port 8000
# Then visit http://localhost:8000/docs
```

Command Line Interface
----------------------
Show config and dependency check
```
hydroflow info --check
hydroflow config
```

Bathymetry analysis (input is whitespace-delimited XYZ file)
```
hydroflow analyze bathymetry data.xyz --method idw --resolution 2.0 -o bathy.json
```

Sediment analysis (CSV with columns velocity,depth)
```
hydroflow analyze sediment flow.csv --d50 0.5 --d90 1.0 -o sediment.json
```

Vegetation analysis (.npy file with shape: bands x H x W)
```
hydroflow analyze vegetation imagery.npy --bands red --bands green --bands blue --bands nir
```

Report generation (HTML)
```
hydroflow report --input metrics.json --type bathymetry --format html --output report.html
```

API Endpoints
-------------
- Health: `GET /health`
- Version: `GET /version`
- Config: `GET /config`
- Bathymetry: `POST /analyze/bathymetry` body: `{ points:[[x,y,z],...], method:"idw", resolution:1.0 }`
- Sediment: `POST /analyze/sediment` body: `{ velocity:[...], depth:[...], d50:0.5, d90:1.0 }`
- Vegetation: `POST /analyze/vegetation` body: `{ imagery:[[...bands,H,W...]], bands:["red","green","blue","nir"] }`
- Remediation: `POST /remediate/dredging` body: `{ bathymetry:[[...]], target_depth:3.0, max_volume:100.0 }`
- Monitoring (HTTP stream): `GET /monitor/stream?sensor_id=s-1&n=5`
- Monitoring (WebSocket): `ws://host/ws/monitor`

Example curl
```
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/analyze/sediment \
  -H 'content-type: application/json' \
  -d '{"velocity":[1.0,1.2,1.5],"depth":[2.0,2.5,3.0],"d50":0.5,"d90":1.0}'
```

Monitoring & WebSocket Streaming
--------------------------------
- `GET /monitor/stream` returns a short list of generated `SensorReading` objects.
- `ws://.../ws/monitor` pushes a brief burst of readings and closes.
These endpoints are designed for integration tests and as a template for wiring a real sensor backend.

Remediation Planning
--------------------
`hydroflow.remediation.dredging.DredgingOptimizer` clusters deficit regions, prioritizes areas, and estimates cost/volume. Use via API or CLI+Python.

Vegetation Analysis
-------------------
Lightweight indices NDVI/EVI/SAVI and a threshold-based classifier are provided for rapid prototyping. Replace with your classifier by extending `VegetationAnalyzer`.

Reporting
---------
`ReportGenerator` emits simple HTML summaries. Extend this for PDF/docx as required.

Agents (OpenAI integration)
---------------------------
The optional agent wrapper uses the OpenAI Python SDK to summarize metrics.

Environment variable:
```
export OPENAI_API_KEY=...  # required for agent usage
```

CLI usage:
```
hydroflow agents summarize --metrics metrics.json
```

API usage:
```
curl -s -X POST http://localhost:8000/agents/summarize \
  -H 'content-type: application/json' \
  -d '{"metrics":{"mean_bedload":0.012}}'
```

Configuration
-------------
`hydroflow.core.config.Config` manages paths and defaults. Override by environment:
- `HYDROFLOW_ENVIRONMENT`, `HYDROFLOW_API_HOST`, `HYDROFLOW_API_PORT`
or by YAML file referenced by `HYDROFLOW_CONFIG`.

Docker & Compose
----------------
Build and run locally:
```
docker build -f docker/Dockerfile -t hydroflow-api .
docker run --rm -p 8000:8000 hydroflow-api
```

Development
-----------
Run tests, coverage, and type checks:
```
pytest -q
```
The repository includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs tests on pushes and PRs.

Versioning & Releases
---------------------
- Current version: `0.1.0` (tagged locally as `v0.1.0`).
- Semantic Versioning (MAJOR.MINOR.PATCH).

Security & Privacy
------------------
- Do not commit secrets. Provide `OPENAI_API_KEY` via environment variables or secret stores.
- No PII is required by default. Be mindful when supplying data files.

Maintainer
----------
- Author: Nik Jois <nikjois@llamasearch.ai>

License
-------
This repository is provided under the MIT License. See `pyproject.toml` for license metadata.

Roadmap
-------
- Pluggable sensor backends (MQTT/HTTP/Modbus).
- Rich PDF report generation.
- GPU-accelerated interpolation and larger-scale datasets with Dask.


