"""FastAPI app for HydroFlow."""

from __future__ import annotations

from typing import Dict, Optional

import asyncio
from fastapi import FastAPI, HTTPException
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from hydroflow.core.config import Config
from hydroflow.analysis.bathymetric import BathymetricAnalyzer
from hydroflow.analysis.sediment_transport import SedimentTransportModel, SedimentProperties
from hydroflow.agents.openai_agent import HydroFlowAgent
from hydroflow.remediation.dredging import DredgingOptimizer
from hydroflow.analysis.vegetation import VegetationAnalyzer
from hydroflow.monitoring.realtime import RealtimeMonitor
from hydroflow import __version__


class BathymetryRequest(BaseModel):
    points: list[list[float]]  # [[x,y,z], ...]
    method: str = "idw"
    resolution: float = 1.0


class SedimentRequest(BaseModel):
    velocity: list[float]
    depth: list[float]
    d50: float = 0.5
    d90: float = 1.0


class SummarizeRequest(BaseModel):
    metrics: Dict


def create_app(config: Optional[Config] = None) -> FastAPI:
    cfg = config or Config()
    app = FastAPI(title="HydroFlow API", version="0.1.0")

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/config")
    def get_config() -> Dict:
        return cfg.dict()

    @app.get("/version")
    def version() -> Dict[str, str]:
        return {"version": __version__}

    @app.post("/analyze/bathymetry")
    def analyze_bathymetry(req: BathymetryRequest) -> Dict:
        import numpy as np

        analyzer = BathymetricAnalyzer(cfg.dict())
        pts = np.array(req.points, dtype=float)
        surface = analyzer.create_bathymetric_surface(pts, method=req.method, resolution=req.resolution)
        metrics = analyzer.detect_channel_features(surface)["metrics"]
        return {"shape": list(surface.shape), "metrics": metrics}

    @app.post("/analyze/sediment")
    def analyze_sediment(req: SedimentRequest) -> Dict:
        import numpy as np

        model = SedimentTransportModel(cfg.dict())
        model.set_sediment_properties(SedimentProperties(d50=req.d50, d90=req.d90))
        v = np.array(req.velocity, dtype=float)
        h = np.array(req.depth, dtype=float)
        tau = model.calculate_bed_shear_stress(v, h)
        bedload = model.calculate_bedload_transport(tau)
        suspended = model.calculate_suspended_load(v, h)
        return {
            "mean_bedload": float(np.mean(bedload)),
            "max_bedload": float(np.max(bedload)),
            "mean_suspended": float(np.mean(suspended)),
            "total_transport": float(np.sum(bedload + suspended)),
        }

    @app.post("/agents/summarize")
    def summarize(req: SummarizeRequest) -> Dict:
        import os

        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY not configured")
        agent = HydroFlowAgent()
        summary = agent.summarize_metrics(req.metrics)
        return {"summary": summary}

    @app.post("/remediate/dredging")
    def remediate_dredging(body: Dict) -> Dict:
        import numpy as np

        grid = np.array(body.get("bathymetry"), dtype=float)
        if grid.ndim != 2:
            raise HTTPException(status_code=400, detail="bathymetry must be a 2D array")
        target_depth = float(body.get("target_depth", 3.0))
        max_volume = body.get("max_volume")
        max_volume = float(max_volume) if max_volume is not None else None
        optimizer = DredgingOptimizer(cfg.dict())
        plan = optimizer.optimize_dredging_plan(grid, target_depth, max_volume)
        return plan

    @app.post("/analyze/vegetation")
    def analyze_vegetation(body: Dict) -> Dict:
        import numpy as np

        # Expect imagery as nested list or .npy path
        imagery = body.get("imagery")
        bands = body.get("bands", ["red", "green", "blue", "nir"])
        if isinstance(imagery, str):
            arr = np.load(imagery)
        else:
            arr = np.array(imagery, dtype=float)
        analyzer = VegetationAnalyzer(cfg.dict())
        indices = analyzer._calculate_vegetation_indices(arr, bands)
        classification = analyzer._classify_vegetation(arr, indices)
        metrics = analyzer._calculate_vegetation_metrics(classification)
        return {"metrics": metrics}

    @app.get("/monitor/stream")
    async def http_stream(sensor_id: str = "sensor-1", n: int = 5) -> Dict:
        monitor = RealtimeMonitor(cfg.dict())
        out = []
        for _ in range(n):
            out.append(monitor.generate_reading(sensor_id).to_dict())
        return {"readings": out}

    @app.websocket("/ws/monitor")
    async def ws_monitor(websocket: WebSocket):
        await websocket.accept()
        monitor = RealtimeMonitor(cfg.dict())
        try:
            for _ in range(5):
                reading = monitor.generate_reading("sensor-1").to_dict()
                await websocket.send_json(reading)
                await asyncio.sleep(0.05)
            await websocket.close()
        except WebSocketDisconnect:
            pass

    return app


app = create_app()



