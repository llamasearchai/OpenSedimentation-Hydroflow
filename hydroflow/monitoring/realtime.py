"""Real-time monitoring module (lightweight)."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Callable

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class SensorReading:
    sensor_id: str
    timestamp: datetime
    value: float
    unit: str
    quality: float = 1.0
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


class RealtimeMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.data_buffer: Dict[str, List[SensorReading]] = {}
        self.callbacks: Dict[str, List[Callable]] = {}

    async def stream_sensor_data(self, sensor_id: str, duration: Optional[int] = None):
        start = datetime.now()
        while True:
            reading = self.generate_reading(sensor_id)
            self.data_buffer.setdefault(sensor_id, []).append(reading)
            self.data_buffer[sensor_id] = self.data_buffer[sensor_id][-1000:]
            await asyncio.sleep(self.config.get("sampling_interval", 0.1))
            if duration and (datetime.now() - start).total_seconds() >= duration:
                break

    def generate_reading(self, sensor_id: str) -> SensorReading:
        return SensorReading(
            sensor_id=sensor_id,
            timestamp=datetime.now(),
            value=float(np.random.normal(10, 2)),
            unit="m/s",
            quality=0.95,
        )


