"""Dredging optimization (lightweight)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DredgingArea:
    id: str
    location: Tuple[float, float]
    volume: float
    priority: int
    depth_deficit: float

    @property
    def cost_estimate(self) -> float:
        base_cost = 15.0
        return float(base_cost * self.volume)


class DredgingOptimizer:
    def __init__(self, config: Dict):
        self.config = config

    def optimize_dredging_plan(
        self, bathymetry: np.ndarray, target_depth: float, max_volume: Optional[float] = None
    ) -> Dict:
        areas = self._identify_dredging_areas(bathymetry, target_depth)
        areas.sort(key=lambda a: (-a.priority, -a.volume))
        total_volume = sum(a.volume for a in areas)
        if max_volume and total_volume > max_volume:
            selected: List[DredgingArea] = []
            vol = 0.0
            for a in areas:
                if vol + a.volume <= max_volume:
                    selected.append(a)
                    vol += a.volume
            areas = selected
            total_volume = vol
        return {
            "areas": [self._area_to_dict(a) for a in areas],
            "total_volume": float(total_volume),
            "estimated_cost": float(sum(a.cost_estimate for a in areas)),
            "duration_days": float(total_volume / 5000.0 if total_volume > 0 else 0),
            "priority_areas": [a.id for a in areas[:5]],
        }

    def _identify_dredging_areas(
        self, bathymetry: np.ndarray, target_depth: float
    ) -> List[DredgingArea]:
        deficit = np.maximum(target_depth - bathymetry, 0)
        from scipy.ndimage import label

        mask = deficit > 0.1
        labeled, num = label(mask)
        areas: List[DredgingArea] = []
        for i in range(1, num + 1):
            comp = labeled == i
            if np.sum(comp) < 10:
                continue
            idx = np.where(comp)
            cy, cx = float(np.mean(idx[0])), float(np.mean(idx[1]))
            vol = float(np.sum(deficit[comp]))
            max_def = float(np.max(deficit[comp]))
            priority = (
                5
                if max_def > 2
                else 4 if max_def > 1.5 else 3 if max_def > 1.0 else 2 if max_def > 0.5 else 1
            )
            areas.append(
                DredgingArea(
                    id=f"area_{i}",
                    location=(cx, cy),
                    volume=vol,
                    priority=priority,
                    depth_deficit=max_def,
                )
            )
        return areas

    def _area_to_dict(self, area: DredgingArea) -> Dict:
        return {
            "id": area.id,
            "location": area.location,
            "volume": float(area.volume),
            "priority": int(area.priority),
            "depth_deficit": float(area.depth_deficit),
            "cost_estimate": float(area.cost_estimate),
        }
