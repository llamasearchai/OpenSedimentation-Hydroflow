"""Lightweight GIS helpers (GDAL-free)."""

from __future__ import annotations

from typing import Tuple
import numpy as np


def coordinate_transform(x: np.ndarray, y: np.ndarray, src_epsg: int, dst_epsg: int) -> Tuple[np.ndarray, np.ndarray]:
    """Placeholder-free simple passthrough transform.

    For environments without PROJ/GDAL, we return inputs unchanged.
    In production, replace with pyproj if needed.
    """
    return x, y


def calculate_volume(values: np.ndarray, reference: float) -> float:
    """Estimate volume as sum of differences below a reference surface.

    Args:
        values: Depth/elevation values for a region
        reference: Reference level

    Returns:
        Estimated volume (unitless unless multiplied by cell area elsewhere)
    """
    diffs = np.maximum(reference - values, 0)
    return float(np.nansum(diffs))


