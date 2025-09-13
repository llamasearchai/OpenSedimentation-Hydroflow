"""Bathymetric analysis module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from scipy import interpolate
from scipy.spatial import Delaunay, cKDTree

from hydroflow.core.exceptions import DataProcessingError
from hydroflow.utils.gis import calculate_volume

logger = logging.getLogger(__name__)


class BathymetricAnalyzer:
    """Advanced bathymetric data analysis."""

    def __init__(self, config: Dict):
        self.config = config
        self.data_cache: Dict[str, np.ndarray] = {}
        self.interpolators: Dict[str, object] = {}

    def process_multibeam_sonar(self, raw_data: np.ndarray, metadata: Dict) -> xr.Dataset:
        """Process multibeam sonar data into an xarray Dataset."""
        try:
            x = raw_data[:, 0]
            y = raw_data[:, 1]
            z = raw_data[:, 2]
            intensity = raw_data[:, 3] if raw_data.shape[1] > 3 else None

            # Simple corrections
            zv = self._apply_sound_velocity_correction(z, metadata.get("sound_velocity", 1500))
            zt = self._apply_tidal_correction(zv, metadata.get("tidal_data"))
            zf = self._filter_outliers(zt)

            data_vars = {
                "depth": (["point"], zf),
                "quality": (["point"], self._calculate_quality_metrics(raw_data)),
            }
            if intensity is not None:
                data_vars["intensity"] = (["point"], intensity)

            ds = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "x": ("point", x),
                    "y": ("point", y),
                    # Avoid pandas dependency at import time
                    "time": np.datetime64("now"),
                },
                attrs=metadata,
            )
            return ds
        except Exception as e:
            logger.error(f"Error processing multibeam sonar: {e}")
            raise DataProcessingError(f"Failed to process sonar data: {e}")

    def process_lidar_bathymetry(
        self, point_cloud: np.ndarray, water_surface: Optional[np.ndarray] = None
    ) -> xr.Dataset:
        if water_surface is None:
            water_surface = self._detect_water_surface(point_cloud)

        corrected = self._apply_refraction_correction(point_cloud, water_surface)
        gridded = self._grid_point_cloud(corrected, self.config.get("grid_resolution", 1.0))

        slope = self._calculate_slope(gridded)
        aspect = self._calculate_aspect(gridded)
        curvature = self._calculate_curvature(gridded)

        return xr.Dataset(
            {
                "elevation": (("y", "x"), gridded),
                "slope": (("y", "x"), slope),
                "aspect": (("y", "x"), aspect),
                "curvature": (("y", "x"), curvature),
                "roughness": (("y", "x"), self._calculate_roughness(gridded)),
            }
        )

    def create_bathymetric_surface(
        self,
        points: np.ndarray,
        method: str = "kriging",
        resolution: float = 1.0,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> np.ndarray:
        if bounds is None:
            bounds = (
                float(points[:, 0].min()),
                float(points[:, 1].min()),
                float(points[:, 0].max()),
                float(points[:, 1].max()),
            )

        x_grid = np.arange(bounds[0], bounds[2] + resolution, resolution)
        y_grid = np.arange(bounds[1], bounds[3] + resolution, resolution)
        xx, yy = np.meshgrid(x_grid, y_grid)

        if method == "kriging":
            # Lightweight GP using sklearn; import locally to avoid global dependency if unused
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel

            kernel = RBF(length_scale=10.0) + WhiteKernel(noise_level=0.1)
            gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
            gpr.fit(points[:, :2], points[:, 2])
            grid_points = np.column_stack([xx.ravel(), yy.ravel()])
            predictions = gpr.predict(grid_points)
            surface = predictions.reshape(xx.shape)
        elif method == "idw":
            surface = self._idw_interpolation(points, xx, yy)
        elif method == "spline":
            surface = self._spline_interpolation(points, xx, yy)
        elif method == "triangulation":
            surface = self._triangulation_interpolation(points, xx, yy)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        return surface

    def detect_channel_features(self, bathymetry: np.ndarray, min_depth: float = 1.0) -> Dict:
        features: Dict[str, object] = {}
        features["thalweg"] = self._find_thalweg(bathymetry)
        features["banks"] = self._detect_banks(bathymetry, min_depth)
        features["pools"] = self._detect_pools(bathymetry)
        features["riffles"] = self._detect_riffles(bathymetry)
        features["scour_holes"] = self._detect_scour_holes(bathymetry)
        features["metrics"] = self._calculate_channel_metrics(bathymetry, features)
        return features

    def calculate_volume_change(
        self, surface1: np.ndarray, surface2: np.ndarray, cell_size: float = 1.0
    ) -> Dict:
        diff = surface2 - surface1
        deposition = np.where(diff > 0, diff, 0)
        erosion = np.where(diff < 0, -diff, 0)
        cell_area = cell_size**2
        return {
            "total_deposition": float(np.nansum(deposition) * cell_area),
            "total_erosion": float(np.nansum(erosion) * cell_area),
            "net_change": float(np.nansum(diff) * cell_area),
            "mean_change": float(np.nanmean(diff)),
            "max_deposition": float(np.nanmax(deposition)),
            "max_erosion": float(np.nanmax(erosion)),
            "affected_area": int(np.sum(np.abs(diff) > 0.1)) * cell_area,
        }

    # Helpers
    def _apply_sound_velocity_correction(
        self, depths: np.ndarray, sound_velocity: float
    ) -> np.ndarray:
        reference_velocity = 1500.0
        return depths * (sound_velocity / reference_velocity)

    def _apply_tidal_correction(self, depths: np.ndarray, tidal_data: Optional[Dict]) -> np.ndarray:
        if tidal_data is None:
            return depths
        tide_level = tidal_data.get("level", 0)
        return depths - tide_level

    def _filter_outliers(
        self, data: np.ndarray, method: str = "iqr", threshold: float = 1.5
    ) -> np.ndarray:
        if method == "iqr":
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            return np.where((data >= lower) & (data <= upper), data, np.nan)
        if method == "zscore":
            z = np.abs((data - np.mean(data)) / (np.std(data) + 1e-12))
            return np.where(z < threshold, data, np.nan)
        return data

    def _calculate_quality_metrics(self, data: np.ndarray) -> np.ndarray:
        quality = np.ones(len(data))
        tree = cKDTree(data[:, :2])
        distances, _ = tree.query(data[:, :2], k=min(10, len(data)))
        mean_distances = np.mean(distances, axis=1)
        quality *= np.exp(-mean_distances / (np.median(mean_distances) + 1e-12))
        return quality

    def _idw_interpolation(
        self, points: np.ndarray, xx: np.ndarray, yy: np.ndarray, power: float = 2.0
    ) -> np.ndarray:
        tree = cKDTree(points[:, :2])
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        distances, indices = tree.query(grid_points, k=min(12, len(points)))
        distances = np.maximum(distances, 1e-10)
        weights = 1.0 / distances**power
        weights /= weights.sum(axis=1, keepdims=True)
        values = points[indices, 2]
        interpolated = np.sum(weights * values, axis=1)
        return interpolated.reshape(xx.shape)

    def _spline_interpolation(
        self, points: np.ndarray, xx: np.ndarray, yy: np.ndarray
    ) -> np.ndarray:
        from scipy.interpolate import RBFInterpolator

        interpolator = RBFInterpolator(
            points[:, :2], points[:, 2], kernel="thin_plate_spline", smoothing=0.1
        )
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        interpolated = interpolator(grid_points)
        return interpolated.reshape(xx.shape)

    def _triangulation_interpolation(
        self, points: np.ndarray, xx: np.ndarray, yy: np.ndarray
    ) -> np.ndarray:
        tri = Delaunay(points[:, :2])
        interpolator = interpolate.LinearNDInterpolator(tri, points[:, 2])
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        interpolated = interpolator(grid_points)
        return interpolated.reshape(xx.shape)

    def _calculate_slope(self, grid: np.ndarray) -> np.ndarray:
        dy, dx = np.gradient(grid)
        return np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    def _calculate_aspect(self, grid: np.ndarray) -> np.ndarray:
        dy, dx = np.gradient(grid)
        return np.degrees(np.arctan2(-dy, dx))

    def _calculate_curvature(self, grid: np.ndarray) -> np.ndarray:
        dy, dx = np.gradient(grid)
        dyy, _ = np.gradient(dy)
        _, dxx = np.gradient(dx)
        return dxx + dyy

    def _calculate_roughness(self, grid: np.ndarray, window: int = 3) -> np.ndarray:
        from scipy.ndimage import generic_filter

        return generic_filter(grid, np.std, size=window)

    def _find_thalweg(self, bathymetry: np.ndarray) -> np.ndarray:
        """Return thalweg as sequence of (row, col) coordinates along columns.

        This constructs a simple path by selecting the minimum-depth cell in each
        column and pairing it with its column index, yielding an (N, 2) array.
        """
        min_rows = np.argmin(bathymetry, axis=0)
        cols = np.arange(bathymetry.shape[1])
        coords = np.column_stack([min_rows, cols])
        return coords

    def _detect_banks(
        self, bathymetry: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        slopes = self._calculate_slope(bathymetry)
        return np.where(slopes > threshold)

    def _detect_pools(self, bathymetry: np.ndarray) -> List[Dict]:
        from scipy.ndimage import label, minimum_filter

        local_min = minimum_filter(bathymetry, size=5)
        pools_mask = bathymetry == local_min
        labeled, num_features = label(pools_mask)
        pools: List[Dict] = []
        for i in range(1, num_features + 1):
            mask = labeled == i
            pools.append(
                {
                    "id": i,
                    "centroid": np.mean(np.where(mask), axis=1),
                    "max_depth": float(np.min(bathymetry[mask])),
                    "area": int(np.sum(mask)),
                }
            )
        return pools

    def _detect_riffles(self, bathymetry: np.ndarray) -> List[Dict]:
        from scipy.ndimage import label, maximum_filter

        local_max = maximum_filter(bathymetry, size=5)
        riffles_mask = bathymetry == local_max
        labeled, num_features = label(riffles_mask)
        riffles: List[Dict] = []
        for i in range(1, num_features + 1):
            mask = labeled == i
            riffles.append(
                {
                    "id": i,
                    "centroid": np.mean(np.where(mask), axis=1),
                    "min_depth": float(np.max(bathymetry[mask])),
                    "area": int(np.sum(mask)),
                }
            )
        return riffles

    def _detect_scour_holes(self, bathymetry: np.ndarray) -> List[Dict]:
        from scipy.ndimage import label

        mean_depth = np.nanmean(bathymetry)
        std_depth = np.nanstd(bathymetry)
        threshold = mean_depth - 2 * std_depth
        scour_mask = bathymetry < threshold
        labeled, num_features = label(scour_mask)
        scour_holes: List[Dict] = []
        for i in range(1, num_features + 1):
            mask = labeled == i
            scour_holes.append(
                {
                    "id": i,
                    "centroid": np.mean(np.where(mask), axis=1),
                    "max_depth": float(np.min(bathymetry[mask])),
                    "volume": float(calculate_volume(bathymetry[mask], mean_depth)),
                }
            )
        return scour_holes

    def _calculate_channel_metrics(self, bathymetry: np.ndarray, features: Dict) -> Dict:
        return {
            "mean_depth": float(np.nanmean(bathymetry)),
            "max_depth": float(np.nanmin(bathymetry)),
            "std_depth": float(np.nanstd(bathymetry)),
            "wetted_area": int(np.sum(~np.isnan(bathymetry))),
            "hydraulic_radius": float(self._calculate_hydraulic_radius(bathymetry)),
            "sinuosity": float(self._calculate_sinuosity(features.get("thalweg", np.array([])))),
            "width_depth_ratio": float(self._calculate_width_depth_ratio(bathymetry)),
        }

    def _calculate_hydraulic_radius(self, bathymetry: np.ndarray) -> float:
        area = np.sum(~np.isnan(bathymetry))
        perimeter = self._calculate_wetted_perimeter(bathymetry)
        return float(area / perimeter) if perimeter > 0 else 0.0

    def _calculate_wetted_perimeter(self, bathymetry: np.ndarray) -> float:
        from scipy.ndimage import binary_erosion

        mask = ~np.isnan(bathymetry)
        eroded = binary_erosion(mask)
        perimeter_mask = mask & ~eroded
        return float(np.sum(perimeter_mask))

    def _calculate_sinuosity(self, thalweg: np.ndarray) -> float:
        if thalweg is None:
            return 1.0
        th = np.asarray(thalweg)
        if th.ndim != 2 or th.shape[0] < 2 or th.shape[1] != 2:
            return 1.0
        diffs = np.diff(th, axis=0)
        path_length = float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))
        straight_distance = float(np.sqrt(np.sum((th[-1] - th[0]) ** 2)))
        return path_length / straight_distance if straight_distance > 0 else 1.0

    def _calculate_width_depth_ratio(self, bathymetry: np.ndarray) -> float:
        width = bathymetry.shape[1]
        mean_depth = float(np.nanmean(bathymetry))
        return float(width / abs(mean_depth)) if mean_depth != 0 else 0.0

    def _detect_water_surface(self, point_cloud: np.ndarray) -> np.ndarray:
        z_values = point_cloud[:, 2]
        percentile_95 = np.percentile(z_values, 95)
        return np.full_like(z_values, percentile_95)

    def _apply_refraction_correction(
        self, point_cloud: np.ndarray, water_surface: np.ndarray
    ) -> np.ndarray:
        n_air = 1.0
        n_water = 1.333
        corrected = point_cloud.copy()
        underwater = point_cloud[:, 2] < water_surface
        if np.any(underwater):
            depth = water_surface[underwater] - point_cloud[underwater, 2]
            corrected_depth = depth * (n_water / n_air)
            corrected[underwater, 2] = water_surface[underwater] - corrected_depth
        return corrected

    def _grid_point_cloud(self, points: np.ndarray, resolution: float = 1.0) -> np.ndarray:
        x_min, y_min = points[:, :2].min(axis=0)
        x_max, y_max = points[:, :2].max(axis=0)
        x_bins = np.arange(x_min, x_max + resolution, resolution)
        y_bins = np.arange(y_min, y_max + resolution, resolution)
        grid = np.full((len(y_bins) - 1, len(x_bins) - 1), np.nan)
        for i in range(len(y_bins) - 1):
            for j in range(len(x_bins) - 1):
                mask = (
                    (points[:, 0] >= x_bins[j])
                    & (points[:, 0] < x_bins[j + 1])
                    & (points[:, 1] >= y_bins[i])
                    & (points[:, 1] < y_bins[i + 1])
                )
                if np.any(mask):
                    grid[i, j] = float(np.mean(points[mask, 2]))
        return grid
