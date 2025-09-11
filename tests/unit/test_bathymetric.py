"""Unit tests for bathymetric analysis (subset)."""

import numpy as np
import xarray as xr

from hydroflow.analysis.bathymetric import BathymetricAnalyzer


def sample_points():
    x = np.linspace(0, 100, 50)
    y = np.linspace(0, 100, 50)
    xx, yy = np.meshgrid(x, y)
    z = -5 - 3 * np.exp(-((xx - 50) ** 2 + (yy - 50) ** 2) / 500)
    points = np.column_stack([xx.ravel(), yy.ravel(), z.ravel()])
    return points, z


def test_process_multibeam_sonar():
    analyzer = BathymetricAnalyzer({})
    pts, _ = sample_points()
    intensity = np.random.rand(len(pts))
    data = np.column_stack([pts, intensity])
    ds = analyzer.process_multibeam_sonar(data, {"sound_velocity": 1500, "frequency": 200})
    assert isinstance(ds, xr.Dataset)
    assert "depth" in ds
    assert "quality" in ds


def test_create_bathymetric_surface_methods():
    analyzer = BathymetricAnalyzer({})
    pts, _ = sample_points()
    for method in ["idw", "spline", "triangulation"]:
        surface = analyzer.create_bathymetric_surface(pts, method=method, resolution=5.0)
        assert isinstance(surface, np.ndarray)
        assert surface.ndim == 2
        assert not np.all(np.isnan(surface))


def test_volume_change():
    analyzer = BathymetricAnalyzer({})
    s1 = np.ones((50, 50)) * -5
    s2 = np.ones((50, 50)) * -4
    s2[20:30, 20:30] = -6
    s2[35:40, 35:40] = -3
    result = analyzer.calculate_volume_change(s1, s2, cell_size=1.0)
    assert result["total_deposition"] > 0
    assert result["total_erosion"] > 0


