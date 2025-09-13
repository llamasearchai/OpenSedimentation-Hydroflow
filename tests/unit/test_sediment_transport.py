"""Unit tests for sediment transport modeling (subset)."""

import numpy as np

from hydroflow.analysis.sediment_transport import SedimentProperties, SedimentTransportModel


def test_properties_ws_range():
    props = SedimentProperties(d50=0.5, d90=1.0)
    ws = props.calculate_settling_velocity()
    assert ws > 0
    assert ws < 1.0


def test_bed_shear_stress_and_shields():
    model = SedimentTransportModel({})
    props = SedimentProperties(d50=0.5, d90=1.0)
    model.set_sediment_properties(props)
    v = np.array([1.0, 1.5, 2.0])
    h = np.array([2.0, 3.0, 4.0])
    tau = model.calculate_bed_shear_stress(v, h)
    assert tau.shape == v.shape
    theta = model.calculate_shields_parameter(tau)
    assert theta.shape == v.shape


def test_exner_solver_shapes():
    model = SedimentTransportModel({})
    model.set_sediment_properties(SedimentProperties(d50=0.5, d90=1.0))
    model.set_flow_field(np.ones((10, 10)) * 1.5, np.ones((10, 10)) * 3.0)
    initial_bed = np.ones((10, 10)) * -5.0
    ds = model.solve_exner_equation(initial_bed, (0, 10), dt=1)
    assert "bed_elevation" in ds
    assert ds["bed_elevation"].shape[1:] == initial_bed.shape
