"""Sediment transport modeling module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import logging
import numpy as np
import xarray as xr


logger = logging.getLogger(__name__)


@dataclass
class SedimentProperties:
    d50: float
    d90: float
    density: float = 2650
    porosity: float = 0.4
    angle_of_repose: float = 30
    settling_velocity: Optional[float] = None

    def __post_init__(self):
        if self.settling_velocity is None:
            self.settling_velocity = self.calculate_settling_velocity()

    def calculate_settling_velocity(self) -> float:
        d = self.d50 / 1000.0
        rho_water = 1000.0
        nu = 1e-6
        g = 9.81
        d_star = d * ((self.density / rho_water - 1.0) * g / (nu**2)) ** (1.0 / 3.0)
        if d_star < 1:
            ws = (self.density - rho_water) * g * d**2 / (18 * rho_water * nu)
        elif d_star < 1000:
            ws = nu / d * (np.sqrt(25 + 1.2 * d_star**2) - 5) ** 1.5
        else:
            ws = np.sqrt((self.density / rho_water - 1) * g * d)
        return float(ws)


class SedimentTransportModel:
    def __init__(self, config: Dict):
        self.config = config
        self.sediment_props: Optional[SedimentProperties] = None
        self.flow_field: Optional[Dict[str, np.ndarray]] = None
        self.bathymetry: Optional[np.ndarray] = None

    def set_sediment_properties(self, properties: SedimentProperties):
        self.sediment_props = properties

    def set_flow_field(self, velocity: np.ndarray, depth: np.ndarray):
        self.flow_field = {"velocity": velocity, "depth": depth}

    def calculate_bed_shear_stress(self, velocity: np.ndarray, depth: np.ndarray, roughness: float = 0.03) -> np.ndarray:
        rho_water = 1000.0
        g = 9.81
        tau = rho_water * g * roughness**2 * velocity**2 / (np.power(depth, 1 / 3) + 1e-12)
        return tau

    def calculate_shields_parameter(self, shear_stress: np.ndarray) -> np.ndarray:
        if self.sediment_props is None:
            raise ValueError("Sediment properties not set")
        rho_water = 1000.0
        g = 9.81
        d = self.sediment_props.d50 / 1000.0
        theta = shear_stress / ((self.sediment_props.density - rho_water) * g * d + 1e-12)
        return theta

    def calculate_critical_shear_stress(self) -> float:
        if self.sediment_props is None:
            raise ValueError("Sediment properties not set")
        rho_water = 1000.0
        g = 9.81
        d = self.sediment_props.d50 / 1000.0
        theta_cr = 0.047
        tau_cr = theta_cr * (self.sediment_props.density - rho_water) * g * d
        return float(tau_cr)

    def calculate_bedload_transport(self, shear_stress: np.ndarray, method: str = "meyer-peter") -> np.ndarray:
        if method == "meyer-peter":
            return self._meyer_peter_muller(shear_stress)
        if method == "einstein":
            return self._einstein_bedload(shear_stress)
        if method == "vanrijn":
            return self._vanrijn_bedload(shear_stress)
        raise ValueError(f"Unknown method: {method}")

    def calculate_suspended_load(self, velocity: np.ndarray, depth: np.ndarray, concentration: Optional[np.ndarray] = None) -> np.ndarray:
        if self.sediment_props is None:
            raise ValueError("Sediment properties not set")
        if concentration is None:
            concentration = self._calculate_reference_concentration(velocity, depth)
        qs = concentration * velocity * depth
        return qs

    def solve_exner_equation(self, initial_bed: np.ndarray, time_span: Tuple[float, float], dt: float = 1.0) -> xr.Dataset:
        times = np.arange(time_span[0], time_span[1], dt)
        bed_evolution = np.zeros((len(times), *initial_bed.shape))
        bed_evolution[0] = initial_bed
        for i, _t in enumerate(times[1:], 1):
            velocity = self.flow_field["velocity"] if self.flow_field else np.ones_like(initial_bed)
            depth = self.flow_field["depth"] if self.flow_field else np.ones_like(initial_bed)
            shear_stress = self.calculate_bed_shear_stress(velocity, depth)
            bedload = self.calculate_bedload_transport(shear_stress)
            div_qs = self._calculate_divergence(bedload)
            porosity = self.sediment_props.porosity if self.sediment_props else 0.4
            bed_change = -dt / (1 - porosity) * div_qs
            bed_evolution[i] = bed_evolution[i - 1] + bed_change
        ds = xr.Dataset(
            {
                "bed_elevation": (("time", "y", "x"), bed_evolution),
                "bed_change_rate": (("time", "y", "x"), np.diff(bed_evolution, axis=0, prepend=bed_evolution[[0]])),
            },
            coords={"time": times, "y": np.arange(initial_bed.shape[0]), "x": np.arange(initial_bed.shape[1])},
        )
        return ds

    def predict_deposition_patterns(self, flow_scenarios: List[Dict], time_horizon: float = 365 * 24 * 3600) -> np.ndarray:
        if self.bathymetry is None:
            raise ValueError("Bathymetry not set")
        total_deposition = np.zeros_like(self.bathymetry)
        for scenario in flow_scenarios:
            velocity = scenario["velocity"]
            depth = scenario["depth"]
            probability = scenario["probability"]
            duration = scenario.get("duration", time_horizon * probability)
            shear_stress = self.calculate_bed_shear_stress(velocity, depth)
            transport = self.calculate_bedload_transport(shear_stress)
            deposition = self._calculate_deposition(transport, duration)
            total_deposition += deposition * probability
        return total_deposition

    def calibrate_model(self, observed_transport: np.ndarray, measured_conditions: Dict) -> Dict:
        def objective(params):
            self.config.update({"roughness": params[0], "transport_coefficient": params[1]})
            predicted = self.calculate_bedload_transport(measured_conditions["shear_stress"])
            error = np.sum((predicted - observed_transport) ** 2)
            return float(error)

        x0 = np.array([0.03, 8.0])
        # Simple coordinate descent to avoid adding scipy.optimize for tests
        step = np.array([0.01, 1.0])
        best = x0.copy()
        best_val = objective(best)
        for _ in range(50):
            improved = False
            for i in range(2):
                for delta in (-step[i], step[i]):
                    candidate = best.copy()
                    candidate[i] += delta
                    val = objective(candidate)
                    if val < best_val:
                        best, best_val = candidate, val
                        improved = True
            if not improved:
                step *= 0.5
            if np.all(step < 1e-3):
                break
        rmse = float(np.sqrt(best_val / len(observed_transport)))
        return {"roughness": float(best[0]), "transport_coefficient": float(best[1]), "rmse": rmse}

    # Internals
    def _meyer_peter_muller(self, shear_stress: np.ndarray) -> np.ndarray:
        if self.sediment_props is None:
            raise ValueError("Sediment properties not set")
        rho_water = 1000.0
        rho_s = self.sediment_props.density
        g = 9.81
        d = self.sediment_props.d50 / 1000.0
        tau_cr = self.calculate_critical_shear_stress()
        excess = np.maximum(shear_stress - tau_cr, 0)
        qb = 8 * np.sqrt((rho_s / rho_water - 1) * g * d**3) * excess ** 1.5
        return qb

    def _einstein_bedload(self, shear_stress: np.ndarray) -> np.ndarray:
        if self.sediment_props is None:
            raise ValueError("Sediment properties not set")
        rho_water = 1000.0
        rho_s = self.sediment_props.density
        g = 9.81
        d = self.sediment_props.d50 / 1000.0
        phi = shear_stress / ((rho_s - rho_water) * g * d + 1e-12)
        # Vectorized safe exponential
        phi_b = np.where(phi > 0, 2.15 * np.exp(-0.391 / (phi + 1e-12)), 0.0)
        qb = phi_b * np.sqrt((rho_s / rho_water - 1) * g * d**3)
        return qb

    def _vanrijn_bedload(self, shear_stress: np.ndarray) -> np.ndarray:
        if self.sediment_props is None:
            raise ValueError("Sediment properties not set")
        rho_water = 1000.0
        rho_s = self.sediment_props.density
        g = 9.81
        d = self.sediment_props.d50 / 1000.0
        tau_cr = self.calculate_critical_shear_stress()
        T = (shear_stress - tau_cr) / (tau_cr + 1e-12)
        T = np.maximum(T, 0)
        nu = 1e-6
        D_star = d * ((rho_s / rho_water - 1) * g / (nu**2)) ** (1.0 / 3.0)
        qb = 0.053 * np.sqrt((rho_s / rho_water - 1) * g * d**3) * (T ** 2.1) * (D_star ** (-0.3))
        return qb

    def _calculate_reference_concentration(self, velocity: np.ndarray, depth: np.ndarray) -> np.ndarray:
        if self.sediment_props is None:
            raise ValueError("Sediment properties not set")
        shear_stress = self.calculate_bed_shear_stress(velocity, depth)
        u_star = np.sqrt(np.maximum(shear_stress, 0) / 1000.0)
        d = self.sediment_props.d50 / 1000.0
        a = 2 * d
        tau_cr = self.calculate_critical_shear_stress()
        T = (shear_stress - tau_cr) / (tau_cr + 1e-12)
        T = np.maximum(T, 0)
        c_a = 0.015 * (np.where(a > 0, d / a, 0)) * (T ** 1.5) * (self.sediment_props.d50 ** (-0.3))
        c_a = np.maximum(c_a, 0)
        return c_a

    def _calculate_divergence(self, flux: np.ndarray) -> np.ndarray:
        if flux.ndim == 2:
            dy, dx = np.gradient(flux)
            return dx + dy
        return np.gradient(flux)

    def _calculate_deposition(self, transport: np.ndarray, duration: float) -> np.ndarray:
        porosity = self.sediment_props.porosity if self.sediment_props else 0.4
        density = self.sediment_props.density if self.sediment_props else 2650
        divergence = self._calculate_divergence(transport)
        deposition = -divergence * duration / ((1 - porosity) * density + 1e-12)
        return deposition


