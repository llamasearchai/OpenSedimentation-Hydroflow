"""Vegetation management and analysis module (raster-free core)."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


class VegetationAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.classifier = None
        self.species_database = self._load_species_database()

    def analyze_satellite_imagery(
        self, imagery_path: str, bands: List[str] = ["red", "green", "blue", "nir"]
    ) -> Dict:
        # Minimal implementation without rasterio to keep deps small for tests
        # Expect a NumPy .npz or .npy input for testability
        data = np.load(imagery_path)
        if isinstance(data, np.lib.npyio.NpzFile):
            imagery = data[list(data.files)[0]]
        else:
            imagery = data
        indices = self._calculate_vegetation_indices(imagery, bands)
        classification = self._classify_vegetation(imagery, indices)
        invasive_areas = self._detect_invasive_species(classification, indices)
        metrics = self._calculate_vegetation_metrics(classification)
        return {
            "classification": classification,
            "indices": indices,
            "invasive_areas": invasive_areas,
            "metrics": metrics,
            "transform": None,
            "crs": None,
        }

    def detect_aquatic_vegetation(
        self, multispectral_data: np.ndarray, water_mask: np.ndarray
    ) -> Dict:
        water_data = multispectral_data * water_mask[np.newaxis, :, :]
        ndavi = self._calculate_ndavi(multispectral_data)
        fai = self._calculate_fai(multispectral_data)
        submerged = (ndavi > 0.1) & (ndavi < 0.4)
        floating = fai > 0.05
        emergent = ndavi > 0.4
        biomass = self._estimate_biomass(submerged, floating, emergent)
        return {
            "submerged": submerged,
            "floating": floating,
            "emergent": emergent,
            "biomass": biomass,
            "total_coverage": float(
                np.sum(submerged | floating | emergent) / max(np.sum(water_mask), 1)
            ),
        }

    def identify_invasive_species(
        self, spectral_data: np.ndarray, known_signatures: Optional[Dict] = None
    ) -> Dict:
        if known_signatures is None:
            known_signatures = self._get_default_invasive_signatures()
        identified: Dict[str, Dict] = {}
        for species_name, signature in known_signatures.items():
            similarity = self._spectral_similarity(spectral_data, signature)
            threshold = self.config.get("invasive_detection_threshold", 0.8)
            detected = similarity > threshold
            if np.any(detected):
                identified[species_name] = {
                    "mask": detected,
                    "area": int(np.sum(detected)),
                    "confidence": float(np.mean(similarity[detected])),
                    "locations": self._extract_locations(detected),
                }
        return identified

    def calculate_removal_priority(
        self, vegetation_map: np.ndarray, flow_impact: np.ndarray, ecological_value: np.ndarray
    ) -> np.ndarray:
        vegetation_norm = vegetation_map / (np.max(vegetation_map) + 1e-12)
        flow_norm = flow_impact / (np.max(flow_impact) + 1e-12)
        ecological_norm = ecological_value / (np.max(ecological_value) + 1e-12)
        priority = (
            self.config.get("vegetation_weight", 0.3) * vegetation_norm
            + self.config.get("flow_weight", 0.5) * flow_norm
            + self.config.get("ecological_weight", 0.2) * (1 - ecological_norm)
        )
        return priority

    def optimize_removal_timing(
        self, species_data: Dict, environmental_conditions: np.ndarray | "pd.DataFrame"
    ) -> "np.ndarray | pd.DataFrame":
        import pandas as pd  # Local import

        schedule = []
        for species, data in species_data.items():
            growth_rate = data.get("growth_rate", [])
            reproduction_period = data.get("reproduction_period", [])
            optimal_months = self._find_optimal_removal_window(growth_rate, reproduction_period)
            if isinstance(environmental_conditions, pd.DataFrame):
                feasible_months = self._check_environmental_feasibility(
                    optimal_months, environmental_conditions
                )
            else:
                feasible_months = optimal_months
            schedule.append(
                {
                    "species": species,
                    "optimal_months": optimal_months,
                    "feasible_months": feasible_months,
                    "priority": data.get("priority", "medium"),
                    "estimated_effort": data.get("effort", "medium"),
                }
            )
        return pd.DataFrame(schedule)

    def estimate_regrowth_potential(
        self, species: str, removal_method: str, environmental_factors: Dict
    ) -> Dict:
        species_info = self.species_database.get(species, {})
        base_rate = species_info.get("regrowth_rate", 0.5)
        method_effectiveness = {
            "mechanical": 0.7,
            "chemical": 0.9,
            "biological": 0.6,
            "manual": 0.8,
        }
        removal_factor = method_effectiveness.get(removal_method, 0.5)
        temp = environmental_factors.get("temperature", 20)
        temp_factor = self._temperature_growth_factor(temp)
        nutrient_factor = environmental_factors.get("nutrients", 1.0)
        light_factor = environmental_factors.get("light", 1.0)
        regrowth_rate = (
            base_rate * (1 - removal_factor) * temp_factor * nutrient_factor * light_factor
        )
        time_to_regrowth = 1 / regrowth_rate if regrowth_rate > 0 else float("inf")
        return {
            "regrowth_rate": float(regrowth_rate),
            "time_to_regrowth_days": float(time_to_regrowth * 365),
            "probability_of_regrowth": float(min(regrowth_rate * 2, 1.0)),
            "recommended_monitoring_frequency": self._recommend_monitoring(regrowth_rate),
        }

    # Internals
    def _calculate_vegetation_indices(self, imagery: np.ndarray, bands: List[str]) -> Dict:
        indices: Dict[str, np.ndarray] = {}
        band_map = {band: i for i, band in enumerate(bands)}
        if "nir" in band_map and "red" in band_map:
            nir = imagery[band_map["nir"]]
            red = imagery[band_map["red"]]
            indices["ndvi"] = (nir - red) / (nir + red + 1e-10)
        if all(b in band_map for b in ["nir", "red", "blue"]):
            nir = imagery[band_map["nir"]]
            red = imagery[band_map["red"]]
            blue = imagery[band_map["blue"]]
            indices["evi"] = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
        if "nir" in band_map and "red" in band_map:
            nir = imagery[band_map["nir"]]
            red = imagery[band_map["red"]]
            L = 0.5
            indices["savi"] = (nir - red) / (nir + red + L) * (1 + L)
        return indices

    def _classify_vegetation(self, imagery: np.ndarray, indices: Dict) -> np.ndarray:
        ndvi = indices.get("ndvi", np.zeros_like(imagery[0]))
        classification = np.zeros_like(ndvi, dtype=int)
        classification[ndvi < 0] = 0
        classification[(ndvi >= 0) & (ndvi < 0.2)] = 1
        classification[(ndvi >= 0.2) & (ndvi < 0.4)] = 2
        classification[(ndvi >= 0.4) & (ndvi < 0.6)] = 3
        classification[ndvi >= 0.6] = 4
        return classification

    def _detect_invasive_species(self, classification: np.ndarray, indices: Dict) -> np.ndarray:
        ndvi = indices.get("ndvi", np.zeros_like(classification))
        water_mask = classification == 0
        kernel = np.ones((5, 5), np.uint8)
        # simple dilation without cv2
        from scipy.ndimage import binary_dilation

        dilated = binary_dilation(water_mask, structure=kernel)
        water_margin = dilated & (~water_mask)
        invasive_mask = (ndvi > 0.7) & water_margin
        vegetation_mask = classification >= 3
        clusters = self._detect_unusual_clusters(vegetation_mask)
        return invasive_mask | clusters

    def _calculate_vegetation_metrics(self, classification: np.ndarray) -> Dict:
        total_pixels = classification.size
        return {
            "total_area": int(total_pixels),
            "water_coverage": float(np.sum(classification == 0) / total_pixels),
            "bare_soil_coverage": float(np.sum(classification == 1) / total_pixels),
            "sparse_vegetation_coverage": float(np.sum(classification == 2) / total_pixels),
            "moderate_vegetation_coverage": float(np.sum(classification == 3) / total_pixels),
            "dense_vegetation_coverage": float(np.sum(classification == 4) / total_pixels),
            "vegetation_fragmentation": float(self._calculate_fragmentation(classification >= 2)),
        }

    def _load_species_database(self) -> Dict:
        return {
            "water_hyacinth": {
                "growth_rate": 0.15,
                "reproduction_period": [4, 5, 6, 7, 8, 9],
                "optimal_temperature": 28,
                "regrowth_rate": 0.8,
            },
            "giant_salvinia": {
                "growth_rate": 0.25,
                "reproduction_period": [5, 6, 7, 8],
                "optimal_temperature": 25,
                "regrowth_rate": 0.9,
            },
            "cattail": {
                "growth_rate": 0.05,
                "reproduction_period": [6, 7, 8],
                "optimal_temperature": 22,
                "regrowth_rate": 0.6,
            },
        }

    def _calculate_ndavi(self, data: np.ndarray) -> np.ndarray:
        nir = data[3] if data.shape[0] > 3 else data[0]
        blue = data[2] if data.shape[0] > 2 else data[0]
        return (nir - blue) / (nir + blue + 1e-10)

    def _calculate_fai(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] >= 4:
            nir = data[3]
            red = data[0]
            swir = data[4] if data.shape[0] > 4 else nir * 0.8
            baseline = red + (swir - red) * (860 - 660) / (1640 - 660)
            return nir - baseline
        return np.zeros_like(data[0])

    def _estimate_biomass(
        self, submerged: np.ndarray, floating: np.ndarray, emergent: np.ndarray
    ) -> Dict:
        biomass_factors = {"submerged": 0.5, "floating": 2.0, "emergent": 3.5}
        return {
            "total": float(
                np.sum(submerged) * biomass_factors["submerged"]
                + np.sum(floating) * biomass_factors["floating"]
                + np.sum(emergent) * biomass_factors["emergent"]
            ),
            "submerged": float(np.sum(submerged) * biomass_factors["submerged"]),
            "floating": float(np.sum(floating) * biomass_factors["floating"]),
            "emergent": float(np.sum(emergent) * biomass_factors["emergent"]),
        }

    def _get_default_invasive_signatures(self) -> Dict:
        return {
            "water_hyacinth": np.array([0.15, 0.25, 0.18, 0.75]),
            "giant_salvinia": np.array([0.12, 0.28, 0.15, 0.78]),
            "hydrilla": np.array([0.08, 0.15, 0.12, 0.45]),
        }

    def _spectral_similarity(self, data: np.ndarray, signature: np.ndarray) -> np.ndarray:
        n_bands = min(data.shape[0], len(signature))
        data_subset = data[:n_bands]
        signature_subset = signature[:n_bands]
        dot_product = np.sum(data_subset * signature_subset[:, np.newaxis, np.newaxis], axis=0)
        data_norm = np.sqrt(np.sum(data_subset**2, axis=0))
        signature_norm = float(np.sqrt(np.sum(signature_subset**2)))
        cos_angle = dot_product / (data_norm * signature_norm + 1e-12)
        cos_angle = np.clip(cos_angle, -1, 1)
        similarity = (cos_angle + 1) / 2
        return similarity

    def _extract_locations(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        locations: List[Tuple[int, int]] = []
        labeled, num_features = ndimage.label(mask)
        for i in range(1, num_features + 1):
            component = labeled == i
            centroid = ndimage.center_of_mass(component)
            locations.append((int(centroid[0]), int(centroid[1])))
        return locations

    def _find_optimal_removal_window(
        self, growth_rate: List[float], reproduction_period: List[int]
    ) -> List[int]:
        all_months = set(range(1, 13))
        non_reproduction = list(all_months - set(reproduction_period))
        if growth_rate:
            non_reproduction.sort(
                key=lambda m: growth_rate[m - 1] if m - 1 < len(growth_rate) else 1
            )
            return non_reproduction[:3]
        return non_reproduction[:3]

    def _check_environmental_feasibility(
        self, months: List[int], conditions: "np.ndarray | pd.DataFrame"
    ) -> List[int]:
        import pandas as pd

        feasible: List[int] = []
        for month in months:
            month_conditions = conditions[conditions["month"] == month]
            if not month_conditions.empty:
                avg_water_level = float(month_conditions["water_level"].mean())
                avg_precipitation = float(month_conditions["precipitation"].mean())
                if avg_water_level < 2.0 and avg_precipitation < 50:
                    feasible.append(month)
        return feasible if feasible else months[:1]

    def _temperature_growth_factor(self, temperature: float) -> float:
        optimal = 27.5
        if temperature < 10 or temperature > 40:
            return 0.1
        if 20 <= temperature <= 35:
            return 1.0 - abs(temperature - optimal) / 20
        return 0.5

    def _recommend_monitoring(self, regrowth_rate: float) -> str:
        if regrowth_rate > 0.8:
            return "Weekly"
        if regrowth_rate > 0.5:
            return "Bi-weekly"
        if regrowth_rate > 0.3:
            return "Monthly"
        return "Quarterly"

    def _detect_unusual_clusters(self, vegetation_mask: np.ndarray) -> np.ndarray:
        points = np.column_stack(np.where(vegetation_mask))
        if len(points) == 0:
            return np.zeros_like(vegetation_mask)
        # Simple clustering by connected components size threshold
        labeled, num = ndimage.label(vegetation_mask)
        unusual = np.zeros_like(vegetation_mask)
        for i in range(1, num + 1):
            component = labeled == i
            if np.sum(component) < 100:
                unusual[component] = True
        return unusual

    def _calculate_fragmentation(self, mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        labeled, num_patches = ndimage.label(mask)
        total_area = np.sum(mask)
        frag = num_patches / (total_area / 100.0)
        return float(min(frag, 1.0))
