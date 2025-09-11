"""Configuration management for HydroFlow (lightweight dataclass version)."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional
import os
import yaml


@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    name: str = "hydroflow"
    user: str = "hydroflow"
    password: str = ""
    pool_size: int = 10

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class SensorConfig:
    sonar_frequency: float = 200.0
    lidar_resolution: float = 1.0
    sampling_rate: float = 10.0
    buffer_size: int = 1000


@dataclass
class ProcessingConfig:
    batch_size: int = 1000
    max_workers: int = 4
    chunk_size: int = 10000
    cache_ttl: int = 3600
    enable_gpu: bool = False


@dataclass
class MLConfig:
    model_path: Path = Path("models")
    training_split: float = 0.8
    validation_split: float = 0.1
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001


@dataclass
class NotificationConfig:
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    from_email: str = "noreply@hydroflow.io"
    alert_threshold: float = 0.8


@dataclass
class Config:
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    notification: NotificationConfig = field(default_factory=NotificationConfig)

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    # Storage Configuration
    data_dir: Path = Path("data")
    cache_dir: Path = Path("cache")
    output_dir: Path = Path("output")

    def __post_init__(self):
        # Allow dict inputs for nested configs
        if isinstance(self.database, dict):
            self.database = DatabaseConfig(**self.database)
        if isinstance(self.sensor, dict):
            self.sensor = SensorConfig(**self.sensor)
        if isinstance(self.processing, dict):
            self.processing = ProcessingConfig(**self.processing)
        if isinstance(self.ml, dict):
            # Coerce model_path to Path
            ml = dict(self.ml)
            if isinstance(ml.get("model_path"), str):
                ml["model_path"] = Path(ml["model_path"])  # type: ignore[index]
            self.ml = MLConfig(**ml)
        if isinstance(self.notification, dict):
            self.notification = NotificationConfig(**self.notification)

        # Ensure directories exist
        for p in [self.data_dir, self.cache_dir, self.output_dir, self.ml.model_path]:
            Path(p).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def save_yaml(self, path: Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.dict(), f, default_flow_style=False)

    def dict(self) -> Dict:
        d = asdict(self)
        # Convert Paths to strings for serialization
        def convert(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        return convert(d)


def get_config() -> Config:
    config_path = os.getenv("HYDROFLOW_CONFIG", "config/default.yaml")
    p = Path(config_path)
    if p.exists():
        return Config.from_yaml(p)
    return Config()


