"""Custom exceptions for HydroFlow."""


class HydroFlowError(Exception):
    """Base HydroFlow error."""


class DataProcessingError(HydroFlowError):
    """Raised when data processing fails."""


class MonitoringError(HydroFlowError):
    """Raised for monitoring/streaming errors."""
