"""Data models for PymordialDroid."""

from pydantic import BaseModel


class DeviceRecord(BaseModel):
    """Schema for a saved device in our JSON fleet inventory file."""

    serial: str
    ip: str
    port: int = 5555
    name: str = "Unknown"
    pin: str | None = None


class FleetDeviceHealth(BaseModel):
    """Health snapshot for a single fleet device."""

    ip: str
    name: str
    connected: bool
    battery_level: int | None = None
    last_action: str | None = None


class FleetOperationResult(BaseModel):
    """Per-device result of a fleet-wide operation."""

    ip: str
    name: str
    success: bool
    detail: str | None = None
    data: dict | None = None
