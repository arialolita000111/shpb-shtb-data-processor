from __future__ import annotations

import numpy as np


def length_to_m(value: float, unit: str) -> float:
    factors = {
        "m": 1.0,
        "cm": 1e-2,
        "mm": 1e-3,
        "um": 1e-6,
        "μm": 1e-6,
    }
    return value * _factor(unit, factors, "length")


def area_to_m2(value: float, unit: str) -> float:
    factors = {
        "m2": 1.0,
        "m^2": 1.0,
        "cm2": 1e-4,
        "cm^2": 1e-4,
        "mm2": 1e-6,
        "mm^2": 1e-6,
    }
    return value * _factor(unit, factors, "area")


def modulus_to_pa(value: float, unit: str) -> float:
    factors = {
        "pa": 1.0,
        "kpa": 1e3,
        "mpa": 1e6,
        "gpa": 1e9,
    }
    return value * _factor(unit, factors, "modulus")


def time_to_s(values: float | np.ndarray, unit: str) -> float | np.ndarray:
    factors = {
        "s": 1.0,
        "ms": 1e-3,
        "us": 1e-6,
        "μs": 1e-6,
        "µs": 1e-6,
    }
    return values * _factor(unit, factors, "time")


def strain_to_unitless(
    values: float | np.ndarray,
    unit: str,
    voltage_to_microstrain_per_volt: float | None = None,
) -> float | np.ndarray:
    normalized = unit.lower()
    if normalized in {"strain", "unitless", "1"}:
        return values
    if normalized in {"microstrain", "ue", "με", "µε"}:
        return values * 1e-6
    if normalized in {"voltage", "v"}:
        if voltage_to_microstrain_per_volt is None:
            raise ValueError("Voltage signals require voltage_to_microstrain_per_volt.")
        return values * voltage_to_microstrain_per_volt * 1e-6
    raise ValueError(f"Unsupported strain unit: {unit}")


def _factor(unit: str, factors: dict[str, float], quantity: str) -> float:
    key = unit.strip().lower().replace("²", "2")
    if key not in factors:
        raise ValueError(f"Unsupported {quantity} unit: {unit}")
    return factors[key]
