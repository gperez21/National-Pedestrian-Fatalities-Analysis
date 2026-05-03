"""
geo.py — Lightweight geographic helpers.

Just two functions: timezone lookup and haversine distance.
"""

from __future__ import annotations

import math

import numpy as np
from timezonefinder import TimezoneFinder

# Module-level singleton to avoid re-initializing the R-tree on every call.
_TF = TimezoneFinder()

_FALLBACK_TZ = "UTC"


def get_timezone(lat: float, lon: float) -> str:
    """Return the IANA timezone string for the given coordinates.

    Wraps timezonefinder.TimezoneFinder.timezone_at(). Falls back to
    'UTC' for locations over open ocean where no timezone is defined.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.

    Returns:
        IANA timezone string, e.g. 'America/Chicago'.
    """
    # TODO: test ocean coordinates, edge-of-Antarctica
    tz = _TF.timezone_at(lat=lat, lng=lon)
    if tz is None:
        return _FALLBACK_TZ
    return tz


def haversine_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    """Compute great-circle distance in kilometres between two points (or arrays of points).

    Vectorized: accepts scalars or NumPy arrays. Mixed scalar/array inputs
    are broadcast normally.

    Args:
        lat1: Latitude of first point(s) in decimal degrees.
        lon1: Longitude of first point(s) in decimal degrees.
        lat2: Latitude of second point(s) in decimal degrees.
        lon2: Longitude of second point(s) in decimal degrees.

    Returns:
        Distance(s) in kilometres, same shape as inputs.
    """
    # TODO: test antipodal points, zero-distance input
    R = 6371.0  # Earth mean radius in km

    φ1 = np.radians(lat1)
    φ2 = np.radians(lat2)
    dφ = np.radians(lat2 - lat1)
    dλ = np.radians(lon2 - lon1)

    a = np.sin(dφ / 2.0) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(dλ / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return R * c
