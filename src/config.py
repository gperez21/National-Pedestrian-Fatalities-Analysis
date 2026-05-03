"""
config.py — Paths, constants, and city-pair definitions. No logic.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root and data paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# FARS download base URL (NHTSA bulk downloads)
# ---------------------------------------------------------------------------

FARS_BASE_URL: str = "https://static.nhtsa.gov/nhtsa/downloads/FARS"

# ---------------------------------------------------------------------------
# Year range
# NHTSA typically publishes through the previous calendar year.
# Adjust upper bound as new data is released.
# ---------------------------------------------------------------------------

YEARS = range(1995, 2025)

# ---------------------------------------------------------------------------
# Commute window (local hour, inclusive start, exclusive end)
# ---------------------------------------------------------------------------

COMMUTE_HOURS: tuple[int, int] = (17, 19)  # 5–7 PM local

# ---------------------------------------------------------------------------
# City pairs for natural-experiment comparisons.
# Each pair straddles a timezone boundary; "east" and "west" refer to
# relative position, not necessarily the timezone label.
#
# Keys: name, lat, lon (decimal degrees), tz (IANA timezone string).
# ---------------------------------------------------------------------------

CITY_PAIRS: dict = {
    # Indiana (Eastern) vs. Illinois (Central)
    "indiana_illinois": {
        "east": {
            "name": "Terre Haute, IN",
            "lat": 39.4667,
            "lon": -87.4139,
            "tz": "America/Indiana/Indianapolis",
        },
        "west": {
            "name": "Champaign, IL",
            "lat": 40.1164,
            "lon": -88.2434,
            "tz": "America/Chicago",
        },
    },
    # Florida Eastern vs. Central (Marianna / Tallahassee area straddles the line)
    "florida_et_ct": {
        "east": {
            "name": "Marianna, FL",
            "lat": 30.7741,
            "lon": -85.2294,
            "tz": "America/New_York",
        },
        "west": {
            "name": "Tallahassee, FL",
            "lat": 30.4518,
            "lon": -84.2807,
            "tz": "America/Chicago",
        },
    },
    # Texas Mountain (El Paso) vs. Central (San Antonio area)
    "texas_mt_ct": {
        "east": {
            "name": "San Antonio, TX",
            "lat": 29.4241,
            "lon": -98.4936,
            "tz": "America/Chicago",
        },
        "west": {
            "name": "El Paso, TX",
            "lat": 31.7619,
            "lon": -106.4850,
            "tz": "America/Denver",
        },
    },
    # Idaho (Mountain) vs. Oregon (Pacific)
    "idaho_oregon": {
        "east": {
            "name": "Ontario, OR",
            "lat": 44.0268,
            "lon": -116.9629,
            "tz": "America/Boise",
        },
        "west": {
            "name": "Baker City, OR",
            "lat": 44.7749,
            "lon": -117.8344,
            "tz": "America/Los_Angeles",
        },
    },
}
