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
