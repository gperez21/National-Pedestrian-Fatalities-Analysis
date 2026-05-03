"""
fars_loader.py — Download and parse FARS annual data files.

Handles column-name drift across years defensively. Does NOT perform
analysis or filtering beyond what's needed for a clean per-fatality frame.
"""

from __future__ import annotations

from importlib.resources import path
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from tqdm import tqdm

from . import config

# ---------------------------------------------------------------------------
# Sentinel values that NHTSA uses for "unknown" — coerce these to NaN.
# ---------------------------------------------------------------------------

# Latitude/longitude unknowns
_LAT_LON_SENTINELS = {77.7700, 77.770, 888.8880, 888.888}
# Hour/minute unknowns
_HOUR_SENTINELS = {99}
_MINUTE_SENTINELS = {99}

# Known zip-name patterns, newest-first.
# We try each pattern in order until one yields a 200 response.
_ZIP_PATTERNS = [
    # 2015-present: FARS{year}NationalCSV.zip
    "{base}/{year}/National/FARS{year}NationalCSV.zip",
    # Alternate: accessible through older directory layout
    "{base}/{year}/FARS{year}NationalCSV.zip",
    # Some years use a two-digit suffix
    "{base}/{year}/National/FARS{year}National.zip",
    # Very old (pre-2015) layout
    "{base}/{year}/FARSNATIONALCSV.zip",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_file_in_zip(zf: zipfile.ZipFile, stem: str) -> str | None:
    """Return the name of the first member whose stem matches *stem* (case-insensitive)."""
    stem_lower = stem.lower()
    for name in zf.namelist():
        if Path(name).stem.lower() == stem_lower and name.lower().endswith(".csv"):
            return name
    return None


def _coerce_lat_lon(series: pd.Series) -> pd.Series:
    """Replace known sentinel lat/lon values with NaN and cast to float."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(~s.isin(_LAT_LON_SENTINELS), other=float("nan"))
    # NHTSA uses 77.7700 / 888.8880 — also catch truncated forms
    s = s.where(~((s >= 77.769) & (s <= 77.771)), other=float("nan"))
    s = s.where(~((s >= 888.887) & (s <= 888.889)), other=float("nan"))
    return s


def _coerce_hour(series: pd.Series) -> pd.Series:
    """Replace unknown hour sentinel (99) with NaN and cast to Int64."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(s != 99, other=float("nan"))
    return s.astype("Int64")


def _coerce_minute(series: pd.Series) -> pd.Series:
    """Replace unknown minute sentinel (99) with NaN and cast to Int64."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(s != 99, other=float("nan"))
    return s.astype("Int64")

def _read_csv_safe(path, **kwargs) -> pd.DataFrame:
    import io
    if hasattr(path, "read"):
        # Buffer all bytes upfront — ZipExtFile cannot seek back to 0 on retry
        raw = path.read()
        for encoding in ("utf-8", "latin-1", "windows-1252"):
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=encoding, low_memory=False, **kwargs)
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode file-like object with any known encoding")
    for encoding in ("utf-8", "latin-1", "windows-1252"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path} with any known encoding")



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_fars_year(year: int, force: bool = False) -> Path:
    """Download the annual FARS zip for *year* into data/raw/.

    Skips download if the file already exists unless *force* is True.
    Tries multiple known URL patterns; logs a warning if none succeed.

    Args:
        year: Four-digit year, e.g. 2022.
        force: Re-download even if the local file already exists.

    Returns:
        Path to the downloaded zip file, or raises FileNotFoundError if not found.
    """
    dest = config.DATA_RAW / f"FARS{year}NationalCSV.zip"
    if dest.exists() and not force:
        print(f"[fars_loader] {dest.name} already present, skipping download.")
        return dest

    for pattern in _ZIP_PATTERNS:
        url = pattern.format(base=config.FARS_BASE_URL, year=year)
        try:
            resp = requests.get(url, stream=True, timeout=60)
            if resp.status_code == 200:
                total = int(resp.headers.get("content-length", 0))
                print(f"[fars_loader] Downloading {url}")
                with open(dest, "wb") as fh:
                    with tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc=f"FARS {year}",
                        leave=False,
                    ) as pbar:
                        for chunk in resp.iter_content(chunk_size=65536):
                            fh.write(chunk)
                            pbar.update(len(chunk))
                return dest
        except requests.RequestException as exc:
            print(f"[fars_loader] Warning: request failed for {url}: {exc}")
            continue

    print(f"[fars_loader] Warning: could not download FARS data for {year}. Tried:")
    for p in _ZIP_PATTERNS:
        print(f"  {p.format(base=config.FARS_BASE_URL, year=year)}")
    raise FileNotFoundError(f"FARS zip for {year} not found at any known URL.")


def load_accident(year: int, force: bool = False) -> pd.DataFrame:
    """Load accident.csv for *year* from its FARS zip.

    Returns a DataFrame with at minimum:
        ST_CASE, STATE, COUNTY, LATITUDE, LONGITUD, YEAR,
        MONTH, DAY, HOUR, MINUTE, RUR_URB (where available).

    Sentinel values are coerced to NaN.

    Args:
        year: Four-digit year.
        force: Re-download the zip if it is missing (passed to download_fars_year).

    Returns:
        pd.DataFrame with one row per accident.
    """
    zip_path = config.DATA_RAW / f"FARS{year}NationalCSV.zip"
    if zip_path.exists() and not force:
        print(f"[fars_loader] Raw zip for {year} already present, skipping download.")
    else:
        zip_path = download_fars_year(year, force=force)

    with zipfile.ZipFile(zip_path, "r") as zf:
        member = _find_file_in_zip(zf, "accident")
        if member is None:
            raise FileNotFoundError(
                f"accident.csv not found in {zip_path}. Members: {zf.namelist()}"
            )
        with zf.open(member) as fh:
            df = _read_csv_safe(fh)

    # Normalise column names to upper-case
    df.columns = [c.strip().upper() for c in df.columns]

    # Mandatory columns — rename variants if needed
    _rename_map = {
        "LONGITUD": "LONGITUD",  # canonical
        "LONGITUDE": "LONGITUD",  # seen in some years
    }
    df.rename(columns={k: v for k, v in _rename_map.items() if k in df.columns}, inplace=True)

    # LGT_COND values: 1=Daylight, 2=Dark-Not Lighted, 3=Dark-Lighted,
    #   4=Dawn, 5=Dusk, 6=Dark-Unknown Lighting, 7+=Other/Unknown/Not Reported
    # FUNC_SYS: FARS functional road class (rural 1-6, urban 11-16, 99=unknown)
    desired = ["ST_CASE", "STATE", "COUNTY", "LATITUDE", "LONGITUD",
               "YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "RUR_URB",
               "LGT_COND", "FUNC_SYS"]
    present = [c for c in desired if c in df.columns]
    df = df[present].copy()

    # Inject YEAR if absent (some early files omit it)
    if "YEAR" not in df.columns:
        df["YEAR"] = year

    # Coerce numeric types
    for col in ["ST_CASE", "STATE", "COUNTY", "YEAR", "MONTH", "DAY"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    if "LATITUDE" in df.columns:
        df["LATITUDE"] = _coerce_lat_lon(df["LATITUDE"])
    if "LONGITUD" in df.columns:
        df["LONGITUD"] = _coerce_lat_lon(df["LONGITUD"])

    if "HOUR" in df.columns:
        df["HOUR"] = _coerce_hour(df["HOUR"])
    if "MINUTE" in df.columns:
        df["MINUTE"] = _coerce_minute(df["MINUTE"])

    if "RUR_URB" in df.columns:
        df["RUR_URB"] = pd.to_numeric(df["RUR_URB"], errors="coerce").astype("Int64")

    if "LGT_COND" in df.columns:
        df["LGT_COND"] = pd.to_numeric(df["LGT_COND"], errors="coerce").astype("Int64")

    if "FUNC_SYS" in df.columns:
        df["FUNC_SYS"] = pd.to_numeric(df["FUNC_SYS"], errors="coerce").astype("Int64")

    return df


def load_person(year: int, force: bool = False) -> pd.DataFrame:
    """Load person.csv for *year* from its FARS zip.

    Returns a DataFrame with: ST_CASE, PER_TYP, INJ_SEV, AGE, SEX.

    Args:
        year: Four-digit year.
        force: Re-download the zip if it is missing.

    Returns:
        pd.DataFrame with one row per person record.
    """
    zip_path = config.DATA_RAW / f"FARS{year}NationalCSV.zip"
    if zip_path.exists() and not force:
        pass  # already logged by load_accident for this year
    else:
        zip_path = download_fars_year(year, force=force)

    with zipfile.ZipFile(zip_path, "r") as zf:
        member = _find_file_in_zip(zf, "person")
        if member is None:
            raise FileNotFoundError(
                f"person.csv not found in {zip_path}. Members: {zf.namelist()}"
            )
        with zf.open(member) as fh:
            df = _read_csv_safe(fh, dtype=str)

    df.columns = [c.strip().upper() for c in df.columns]

    desired = ["ST_CASE", "PER_NO", "VEH_NO", "PER_TYP", "INJ_SEV", "AGE", "SEX"]
    present = [c for c in desired if c in df.columns]
    df = df[present].copy()

    for col in ["ST_CASE", "PER_NO", "VEH_NO", "PER_TYP", "INJ_SEV", "AGE", "SEX"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df


def load_pedestrian_fatalities(
    years: Iterable[int],
    force: bool = False,
) -> pd.DataFrame:
    """Build a cleaned per-fatality DataFrame for pedestrian fatalities.

    Filters person records to PER_TYP == 5 (pedestrian) and INJ_SEV == 4
    (fatal), merges with accident data, and drops rows missing lat/long or time.
    Caches the result to data/processed/ped_fatalities.parquet.

    Args:
        years: Iterable of four-digit years to load.
        force: If True, re-download zips and re-build the cache even if it exists.

    Returns:
        pd.DataFrame with one row per pedestrian fatality.
    """
    cache_path = config.DATA_PROCESSED / "ped_fatalities.parquet"
    years_list = sorted(set(years))

    if cache_path.exists() and not force:
        cached = pd.read_parquet(cache_path)
        cached_years = set(cached["YEAR"].dropna().astype(int).unique())
        requested_years = set(years_list)
        if requested_years.issubset(cached_years):
            print(f"[fars_loader] Loading cached fatalities from {cache_path}")
            return cached[cached["YEAR"].isin(requested_years)].reset_index(drop=True)
        missing = sorted(requested_years - cached_years)
        print(f"[fars_loader] Cache missing years {missing}, rebuilding...")
    accident_frames: list[pd.DataFrame] = []
    person_frames: list[pd.DataFrame] = []

    for yr in tqdm(years_list, desc="Loading FARS years"):
        try:
            acc = load_accident(yr, force=force)
            per = load_person(yr, force=force)
        except FileNotFoundError as exc:
            print(f"[fars_loader] Warning: skipping {yr} — {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"[fars_loader] Warning: error loading {yr} — {exc}")
            continue

        # Ensure YEAR is present in accident frame
        if "YEAR" not in acc.columns:
            acc["YEAR"] = yr
        per["YEAR"] = yr

        accident_frames.append(acc)
        person_frames.append(per)

    if not accident_frames:
        raise RuntimeError("No FARS data could be loaded for the requested years.")

    accidents = pd.concat(accident_frames, ignore_index=True)
    persons = pd.concat(person_frames, ignore_index=True)

    # Filter to pedestrian fatalities
    peds = persons[
        (persons["PER_TYP"] == 5) & (persons["INJ_SEV"] == 4)
    ].copy()

    # Merge on (ST_CASE, YEAR)
    df = peds.merge(accidents, on=["ST_CASE", "YEAR"], how="inner")

    # Drop rows without usable location or time
    df = df.dropna(subset=["LATITUDE", "LONGITUD", "HOUR", "MONTH", "DAY"])

    # Keep HOUR as integer for downstream use; local-time conversion is in lighting.py
    df = df.reset_index(drop=True)

    df.to_parquet(cache_path, index=False)
    print(f"[fars_loader] Saved {len(df):,} pedestrian fatalities → {cache_path}")

    return df


def load_vehicle_speed_limits(
    years: Iterable[int],
    force: bool = False,
) -> pd.DataFrame:
    """Load posted speed limits from FARS vehicle.csv across *years*.

    For each crash, takes the maximum VSPD_LIM across all vehicles — a
    conservative choice representing the highest posted speed at the scene.
    This is the appropriate aggregate for road-level exposure analysis.

    Sentinel values 98 (not reported) and 99 (unknown) are coerced to NaN.
    Pre-2010 FARS files may lack VSPD_LIM entirely; those years are omitted
    from the result rather than appearing with NaN.

    Note: TRAV_SP (travel speed) is a different variable and is not used here.
    VSPD_LIM is the posted speed limit. Some very early FARS years lack this
    column in vehicle.csv; if absent the year is skipped silently.

    Caches to data/processed/speed_limits.parquet.

    Args:
        years: Iterable of four-digit years to load.
        force: Re-download / re-build even if cache exists.

    Returns:
        pd.DataFrame with columns: ST_CASE, YEAR, VSPD_LIM.
    """
    cache_path = config.DATA_PROCESSED / "speed_limits.parquet"
    years_list = sorted(set(years))

    if cache_path.exists() and not force:
        cached = pd.read_parquet(cache_path)
        cached_years = set(cached["YEAR"].dropna().astype(int).unique())
        if set(years_list).issubset(cached_years):
            print(f"[fars_loader] Loaded {len(cached):,} speed limit records from cache")
            return cached[cached["YEAR"].isin(years_list)].reset_index(drop=True)
        missing = sorted(set(years_list) - cached_years)
        print(f"[fars_loader] Cache missing years {missing}, rebuilding speed limits...")

    frames: list[pd.DataFrame] = []
    for yr in tqdm(years_list, desc="Loading speed limits"):
        zip_path = config.DATA_RAW / f"FARS{yr}NationalCSV.zip"
        if not zip_path.exists():
            continue
        try:
            with zipfile.ZipFile(zip_path) as zf:
                mem = _find_file_in_zip(zf, "vehicle")
                if mem is None:
                    continue
                with zf.open(mem) as fh:
                    df_v = _read_csv_safe(fh)
            df_v.columns = [c.strip().upper() for c in df_v.columns]
            if "VSPD_LIM" not in df_v.columns:
                continue
            df_v = df_v[["ST_CASE", "VSPD_LIM"]].copy()
            df_v["ST_CASE"] = pd.to_numeric(df_v["ST_CASE"], errors="coerce")
            df_v["VSPD_LIM"] = pd.to_numeric(df_v["VSPD_LIM"], errors="coerce")
            # Coerce sentinels: 98 = not reported, 99 = unknown → NaN
            df_v["VSPD_LIM"] = df_v["VSPD_LIM"].where(
                df_v["VSPD_LIM"].between(1, 97), other=float("nan")
            )
            # Take the maximum speed limit across vehicles at each crash
            df_v = df_v.groupby("ST_CASE")["VSPD_LIM"].max().reset_index()
            df_v["YEAR"] = yr
            frames.append(df_v)
        except Exception as exc:  # noqa: BLE001
            print(f"[fars_loader] Warning: speed limits for {yr} — {exc}")

    if not frames:
        print("[fars_loader] No speed limit data found in any year.")
        return pd.DataFrame(columns=["ST_CASE", "VSPD_LIM", "YEAR"])

    result = pd.concat(frames, ignore_index=True)
    result.to_parquet(cache_path, index=False)
    print(f"[fars_loader] Saved {len(result):,} speed limit records → {cache_path}")
    return result


def load_pedestrian_fatalities_with_context(
    years: Iterable[int],
    force: bool = False,
) -> pd.DataFrame:
    """Build a per-fatality DataFrame enriched with lighting and speed context.

    Extends the base pedestrian fatality data with:

    - **LGT_COND**: FARS officer-reported lighting condition (accident.csv).
      Code values:
        1 = Daylight
        2 = Dark — Not Lighted
        3 = Dark — Lighted
        4 = Dawn
        5 = Dusk
        6 = Dark — Unknown Lighting
        7+ = Other / Unknown / Not Reported

    - **FUNC_SYS**: FARS functional road class (accident.csv).
      Rural: 1=Interstate, 2=Principal Arterial Other, 3=Minor Arterial,
             4=Major Collector, 5=Minor Collector, 6=Local, 7=Unknown.
      Urban: 11=Interstate/Freeway, 12=Freeway/Expressway,
             13=Principal Arterial, 14=Minor Arterial, 15=Collector,
             16=Local, 17=Unknown. 96-99=Unknown/Not Reported.

    - **VSPD_LIM**: posted speed limit in mph from VEH_NO == 1 in vehicle.csv.
      VEH_NO == 1 is the striking vehicle in the vast majority of pedestrian
      crashes (>95% are single-vehicle; in multi-vehicle crashes the at-fault
      vehicle is typically coded first). Sentinel values 98 and 99 are NaN.
      Pre-2010 VSPD_LIM coverage is sparse; expect high NaN rates before 2010.

    Caches to data/processed/ped_fatalities_with_context.parquet.

    Args:
        years: Iterable of four-digit years to load.
        force: Re-download / re-build even if cache exists.

    Returns:
        pd.DataFrame with one row per pedestrian fatality, enriched columns.
    """
    cache_path = config.DATA_PROCESSED / "ped_fatalities_with_context.parquet"
    years_list = sorted(set(years))

    if cache_path.exists() and not force:
        cached = pd.read_parquet(cache_path)
        cached_years = set(cached["YEAR"].dropna().astype(int).unique())
        if set(years_list).issubset(cached_years):
            print(f"[fars_loader] Loading cached context fatalities from {cache_path}")
            return cached[cached["YEAR"].isin(years_list)].reset_index(drop=True)
        missing = sorted(set(years_list) - cached_years)
        print(f"[fars_loader] Cache missing years {missing}, rebuilding...")

    accident_frames: list[pd.DataFrame] = []
    person_frames: list[pd.DataFrame] = []
    vehicle_frames: list[pd.DataFrame] = []

    for yr in tqdm(years_list, desc="Loading FARS years (with context)"):
        try:
            acc = load_accident(yr, force=force)
            per = load_person(yr, force=force)
        except FileNotFoundError as exc:
            print(f"[fars_loader] Warning: skipping {yr} — {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"[fars_loader] Warning: error loading {yr} — {exc}")
            continue

        if "YEAR" not in acc.columns:
            acc["YEAR"] = yr
        per["YEAR"] = yr

        accident_frames.append(acc)
        person_frames.append(per)

        # Load striking vehicle (VEH_NO == 1) for posted speed limit.
        # VEH_NO == 1 is the first-coded vehicle in FARS — the striking vehicle
        # in single-vehicle pedestrian crashes and typically the at-fault vehicle
        # in multi-vehicle crashes.
        zip_path = config.DATA_RAW / f"FARS{yr}NationalCSV.zip"
        if zip_path.exists():
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    mem = _find_file_in_zip(zf, "vehicle")
                    if mem is not None:
                        with zf.open(mem) as fh:
                            df_v = _read_csv_safe(fh)
                df_v.columns = [c.strip().upper() for c in df_v.columns]
                if "VSPD_LIM" in df_v.columns and "VEH_NO" in df_v.columns:
                    df_v = df_v[["ST_CASE", "VEH_NO", "VSPD_LIM"]].copy()
                    df_v["ST_CASE"] = pd.to_numeric(df_v["ST_CASE"], errors="coerce")
                    df_v["VEH_NO"] = pd.to_numeric(df_v["VEH_NO"], errors="coerce")
                    df_v["VSPD_LIM"] = pd.to_numeric(df_v["VSPD_LIM"], errors="coerce")
                    # Coerce sentinels 98/99 → NaN
                    df_v["VSPD_LIM"] = df_v["VSPD_LIM"].where(
                        df_v["VSPD_LIM"].between(1, 97), other=float("nan")
                    )
                    df_v = df_v[df_v["VEH_NO"] == 1][["ST_CASE", "VSPD_LIM"]].copy()
                    df_v["YEAR"] = yr
                    vehicle_frames.append(df_v)
            except Exception as exc:  # noqa: BLE001
                print(f"[fars_loader] Warning: vehicle load for {yr} — {exc}")

    if not accident_frames:
        raise RuntimeError("No FARS data could be loaded for the requested years.")

    accidents = pd.concat(accident_frames, ignore_index=True)
    persons = pd.concat(person_frames, ignore_index=True)

    # Filter to pedestrian fatalities
    peds = persons[
        (persons["PER_TYP"] == 5) & (persons["INJ_SEV"] == 4)
    ].copy()

    # Merge accident data (provides LGT_COND, FUNC_SYS, RUR_URB, location, time)
    df = peds.merge(accidents, on=["ST_CASE", "YEAR"], how="inner")

    # Drop rows without usable location or time
    df = df.dropna(subset=["LATITUDE", "LONGITUD", "HOUR", "MONTH", "DAY"])

    # Merge striking-vehicle speed limit on (ST_CASE, YEAR)
    if vehicle_frames:
        vehicles = pd.concat(vehicle_frames, ignore_index=True)
        df = df.merge(vehicles, on=["ST_CASE", "YEAR"], how="left")
    else:
        df["VSPD_LIM"] = float("nan")

    df = df.reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    print(
        f"[fars_loader] Saved {len(df):,} pedestrian fatalities with context "
        f"→ {cache_path}"
    )
    return df
