"""
lighting.py — Classify crash time as daylight, civil_twilight, or darkness.

The key operation: given a lat/lon and a local datetime, compare the time
against sunrise/sunset/dawn/dusk from astral. Caches sun events per
(rounded location, date) to avoid recomputing for every row in a city.
"""

from __future__ import annotations

from datetime import datetime, date, timedelta
from typing import Literal

import pandas as pd
import pytz
from astral import LocationInfo
from astral.sun import sun
from tqdm import tqdm

from . import config
from .geo import get_timezone

LightingCategory = Literal["daylight", "civil_twilight", "darkness"]

# Precision for location rounding when building the per-(location, date) cache.
# 1 decimal degree ≈ 111 km; sufficient for sun-position purposes.
_ROUND_DEGREES = 1


def _get_sun_events(lat: float, lon: float, d: date, tz_name: str) -> dict:
    """Compute dawn, sunrise, sunset, dusk for a given location and date.

    Returns a dict with keys: dawn, sunrise, sunset, dusk — all as
    timezone-aware datetimes. On error (polar day/night, astral raise),
    returns None so the caller can fall back gracefully.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        d: Calendar date.
        tz_name: IANA timezone string for the location.

    Returns:
        Dict of sun events or None on failure.
    """
    try:
        loc = LocationInfo(latitude=lat, longitude=lon)
        tz = pytz.timezone(tz_name)
        s = sun(loc.observer, date=d, tzinfo=tz)
        return s
    except Exception:  # noqa: BLE001 — astral raises ValueError for polar extremes
        return None


def classify_lighting(lat: float, lon: float, local_dt: datetime) -> LightingCategory:
    """Classify a crash time as 'daylight', 'civil_twilight', or 'darkness'.

    Localizes the naive *local_dt* to the timezone inferred from *lat*/*lon*,
    then compares against dawn/sunrise/sunset/dusk for that day.

    Edge cases:
    - astral raising (polar night/day): returns 'darkness' or 'daylight'
      based on month heuristic (Jun–Aug → 'daylight', else 'darkness').
    - timezonefinder ocean miss: falls back to UTC.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        local_dt: Naive datetime representing local wall-clock time.

    Returns:
        One of 'daylight', 'civil_twilight', 'darkness'.
    """
    assert local_dt.tzinfo is None, (
        "classify_lighting expects a naive local datetime; got tz-aware input."
    )

    tz_name = get_timezone(lat, lon)
    tz = pytz.timezone(tz_name)
    aware_dt = tz.localize(local_dt, is_dst=None)

    # Use the *local* date (naive) to anchor sun events
    local_date = local_dt.date()
    events = _get_sun_events(lat, lon, local_date, tz_name)

    if events is None:
        # Polar fallback: summer months → daylight, otherwise darkness
        month = local_dt.month
        return "daylight" if 5 <= month <= 8 else "darkness"

    dawn = events.get("dawn")
    dusk = events.get("dusk")
    sunrise = events.get("sunrise")
    sunset = events.get("sunset")

    # Prefer civil twilight boundaries (dawn/dusk) if available
    if dawn is not None and dusk is not None:
        if aware_dt < dawn or aware_dt > dusk:
            return "darkness"
        elif aware_dt < sunrise or aware_dt > sunset:
            return "civil_twilight"
        else:
            return "daylight"
    elif sunrise is not None and sunset is not None:
        # Fall back to sunrise/sunset only
        if aware_dt < sunrise or aware_dt > sunset:
            return "darkness"
        else:
            return "daylight"
    else:
        return "darkness"


def classify_lighting_batch(
    df: pd.DataFrame,
    lat_col: str = "LATITUDE",
    lon_col: str = "LONGITUD",
    datetime_col: str | None = None,
    year_col: str = "YEAR",
    month_col: str = "MONTH",
    day_col: str = "DAY",
    hour_col: str = "HOUR",
    minute_col: str = "MINUTE",
    force: bool = False,
) -> pd.Series:
    """Classify lighting for every row in *df*.

    Groups rows by (rounded_lat, rounded_lon, date) to compute sun events
    once per unique (location, date) combination rather than per row — this
    reduces the number of astral calls by ~100x for typical FARS data.

    Requires either a pre-built *datetime_col* (naive local datetimes) or
    the five component columns YEAR/MONTH/DAY/HOUR/MINUTE.

    Caches results to data/processed/lighting.parquet keyed by ST_CASE.
    Loads the cached file on subsequent calls unless *force* is True.

    Args:
        df: DataFrame with FARS fatality records.
        lat_col: Column name for latitude.
        lon_col: Column name for longitude.
        datetime_col: Optional column of pre-built naive local datetimes.
        year_col, month_col, day_col, hour_col, minute_col: Component cols.
        force: If True, recompute even if cache exists.

    Returns:
        pd.Series of LightingCategory strings, indexed to match *df*.
    """
    cache_path = config.DATA_PROCESSED / "lighting.parquet"
    _UID = ["ST_CASE", "YEAR", "PER_NO"]
    _existing_cache: pd.DataFrame | None = None
    if cache_path.exists() and not force:
        print(f"[lighting] Loading cached lighting from {cache_path}")
        _raw_cache = pd.read_parquet(cache_path)
        if not all(c in _raw_cache.columns for c in _UID):
            print("[lighting] Cache format outdated — reclassifying all rows.")
        else:
            _existing_cache = _raw_cache
            cached_lookup = {
                (r.ST_CASE, r.YEAR, r.PER_NO): r.lighting
                for r in _existing_cache.itertuples(index=False)
            }
            df_keys = list(zip(df["ST_CASE"], df["YEAR"], df["PER_NO"]))
            if all(k in cached_lookup for k in df_keys):
                return pd.array([cached_lookup[k] for k in df_keys], dtype=object)
            n_missing = sum(1 for k in df_keys if k not in cached_lookup)
            print(f"[lighting] Cache missing {n_missing:,} rows — classifying incrementally...")
            in_cache = pd.Series([k in cached_lookup for k in df_keys], index=df.index)
            df_to_classify = df[~in_cache].reset_index(drop=True)
    if _existing_cache is None:
        df_to_classify = df.reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Build local datetimes
    # -----------------------------------------------------------------------
    if datetime_col is not None:
        local_dts = df_to_classify[datetime_col]
    else:
        required = [year_col, month_col, day_col, hour_col, minute_col]
        missing_cols = [c for c in required if c not in df_to_classify.columns]
        if missing_cols:
            raise ValueError(f"classify_lighting_batch: missing columns {missing_cols}")

        def _safe_dt(row: pd.Series) -> datetime | None:
            try:
                return datetime(
                    int(row[year_col]),
                    int(row[month_col]),
                    int(row[day_col]),
                    int(row[hour_col]),
                    int(row[minute_col]) if pd.notna(row[minute_col]) else 0,
                )
            except Exception:  # noqa: BLE001
                return None

        local_dts = df_to_classify.apply(_safe_dt, axis=1)

    # -----------------------------------------------------------------------
    # Assert no HOUR == 99 (should be filtered upstream)
    # -----------------------------------------------------------------------
    bad_hours = df_to_classify[df_to_classify[hour_col] == 99]
    if len(bad_hours) > 0:
        raise AssertionError(
            f"classify_lighting_batch received {len(bad_hours)} rows with HOUR=99. "
            "These should be filtered before calling this function."
        )

    # -----------------------------------------------------------------------
    # Build per-(rounded_lat, rounded_lon, date) sun-event cache
    # -----------------------------------------------------------------------
    r = _ROUND_DEGREES
    rounded_lat = df_to_classify[lat_col].round(r)
    rounded_lon = df_to_classify[lon_col].round(r)
    dates = local_dts.apply(lambda dt: dt.date() if dt is not None else None)

    # Unique keys (as a set of tuples)
    keys = set(zip(rounded_lat, rounded_lon, dates))
    keys = {k for k in keys if None not in k and not any(
        isinstance(v, float) and pd.isna(v) for v in k
    )}

    print(f"[lighting] Computing sun events for {len(keys):,} unique (location, date) keys...")
    sun_cache: dict[tuple, dict | None] = {}
    for rlat, rlon, d in tqdm(keys, desc="Sun events"):
        tz_name = get_timezone(float(rlat), float(rlon))
        sun_cache[(rlat, rlon, d)] = _get_sun_events(float(rlat), float(rlon), d, tz_name)

    # -----------------------------------------------------------------------
    # Classify each row
    # -----------------------------------------------------------------------
    results: list[LightingCategory | None] = []
    for i in tqdm(range(len(df_to_classify)), desc="Classifying rows"):
        row = df_to_classify.iloc[i]
        dt = local_dts.iloc[i]
        lat = row[lat_col]
        lon = row[lon_col]

        if dt is None or pd.isna(lat) or pd.isna(lon):
            results.append(None)
            continue

        rlat = round(lat, r)
        rlon = round(lon, r)
        d = dt.date()
        events = sun_cache.get((rlat, rlon, d))

        if events is None:
            # Polar fallback
            month = dt.month
            results.append("daylight" if 5 <= month <= 8 else "darkness")
            continue

        tz_name = get_timezone(lat, lon)
        tz = pytz.timezone(tz_name)
        try:
            aware_dt = tz.localize(dt, is_dst=None)
        except pytz.exceptions.AmbiguousTimeError:
            aware_dt = tz.localize(dt, is_dst=False)
        except pytz.exceptions.NonExistentTimeError:
            aware_dt = tz.localize(dt, is_dst=True)

        dawn = events.get("dawn")
        dusk = events.get("dusk")
        sunrise = events.get("sunrise")
        sunset = events.get("sunset")

        if dawn is not None and dusk is not None:
            if aware_dt < dawn or aware_dt > dusk:
                results.append("darkness")
            elif aware_dt < sunrise or aware_dt > sunset:
                results.append("civil_twilight")
            else:
                results.append("daylight")
        elif sunrise is not None and sunset is not None:
            if aware_dt < sunrise or aware_dt > sunset:
                results.append("darkness")
            else:
                results.append("daylight")
        else:
            results.append("darkness")

    new_cache_df = pd.DataFrame({
        "ST_CASE": df_to_classify["ST_CASE"].values,
        "YEAR": df_to_classify["YEAR"].values,
        "PER_NO": df_to_classify["PER_NO"].values,
        "lighting": results,
    })

    # Merge with existing cache if doing incremental update
    if _existing_cache is not None:
        full_cache = pd.concat([_existing_cache, new_cache_df], ignore_index=True).drop_duplicates(
            subset=["ST_CASE", "YEAR", "PER_NO"], keep="last"
        )
    else:
        full_cache = new_cache_df

    full_cache.to_parquet(cache_path, index=False)
    print(f"[lighting] Saved lighting classifications → {cache_path}")

    # Return aligned to original df
    full_lookup = {
        (r.ST_CASE, r.YEAR, r.PER_NO): r.lighting
        for r in full_cache.itertuples(index=False)
    }
    df_keys = list(zip(df["ST_CASE"], df["YEAR"], df["PER_NO"]))
    return pd.array([full_lookup.get(k) for k in df_keys], dtype=object)


def minutes_from_sunset_batch(
    df: pd.DataFrame,
    lat_col: str = "LATITUDE",
    lon_col: str = "LONGITUD",
    year_col: str = "YEAR",
    month_col: str = "MONTH",
    day_col: str = "DAY",
    hour_col: str = "HOUR",
    minute_col: str = "MINUTE",
    force: bool = False,
) -> pd.Series:
    """Compute signed minutes from local sunset for each row in *df*.

    Negative → before sunset (daylight / approaching dusk).
    Positive → after sunset (twilight / darkness).
    NaN → sunset unavailable (polar extremes, missing coords/time).

    Uses the same (rounded lat/lon, date) grouping as classify_lighting_batch
    to compute sun events once per unique location-date. Caches results to
    data/processed/mins_from_sunset.parquet keyed by (ST_CASE, YEAR, PER_NO).

    Args:
        df: DataFrame with LATITUDE, LONGITUD, YEAR, MONTH, DAY, HOUR, MINUTE
            and ST_CASE, PER_NO as row keys.
        force: Recompute even if cache exists.

    Returns:
        pd.Series of float (minutes from local sunset), indexed like *df*.
    """
    cache_path = config.DATA_PROCESSED / "mins_from_sunset.parquet"
    _UID = ["ST_CASE", "YEAR", "PER_NO"]

    _existing_cache: pd.DataFrame | None = None
    df_todo: pd.DataFrame

    if cache_path.exists() and not force:
        _raw = pd.read_parquet(cache_path)
        if all(c in _raw.columns for c in _UID + ["mins_from_sunset"]):
            _existing_cache = _raw
            cached_lookup = {
                (r.ST_CASE, r.YEAR, r.PER_NO): r.mins_from_sunset
                for r in _existing_cache.itertuples(index=False)
            }
            df_keys = list(zip(df["ST_CASE"], df["YEAR"], df["PER_NO"]))
            if all(k in cached_lookup for k in df_keys):
                print(f"[sunset] Loaded minutes-from-sunset cache ({len(_existing_cache):,} rows)")
                return pd.Series(
                    [cached_lookup[k] for k in df_keys],
                    index=df.index, dtype=float,
                )
            missing_keys = (
                set(zip(df["ST_CASE"], df["YEAR"], df["PER_NO"])) - set(cached_lookup)
            )
            print(f"[sunset] Cache missing {len(missing_keys):,} rows; computing incrementally...")
            df_todo = df[
                df.apply(lambda r: (r["ST_CASE"], r["YEAR"], r["PER_NO"]) in missing_keys, axis=1)
            ].reset_index(drop=True)
        else:
            df_todo = df.reset_index(drop=True)
    else:
        df_todo = df.reset_index(drop=True)
        _existing_cache = None

    # Build naive local datetimes
    def _safe_dt(row: pd.Series) -> datetime | None:
        try:
            return datetime(
                int(row[year_col]), int(row[month_col]), int(row[day_col]),
                int(row[hour_col]),
                int(row[minute_col]) if pd.notna(row[minute_col]) else 0,
            )
        except Exception:  # noqa: BLE001
            return None

    local_dts = df_todo.apply(_safe_dt, axis=1)

    # Per-(rounded_lat, rounded_lon, date) sun-event cache
    r = _ROUND_DEGREES
    rounded_lat = df_todo[lat_col].round(r)
    rounded_lon = df_todo[lon_col].round(r)
    dates = local_dts.apply(lambda dt: dt.date() if dt is not None else None)

    keys = {
        k for k in zip(rounded_lat, rounded_lon, dates)
        if None not in k and not any(isinstance(v, float) and pd.isna(v) for v in k)
    }

    print(f"[sunset] Computing sun events for {len(keys):,} unique (location, date) keys...")
    sun_cache: dict[tuple, dict | None] = {}
    for rlat, rlon, d in tqdm(keys, desc="Sun events (sunset)"):
        tz_name = get_timezone(float(rlat), float(rlon))
        sun_cache[(rlat, rlon, d)] = _get_sun_events(float(rlat), float(rlon), d, tz_name)

    # Compute minutes from sunset per row
    results: list[float | None] = []
    for i in tqdm(range(len(df_todo)), desc="Minutes from sunset"):
        row = df_todo.iloc[i]
        dt = local_dts.iloc[i]
        lat = row[lat_col]
        lon = row[lon_col]

        if dt is None or pd.isna(lat) or pd.isna(lon):
            results.append(None)
            continue

        rlat = round(lat, r)
        rlon = round(lon, r)
        events = sun_cache.get((rlat, rlon, dt.date()))

        if events is None or events.get("sunset") is None:
            results.append(None)
            continue

        tz_name = get_timezone(lat, lon)
        tz = pytz.timezone(tz_name)
        try:
            aware_dt = tz.localize(dt, is_dst=None)
        except pytz.exceptions.AmbiguousTimeError:
            aware_dt = tz.localize(dt, is_dst=False)
        except pytz.exceptions.NonExistentTimeError:
            aware_dt = tz.localize(dt, is_dst=True)

        diff_minutes = (aware_dt - events["sunset"]).total_seconds() / 60.0
        results.append(diff_minutes)

    new_rows = pd.DataFrame({
        "ST_CASE":           df_todo["ST_CASE"].values,
        "YEAR":              df_todo["YEAR"].values,
        "PER_NO":            df_todo["PER_NO"].values,
        "mins_from_sunset":  results,
    })

    if _existing_cache is not None and len(_existing_cache) > 0:
        full_cache = pd.concat([_existing_cache, new_rows], ignore_index=True).drop_duplicates(
            subset=_UID, keep="last"
        )
    else:
        full_cache = new_rows

    full_cache.to_parquet(cache_path, index=False)
    print(f"[sunset] Saved minutes-from-sunset → {cache_path} ({len(full_cache):,} rows)")

    full_lookup = {
        (r.ST_CASE, r.YEAR, r.PER_NO): r.mins_from_sunset
        for r in full_cache.itertuples(index=False)
    }
    df_keys = list(zip(df["ST_CASE"], df["YEAR"], df["PER_NO"]))
    return pd.Series(
        [full_lookup.get(k, None) for k in df_keys],
        index=df.index, dtype=float,
    )


def classify_lighting_at_offset(
    lat: float,
    lon: float,
    local_dt: datetime,
    hour_offset: int,
) -> LightingCategory:
    """Classify lighting at the same location but with the clock shifted by *hour_offset*.

    Used to compute a crash's counterfactual lighting under an alternate DST rule.
    The lat/lon stays the same; only the wall-clock time changes.

    A crash's actual lighting is ``classify_lighting(lat, lon, local_dt)``.
    Its counterfactual under +1 hour DST shift is
    ``classify_lighting_at_offset(lat, lon, local_dt, +1)``, and under -1 hour
    (standard-time rule) is ``classify_lighting_at_offset(lat, lon, local_dt, -1)``.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        local_dt: Naive datetime representing the actual wall-clock time of the crash.
        hour_offset: Integer hours to shift the clock (e.g. +1 or -1).

    Returns:
        One of 'daylight', 'civil_twilight', 'darkness'.
    """
    shifted_dt = local_dt + timedelta(hours=hour_offset)
    return classify_lighting(lat, lon, shifted_dt)

