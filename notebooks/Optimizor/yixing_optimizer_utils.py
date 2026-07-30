"""Small shared helpers for the Yixing optimizer workflow."""

from __future__ import annotations

from pathlib import Path
import math
from typing import Iterable

import h3
import numpy as np
import pandas as pd

from config import H3_RESOLUTION, OUTPUT_DIR


def ensure_output_dir(output_dir: Path = OUTPUT_DIR) -> Path:
    """Create and return the Yixing output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def read_csv_flexible(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV that may be encoded as UTF-8 with BOM or GB18030."""

    if not path.exists():
        raise FileNotFoundError(f"Missing expected input CSV: {path}")
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gb18030", **kwargs)


def add_h3_index(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    h3_col: str = "h3_l7",
    resolution: int = H3_RESOLUTION,
) -> pd.DataFrame:
    """Attach an H3 index using h3 3.x/4.x compatible APIs."""

    out = df.copy()
    if hasattr(h3, "geo_to_h3"):
        out[h3_col] = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in zip(out[lat_col], out[lon_col])]
    else:
        out[h3_col] = [h3.latlng_to_cell(lat, lon, resolution) for lat, lon in zip(out[lat_col], out[lon_col])]
    return out


def h3_center_latlon(h3_id: str) -> tuple[float, float]:
    """Return an H3 center as (lat, lon)."""

    if hasattr(h3, "h3_to_geo"):
        lat, lon = h3.h3_to_geo(h3_id)
    else:
        lat, lon = h3.cell_to_latlng(h3_id)
    return float(lat), float(lon)


def h3_boundary_lnglat(h3_id: str) -> list[tuple[float, float]]:
    """Return H3 boundary coordinates as (lon, lat)."""

    if hasattr(h3, "h3_to_geo_boundary"):
        try:
            return list(h3.h3_to_geo_boundary(h3_id, geo_json=True))
        except TypeError:
            return [(lon, lat) for lat, lon in h3.h3_to_geo_boundary(h3_id)]
    return [(lon, lat) for lat, lon in h3.cell_to_boundary(h3_id)]


def intersection_area(r1: float, r2: float, d: float) -> float:
    """Area of intersection between two circles."""

    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    term1 = r1**2 * math.acos((d**2 + r1**2 - r2**2) / (2 * d * r1))
    term2 = r2**2 * math.acos((d**2 + r2**2 - r1**2) / (2 * d * r2))
    term3 = 0.5 * math.sqrt(max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)))
    return term1 + term2 - term3


def haversine_array(lat1: float, lon1: float, lat2_array: Iterable[float], lon2_array: Iterable[float]) -> np.ndarray:
    """Vectorized Haversine distance in kilometers."""

    lat1_rad, lon1_rad = map(math.radians, [lat1, lon1])
    lat2 = np.radians(np.asarray(lat2_array, dtype=float))
    lon2 = np.radians(np.asarray(lon2_array, dtype=float))
    dlat = lat2 - lat1_rad
    dlon = lon2 - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371.0 * c


def cache_h3_centers(df: pd.DataFrame, h3_col: str = "h3_l7") -> pd.DataFrame:
    """Attach H3 center latitude/longitude columns."""

    out = df.copy()
    unique_h3 = out[h3_col].dropna().astype(str).unique()
    centers = {hid: h3_center_latlon(hid) for hid in unique_h3}
    out["center_lat"] = out[h3_col].astype(str).map(lambda hid: centers[hid][0])
    out["center_lon"] = out[h3_col].astype(str).map(lambda hid: centers[hid][1])
    return out
