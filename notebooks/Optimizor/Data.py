"""Yixing optimizer data loader.

This mirrors the DSS ``Data`` role: read scored candidate locations, expose
arrays used by the optimization model, and build or load distance-conflict
indicators.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from config import OPTIMIZATION_DISTANCE_THRESHOLD_M, OUTPUT_DIR, PROJECTED_CRS, WGS84_CRS
from yixing_optimizer_utils import read_csv_flexible


def format_distance_threshold_m(threshold_m: float) -> str:
    """Return a stable filename token for a distance threshold in meters."""

    value = float(threshold_m)
    if value.is_integer():
        return f"{int(value)}m"
    return f"{value:g}".replace(".", "_") + "m"


def default_indicator_cache_path(
    *,
    input_path: Path | None = None,
    output_dir: Path | None = None,
    threshold_m: float = OPTIMIZATION_DISTANCE_THRESHOLD_M,
) -> Path:
    """Return the default dense indicator cache path for a threshold."""

    if output_dir is None and input_path is not None:
        output_dir = Path(input_path).parent
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    return output_dir / f"indicator_i_j_{format_distance_threshold_m(threshold_m)}.npy"


def indicator_metadata_path(indicator_path: Path) -> Path:
    """Return the JSON metadata path paired with an indicator npy file."""

    return Path(indicator_path).with_suffix(".json")


def candidate_id_hash(candidate_df: pd.DataFrame) -> str:
    """Hash candidate row identity in the order used by optimization."""

    if "candidate_id" in candidate_df.columns:
        identity = candidate_df["candidate_id"].fillna("").astype(str)
    elif "id" in candidate_df.columns:
        identity = candidate_df["id"].fillna("").astype(str)
    else:
        lat_col, lon_col = Data._lat_lon_columns(candidate_df)
        identity = candidate_df[[lat_col, lon_col]].astype(str).agg("|".join, axis=1)
    digest = hashlib.sha256()
    for value in identity:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_distance_conflict_pairs(
    candidate_df: pd.DataFrame,
    *,
    threshold_m: float = OPTIMIZATION_DISTANCE_THRESHOLD_M,
    projected_crs: str = PROJECTED_CRS,
) -> list[tuple[int, int]]:
    """Return index pairs whose projected distance is within threshold_m."""

    if candidate_df.empty:
        return []

    import geopandas as gpd
    from scipy.spatial import cKDTree

    geometry = gpd.points_from_xy(candidate_df["longitude"], candidate_df["latitude"])
    gdf = gpd.GeoDataFrame(candidate_df.copy(), geometry=geometry, crs=WGS84_CRS).to_crs(projected_crs)
    coords = np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])
    pairs = sorted((int(i), int(j)) for i, j in cKDTree(coords).query_pairs(float(threshold_m)))
    return pairs


def build_indicator_matrix(
    candidate_df: pd.DataFrame,
    *,
    threshold_m: float = OPTIMIZATION_DISTANCE_THRESHOLD_M,
    projected_crs: str = PROJECTED_CRS,
) -> np.ndarray:
    """Build a DSS-style dense bool indicator matrix from candidate coordinates."""

    pairs = build_distance_conflict_pairs(candidate_df, threshold_m=threshold_m, projected_crs=projected_crs)
    indicator = np.zeros((len(candidate_df), len(candidate_df)), dtype=bool)
    if pairs:
        pair_array = np.asarray(pairs, dtype=np.int64)
        indicator[pair_array[:, 0], pair_array[:, 1]] = True
        indicator[pair_array[:, 1], pair_array[:, 0]] = True
    np.fill_diagonal(indicator, False)
    return indicator


def indicator_matrix_to_pairs(indicator: np.ndarray) -> list[tuple[int, int]]:
    """Convert a dense indicator matrix to upper-triangle conflict pairs."""

    rows, cols = np.where(np.triu(np.asarray(indicator, dtype=bool), k=1))
    return list(zip(rows.astype(int).tolist(), cols.astype(int).tolist()))


def write_indicator_cache(
    candidate_df: pd.DataFrame,
    *,
    indicator_path: Path,
    threshold_m: float = OPTIMIZATION_DISTANCE_THRESHOLD_M,
    input_path: Path | None = None,
    projected_crs: str = PROJECTED_CRS,
) -> tuple[Path, Path, np.ndarray, dict[str, object]]:
    """Write a dense indicator matrix and its metadata."""

    from datetime import datetime, timezone

    indicator_path = Path(indicator_path)
    indicator_path.parent.mkdir(parents=True, exist_ok=True)
    source_df = candidate_df.reset_index(drop=True).copy()
    indicator = build_indicator_matrix(source_df, threshold_m=threshold_m, projected_crs=projected_crs)
    np.save(indicator_path, indicator)

    metadata = {
        "input_path": str(Path(input_path).resolve()) if input_path is not None else None,
        "candidate_count": int(len(source_df)),
        "distance_threshold_m": float(threshold_m),
        "candidate_id_hash": candidate_id_hash(source_df),
        "matrix_shape": [int(indicator.shape[0]), int(indicator.shape[1])],
        "dtype": str(indicator.dtype),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = indicator_metadata_path(indicator_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return indicator_path, metadata_path, indicator, metadata


def load_valid_indicator_cache(
    candidate_df: pd.DataFrame,
    *,
    indicator_path: Path,
    threshold_m: float = OPTIMIZATION_DISTANCE_THRESHOLD_M,
    input_path: Path | None = None,
) -> np.ndarray | None:
    """Load an indicator cache only when metadata matches the candidate data."""

    indicator_path = Path(indicator_path)
    metadata_path = indicator_metadata_path(indicator_path)
    if not indicator_path.exists() or not metadata_path.exists():
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    source_df = candidate_df.reset_index(drop=True).copy()
    expected_count = int(len(source_df))
    if int(metadata.get("candidate_count", -1)) != expected_count:
        return None
    if abs(float(metadata.get("distance_threshold_m", -1.0)) - float(threshold_m)) > 1e-9:
        return None
    if input_path is not None:
        metadata_input_path = metadata.get("input_path")
        if metadata_input_path is None:
            return None
        try:
            if Path(metadata_input_path).resolve() != Path(input_path).resolve():
                return None
        except OSError:
            return None
    if metadata.get("candidate_id_hash") != candidate_id_hash(source_df):
        return None

    indicator = np.load(indicator_path)
    if indicator.shape != (expected_count, expected_count) or indicator.dtype != np.bool_:
        return None
    return indicator


class Data(object):
    def __init__(self):
        self.loc_lat = None
        self.loc_lon = None
        self.loc_score = None
        self.loc_score_mlp = None

        self.dist_i_j = {}
        self.indicator_i_j = {}
        self.conflict_pairs = []
        self.indicator_cache_path = None
        self.indicator_cache_used = False
        self.infinite = 0

        self.build_num = 0
        self.loc_num = 0
        self.dist_limit = 0.0
        self.dist_limit_m = 0.0

        self.candidate_df = pd.DataFrame()

    def read_data(
        self,
        file_name,
        loc_num=5,
        dist_limit_m=OPTIMIZATION_DISTANCE_THRESHOLD_M,
        build_num=None,
        score_col="total_score",
        mlp_score_col="total_score_mlp",
        indicator_path=None,
        use_indicator_cache=True,
    ):
        input_path = Path(file_name)
        df = read_csv_flexible(input_path)
        if indicator_path is None and use_indicator_cache:
            indicator_path = default_indicator_cache_path(input_path=input_path, threshold_m=dist_limit_m)
        return self.read_dataframe(
            df,
            loc_num=loc_num,
            dist_limit_m=dist_limit_m,
            build_num=build_num,
            score_col=score_col,
            mlp_score_col=mlp_score_col,
            indicator_path=indicator_path,
            use_indicator_cache=use_indicator_cache,
            input_path=input_path,
        )

    def read_dataframe(
        self,
        candidate_df,
        loc_num=5,
        dist_limit_m=OPTIMIZATION_DISTANCE_THRESHOLD_M,
        build_num=None,
        score_col="total_score",
        mlp_score_col="total_score_mlp",
        indicator_path=None,
        use_indicator_cache=True,
        input_path=None,
    ):
        full_df = candidate_df.reset_index(drop=True).copy()
        cached_indicator = None
        self.indicator_cache_path = Path(indicator_path) if indicator_path is not None else None
        self.indicator_cache_used = False
        if use_indicator_cache and self.indicator_cache_path is not None:
            cached_indicator = load_valid_indicator_cache(
                full_df,
                indicator_path=self.indicator_cache_path,
                threshold_m=dist_limit_m,
                input_path=input_path,
            )

        df = full_df
        if build_num is not None:
            df = df.head(int(build_num)).copy()

        lat_col, lon_col = self._lat_lon_columns(df)
        if score_col not in df.columns:
            raise KeyError(f"Missing score column for optimization: {score_col}")

        self.candidate_df = df
        self.loc_lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=float)
        self.loc_lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=float)
        self.loc_score = pd.to_numeric(df[score_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        if mlp_score_col in df.columns:
            self.loc_score_mlp = pd.to_numeric(df[mlp_score_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        else:
            self.loc_score_mlp = None

        self.loc_num = int(loc_num)
        self.build_num = len(df)
        self.dist_limit_m = float(dist_limit_m)
        self.dist_limit = self.dist_limit_m / 1000.0
        self._build_distance_indicator(cached_indicator=cached_indicator)
        return self

    def haversine(self, lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        d_lat = lat2 - lat1
        d_lon = lon2 - lon1
        a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return c * 6371.0

    def has_distance_conflict(self, i, j):
        if self.infinite != 0:
            return False
        return bool(self.indicator_i_j.get((i, j), self.indicator_i_j.get((j, i), 0)))

    def _build_distance_indicator(self, cached_indicator=None):
        self.dist_i_j = {}
        self.indicator_i_j = {}
        self.conflict_pairs = []

        if self.dist_limit_m <= 0 or self.candidate_df.empty:
            self.infinite = 1
            return

        self.infinite = 0
        if cached_indicator is not None:
            indicator_slice = np.array(cached_indicator[: self.build_num, : self.build_num], dtype=bool, copy=True)
            np.fill_diagonal(indicator_slice, False)
            self.conflict_pairs = indicator_matrix_to_pairs(indicator_slice)
            self.indicator_cache_used = True
        else:
            self.conflict_pairs = build_distance_conflict_pairs(self.candidate_df, threshold_m=self.dist_limit_m)
            self.indicator_cache_used = False

        for i, j in self.conflict_pairs:
            distance_km = self.haversine(self.loc_lat[i], self.loc_lon[i], self.loc_lat[j], self.loc_lon[j])
            self.dist_i_j[i, j] = distance_km
            self.dist_i_j[j, i] = distance_km
            self.indicator_i_j[i, j] = 1
            self.indicator_i_j[j, i] = 1

    @staticmethod
    def _lat_lon_columns(df):
        if "latitude" in df.columns and "longitude" in df.columns:
            return "latitude", "longitude"
        if "lat" in df.columns and "lon" in df.columns:
            return "lat", "lon"
        raise KeyError("Expected candidate latitude/longitude columns were not found.")