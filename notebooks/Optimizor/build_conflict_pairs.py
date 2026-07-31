"""Build cached DSS-style distance indicators for Yixing optimization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import OPTIMIZATION_DISTANCE_THRESHOLD_M, OUTPUT_DIR, PROJECTED_CRS, TOTAL_SCORE_PATH, WGS84_CRS
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
    return output_dir / "cache" / f"indicator_i_j_{format_distance_threshold_m(threshold_m)}.npy"


def default_conflict_pairs_path(
    *,
    input_path: Path | None = None,
    output_dir: Path | None = None,
    threshold_m: float = OPTIMIZATION_DISTANCE_THRESHOLD_M,
) -> Path:
    """Return the default sparse conflict-pair cache path for a threshold."""

    if output_dir is None and input_path is not None:
        output_dir = Path(input_path).parent
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    return output_dir / "cache" / f"conflict_pairs_{format_distance_threshold_m(threshold_m)}.npy"


def indicator_metadata_path(indicator_path: Path) -> Path:
    """Return the JSON metadata path paired with an indicator npy file."""

    return Path(indicator_path).with_suffix(".json")


def conflict_pairs_metadata_path(conflict_pairs_path: Path) -> Path:
    """Return the JSON metadata path paired with a sparse conflict-pair npy file."""

    return Path(conflict_pairs_path).with_suffix(".json")


def candidate_id_hash(candidate_df: pd.DataFrame) -> str:
    """Hash candidate row identity in the order used by optimization."""

    if "candidate_id" in candidate_df.columns:
        identity = candidate_df["candidate_id"].fillna("").astype(str)
    elif "id" in candidate_df.columns:
        identity = candidate_df["id"].fillna("").astype(str)
    else:
        lat_col, lon_col = _lat_lon_columns(candidate_df)
        identity = candidate_df[[lat_col, lon_col]].astype(str).agg("|".join, axis=1)

    digest = hashlib.sha256()
    for value in identity:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_distance_conflict_pair_array(
    candidate_df: pd.DataFrame,
    *,
    threshold_m: float = OPTIMIZATION_DISTANCE_THRESHOLD_M,
    projected_crs: str = PROJECTED_CRS,
) -> np.ndarray:
    """Return an ``(n_pairs, 2)`` int64 array of nearby candidate index pairs."""

    if candidate_df.empty:
        return np.empty((0, 2), dtype=np.int64)

    import geopandas as gpd
    from scipy.spatial import cKDTree

    lat_col, lon_col = _lat_lon_columns(candidate_df)
    geometry = gpd.points_from_xy(candidate_df[lon_col], candidate_df[lat_col])
    gdf = gpd.GeoDataFrame(candidate_df.copy(), geometry=geometry, crs=WGS84_CRS).to_crs(projected_crs)
    coords = np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])
    pairs = cKDTree(coords).query_pairs(float(threshold_m), output_type="ndarray")
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if pairs.size == 0:
        return pairs

    pairs.sort(axis=1)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order]


def build_distance_conflict_pairs(
    candidate_df: pd.DataFrame,
    *,
    threshold_m: float = OPTIMIZATION_DISTANCE_THRESHOLD_M,
    projected_crs: str = PROJECTED_CRS,
) -> list[tuple[int, int]]:
    """Return index pairs whose projected distance is within threshold_m."""

    pair_array = build_distance_conflict_pair_array(
        candidate_df,
        threshold_m=threshold_m,
        projected_crs=projected_crs,
    )
    return [(int(i), int(j)) for i, j in pair_array]


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

def write_conflict_pair_cache(
    candidate_df: pd.DataFrame,
    *,
    conflict_pairs_path: Path,
    threshold_m: float = OPTIMIZATION_DISTANCE_THRESHOLD_M,
    input_path: Path | None = None,
    projected_crs: str = PROJECTED_CRS,
) -> tuple[Path, Path, np.ndarray, dict[str, object]]:
    """Write a sparse ``(n_pairs, 2)`` conflict-pair cache and metadata."""

    conflict_pairs_path = Path(conflict_pairs_path)
    conflict_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    source_df = candidate_df.reset_index(drop=True).copy()
    pairs = build_distance_conflict_pair_array(source_df, threshold_m=threshold_m, projected_crs=projected_crs)
    np.save(conflict_pairs_path, pairs)

    metadata = {
        "cache_type": "sparse_conflict_pairs",
        "input_path": str(Path(input_path).resolve()) if input_path is not None else None,
        "candidate_count": int(len(source_df)),
        "distance_threshold_m": float(threshold_m),
        "candidate_id_hash": candidate_id_hash(source_df),
        "pair_shape": [int(pairs.shape[0]), int(pairs.shape[1])],
        "dtype": str(pairs.dtype),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = conflict_pairs_metadata_path(conflict_pairs_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return conflict_pairs_path, metadata_path, pairs, metadata


def load_valid_conflict_pair_cache(
    candidate_df: pd.DataFrame,
    *,
    conflict_pairs_path: Path,
    threshold_m: float = OPTIMIZATION_DISTANCE_THRESHOLD_M,
    input_path: Path | None = None,
) -> np.ndarray | None:
    """Load sparse conflict pairs only when metadata matches the current candidates."""

    conflict_pairs_path = Path(conflict_pairs_path)
    metadata_path = conflict_pairs_metadata_path(conflict_pairs_path)
    if not conflict_pairs_path.exists() or not metadata_path.exists():
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    source_df = candidate_df.reset_index(drop=True).copy()
    expected_count = int(len(source_df))
    if metadata.get("cache_type") != "sparse_conflict_pairs":
        return None
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

    pairs = np.asarray(np.load(conflict_pairs_path), dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        return None
    if pairs.size and (pairs.min() < 0 or pairs.max() >= expected_count):
        return None
    return pairs


def load_or_create_conflict_pair_cache(
    candidate_df: pd.DataFrame,
    *,
    source_path: Path,
    conflict_pairs_path: Path,
    threshold_m: float = OPTIMIZATION_DISTANCE_THRESHOLD_M,
    force_rebuild: bool = False,
) -> tuple[np.ndarray, Path]:
    """Load a valid sparse conflict-pair cache or build it once."""

    conflict_pairs_path = Path(conflict_pairs_path)
    metadata_path = conflict_pairs_metadata_path(conflict_pairs_path)
    if conflict_pairs_path.exists() and metadata_path.exists() and not force_rebuild:
        pairs = load_valid_conflict_pair_cache(
            candidate_df,
            conflict_pairs_path=conflict_pairs_path,
            threshold_m=threshold_m,
            input_path=source_path,
        )
        if pairs is None:
            raise ValueError(f"Saved conflict-pair cache does not match current input: {conflict_pairs_path}")
        return pairs, metadata_path

    _, metadata_path, pairs, _ = write_conflict_pair_cache(
        candidate_df,
        conflict_pairs_path=conflict_pairs_path,
        threshold_m=threshold_m,
        input_path=source_path,
    )
    return pairs, metadata_path


def filter_conflict_pairs_for_sample(
    full_conflict_pairs: np.ndarray,
    sample_indices: np.ndarray,
    *,
    candidate_count: int,
) -> list[tuple[int, int]]:
    """Filter full-candidate conflict pairs to local sample row indices."""

    pairs = np.asarray(full_conflict_pairs, dtype=np.int64).reshape(-1, 2)
    sample_indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    if len(sample_indices) == 0 or pairs.size == 0:
        return []
    if sample_indices.min() < 0 or sample_indices.max() >= int(candidate_count):
        raise IndexError("sample_indices contains out-of-range candidate indices.")

    local_index = np.full(int(candidate_count), -1, dtype=np.int64)
    local_index[sample_indices] = np.arange(len(sample_indices), dtype=np.int64)
    local_pairs = local_index[pairs]
    mask = (local_pairs[:, 0] >= 0) & (local_pairs[:, 1] >= 0)
    local_pairs = local_pairs[mask]
    if local_pairs.size == 0:
        return []

    local_pairs.sort(axis=1)
    order = np.lexsort((local_pairs[:, 1], local_pairs[:, 0]))
    local_pairs = local_pairs[order]
    return [(int(i), int(j)) for i, j in local_pairs]


@dataclass(frozen=True)
class ConflictPairBuildResult:
    """Outputs from building a sparse conflict-pair cache."""

    conflict_pairs_path: Path
    metadata_path: Path
    candidate_count: int
    conflict_pair_count: int
    distance_threshold_m: float


@dataclass(frozen=True)
class ConflictIndicatorBuildResult:
    """Outputs from building a dense indicator matrix."""

    indicator_path: Path
    metadata_path: Path
    candidate_count: int
    conflict_pair_count: int
    distance_threshold_m: float


def build_conflict_pair_file(
    *,
    input_path: Path = TOTAL_SCORE_PATH,
    output_path: Path | None = None,
    distance_threshold_m: float = OPTIMIZATION_DISTANCE_THRESHOLD_M,
    build_num: int | None = None,
) -> ConflictPairBuildResult:
    """Build and write a sparse conflict-pair cache from scored candidates."""

    input_path = Path(input_path)
    output_path = default_conflict_pairs_path(input_path=input_path, threshold_m=distance_threshold_m) if output_path is None else Path(output_path)

    with tqdm(total=4, desc="Building conflict pairs", unit="step", dynamic_ncols=True) as progress:
        candidate_df = read_csv_flexible(input_path)
        if build_num is not None:
            candidate_df = candidate_df.head(int(build_num)).copy()
        candidate_df = candidate_df.reset_index(drop=True)
        progress.set_postfix_str(f"loaded {len(candidate_df):,} candidates")
        progress.update()

        conflict_pairs_path, metadata_path, pairs, _ = write_conflict_pair_cache(
            candidate_df,
            conflict_pairs_path=output_path,
            threshold_m=distance_threshold_m,
            input_path=input_path,
        )
        progress.set_postfix_str(f"{len(pairs):,} sparse conflict pairs")
        progress.update()

        progress.set_postfix_str(f"array shape {pairs.shape[0]:,} x {pairs.shape[1] if pairs.ndim == 2 else 0}")
        progress.update()

        progress.set_postfix_str("cache written")
        progress.update()

    return ConflictPairBuildResult(
        conflict_pairs_path=conflict_pairs_path,
        metadata_path=metadata_path,
        candidate_count=len(candidate_df),
        conflict_pair_count=len(pairs),
        distance_threshold_m=float(distance_threshold_m),
    )

def build_conflict_indicator(
    *,
    input_path: Path = TOTAL_SCORE_PATH,
    output_path: Path | None = None,
    distance_threshold_m: float = OPTIMIZATION_DISTANCE_THRESHOLD_M,
    build_num: int | None = None,
) -> ConflictIndicatorBuildResult:
    """Build and write a dense bool indicator matrix from scored candidates."""

    input_path = Path(input_path)
    output_path = default_indicator_cache_path(input_path=input_path, threshold_m=distance_threshold_m) if output_path is None else Path(output_path)

    with tqdm(total=4, desc="Building conflict indicator", unit="step", dynamic_ncols=True) as progress:
        candidate_df = read_csv_flexible(input_path)
        if build_num is not None:
            candidate_df = candidate_df.head(int(build_num)).copy()
        candidate_df = candidate_df.reset_index(drop=True)
        progress.set_postfix_str(f"loaded {len(candidate_df):,} candidates")
        progress.update()

        indicator_path, metadata_path, indicator, _ = write_indicator_cache(
            candidate_df,
            indicator_path=output_path,
            threshold_m=distance_threshold_m,
            input_path=input_path,
        )
        progress.set_postfix_str(f"matrix {indicator.shape[0]:,} x {indicator.shape[1]:,}")
        progress.update()

        conflict_pair_count = int(np.triu(indicator, k=1).sum())
        progress.set_postfix_str(f"{conflict_pair_count:,} conflict pairs")
        progress.update()

        progress.set_postfix_str("cache written")
        progress.update()

    return ConflictIndicatorBuildResult(
        indicator_path=indicator_path,
        metadata_path=metadata_path,
        candidate_count=len(candidate_df),
        conflict_pair_count=conflict_pair_count,
        distance_threshold_m=float(distance_threshold_m),
    )


def _lat_lon_columns(df: pd.DataFrame) -> tuple[str, str]:
    if "latitude" in df.columns and "longitude" in df.columns:
        return "latitude", "longitude"
    if "lat" in df.columns and "lon" in df.columns:
        return "lat", "lon"
    raise KeyError("Expected candidate latitude/longitude columns were not found.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Yixing sparse distance-conflict pair cache.")
    parser.add_argument("--input", type=Path, default=TOTAL_SCORE_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--distance-threshold-m", type=float, default=OPTIMIZATION_DISTANCE_THRESHOLD_M)
    parser.add_argument("--build-num", type=int, default=None)
    parser.add_argument("--dense-indicator", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> ConflictPairBuildResult | ConflictIndicatorBuildResult:
    args = parse_args(argv)
    if args.dense_indicator:
        result = build_conflict_indicator(
            input_path=args.input,
            output_path=args.output,
            distance_threshold_m=args.distance_threshold_m,
            build_num=args.build_num,
        )
        print("candidate_count =", result.candidate_count)
        print("conflict_pair_count =", result.conflict_pair_count)
        print("indicator_output =", result.indicator_path)
        print("metadata_output =", result.metadata_path)
        return result

    result = build_conflict_pair_file(
        input_path=args.input,
        output_path=args.output,
        distance_threshold_m=args.distance_threshold_m,
        build_num=args.build_num,
    )
    print("candidate_count =", result.candidate_count)
    print("conflict_pair_count =", result.conflict_pair_count)
    print("conflict_pairs_output =", result.conflict_pairs_path)
    print("metadata_output =", result.metadata_path)
    return result


if __name__ == "__main__":
    main()