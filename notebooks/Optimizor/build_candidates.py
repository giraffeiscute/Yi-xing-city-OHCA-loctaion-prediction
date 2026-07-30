"""Build deduplicated whitelist candidates for the Yixing optimizer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import (
    CANDIDATE_FEATURE_GROUPS,
    CANDIDATE_WHITELIST_PATH,
    CITY_NAME,
    DEFAULT_MAX_CANDIDATES,
    DEDUP_DISTANCE_THRESHOLD_M,
    FEATURE_SOURCE_COLUMNS,
    MAPPED_POI_PATH,
    MAX_CANDIDATES,
    OUTPUT_DIR,
    PROJECTED_CRS,
    RANDOM_SEED,
    WGS84_CRS,
)
from yixing_optimizer_utils import add_h3_index, ensure_output_dir, read_csv_flexible


ALLOWED_FEATURES: set[str] = {
    feature
    for features in CANDIDATE_FEATURE_GROUPS.values()
    for feature in features
}

FEATURE_TO_GROUP: dict[str, str] = {
    feature: group
    for group, features in CANDIDATE_FEATURE_GROUPS.items()
    for feature in features
}

FEATURE_PRIORITY: dict[str, int] = {
    feature: index
    for index, feature in enumerate(
        feature
        for features in CANDIDATE_FEATURE_GROUPS.values()
        for feature in features
    )
}
FEATURE_SPLIT_PATTERN = re.compile(r"[|;,/]+")


@dataclass(frozen=True)
class PipelineSummary:
    """Row-count summary for candidate generation."""

    raw_poi_count: int
    valid_geometry_count: int
    inside_boundary_count: int
    whitelist_candidate_count: int
    deduplicated_candidate_count: int
    sampled_candidate_count: int


@dataclass(frozen=True)
class CandidateBuildResult:
    """In-memory result for the candidate build step."""

    candidates: pd.DataFrame
    summary: PipelineSummary
    summary_df: pd.DataFrame
    excluded_feature_counts: pd.DataFrame
    output_path: Path


def parse_candidate_limit(value: str | int | None) -> int | None:
    """Parse a CLI candidate cap; all/none means no sampling cap."""

    if value is None:
        return None
    if isinstance(value, int):
        return value
    normalized = value.strip().lower()
    if normalized in {"", "all", "none", "null"}:
        return None
    parsed = int(normalized)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--max-candidates must be positive, all, or none.")
    return parsed


def normalize_feature_value(value: object) -> list[str]:
    """Normalize one raw feature field into possible feature tokens."""

    if value is None or pd.isna(value):
        return []
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return []
    tokens = []
    for token in FEATURE_SPLIT_PATTERN.split(text):
        clean = token.strip().lower()
        if clean and clean not in {"nan", "none", "null"}:
            tokens.append(clean)
    return tokens


def extract_candidate_feature_matches(row: pd.Series) -> list[dict[str, str]]:
    """Extract allowed feature matches and preserve their source fields."""

    matches: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source_field in FEATURE_SOURCE_COLUMNS:
        if source_field not in row.index:
            continue
        raw_value = row[source_field]
        for feature in normalize_feature_value(raw_value):
            if feature not in ALLOWED_FEATURES:
                continue
            key = (feature, source_field)
            if key in seen:
                continue
            seen.add(key)
            matches.append(
                {
                    "feature": feature,
                    "source_feature_field": source_field,
                    "source_feature_value": "" if pd.isna(raw_value) else str(raw_value),
                }
            )
    matches.sort(key=lambda item: FEATURE_PRIORITY[item["feature"]])
    return matches


def extract_candidate_features(row: pd.Series) -> list[str]:
    """Return allowed candidate features from a raw POI row."""

    features: list[str] = []
    seen: set[str] = set()
    for match in extract_candidate_feature_matches(row):
        feature = match["feature"]
        if feature not in seen:
            seen.add(feature)
            features.append(feature)
    return features


def load_yixing_boundary(place_name: str = CITY_NAME):
    """Load the Yixing administrative boundary as EPSG:4326 GeoDataFrame."""

    import osmnx as ox

    return ox.geocode_to_gdf([place_name]).to_crs(WGS84_CRS)


def load_candidate_source(path: Path = MAPPED_POI_PATH) -> pd.DataFrame:
    """Read the mapped Yixing POI source file."""

    return read_csv_flexible(path)


def build_whitelist_candidates(
    *,
    input_path: Path = MAPPED_POI_PATH,
    output_dir: Path = OUTPUT_DIR,
    run_boundary_filter: bool = True,
    dedup_distance_m: float = DEDUP_DISTANCE_THRESHOLD_M,
    max_candidates: int | None = DEFAULT_MAX_CANDIDATES,
    random_seed: int = RANDOM_SEED,
) -> CandidateBuildResult:
    """Build and write whitelist candidates from mapped Yixing POI data."""

    with tqdm(total=5, desc="Building whitelist candidates", unit="step", dynamic_ncols=True) as progress:
        raw_poi_df = load_candidate_source(Path(input_path))
        progress.set_postfix_str(f"loaded {len(raw_poi_df):,} POI rows")
        progress.update()

        boundary_gdf = load_yixing_boundary() if run_boundary_filter else None
        progress.set_postfix_str("boundary filter on" if run_boundary_filter else "boundary filter off")
        progress.update()

        candidates, summary, summary_df, excluded_feature_counts = build_candidate_points(
            raw_poi_df,
            boundary_gdf,
            dedup_distance_m=dedup_distance_m,
            max_candidates=max_candidates,
            random_seed=random_seed,
            show_progress=True,
        )
        progress.set_postfix_str(f"deduplicated {len(candidates):,} candidates")
        progress.update()

        validate_candidates(candidates, max_candidates=max_candidates)
        progress.set_postfix_str("validated")
        progress.update()

        write_candidate_outputs(candidates, summary_df, excluded_feature_counts, output_dir=Path(output_dir))
        progress.set_postfix_str("outputs written")
        progress.update()

    return CandidateBuildResult(
        candidates=candidates,
        summary=summary,
        summary_df=summary_df,
        excluded_feature_counts=excluded_feature_counts,
        output_path=Path(output_dir) / CANDIDATE_WHITELIST_PATH.name,
    )


def build_candidate_points(
    raw_df: pd.DataFrame,
    boundary_gdf=None,
    *,
    dedup_distance_m: float = DEDUP_DISTANCE_THRESHOLD_M,
    max_candidates: int | None = DEFAULT_MAX_CANDIDATES,
    random_seed: int = RANDOM_SEED,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, PipelineSummary, pd.DataFrame, pd.DataFrame]:
    """Build Yixing candidates from raw mapped POI rows."""

    ensure_output_dir()
    lon_col, lat_col = _wgs84_columns(raw_df)
    df = raw_df.copy()
    df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
    df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
    valid_coord = (
        df["longitude"].between(-180, 180)
        & df["latitude"].between(-90, 90)
        & df["longitude"].notna()
        & df["latitude"].notna()
    )
    valid_df = df.loc[valid_coord].copy()

    excluded = _excluded_feature_counts(valid_df)
    candidate_df = assign_candidate_metadata(valid_df, show_progress=show_progress)

    if not candidate_df.empty:
        candidate_df = add_h3_index(candidate_df)
        candidate_df["source_longitude"] = candidate_df["longitude"]
        candidate_df["source_latitude"] = candidate_df["latitude"]

    inside_count = len(candidate_df)
    if boundary_gdf is not None and not candidate_df.empty:
        candidate_df = filter_points_within_boundary(candidate_df, boundary_gdf)
        inside_count = len(candidate_df)

    deduped = deduplicate_candidates(candidate_df, threshold_m=dedup_distance_m, show_progress=show_progress)
    sampled = sample_candidates(deduped, max_candidates=max_candidates, random_seed=random_seed)

    summary = PipelineSummary(
        raw_poi_count=len(raw_df),
        valid_geometry_count=len(valid_df),
        inside_boundary_count=inside_count,
        whitelist_candidate_count=len(candidate_df),
        deduplicated_candidate_count=len(deduped),
        sampled_candidate_count=len(sampled),
    )
    summary_df = pd.DataFrame([summary.__dict__])
    return sampled, summary, summary_df, excluded


def assign_candidate_metadata(
    df: pd.DataFrame,
    *,
    show_progress: bool = False,
    progress_desc: str = "Filtering whitelist POIs",
) -> pd.DataFrame:
    """Attach feature, group, source, and stable candidate-id metadata."""

    rows: list[dict[str, object]] = []
    iterator = df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(df), desc=progress_desc, unit="row", leave=False, dynamic_ncols=True)
    for _, row in iterator:
        matches = extract_candidate_feature_matches(row)
        if not matches:
            continue
        features = []
        seen_features: set[str] = set()
        for match in matches:
            feature = match["feature"]
            if feature in seen_features:
                continue
            seen_features.add(feature)
            features.append(feature)
        primary = features[0]
        first_match = next(match for match in matches if match["feature"] == primary)
        record = row.to_dict()
        record.update(
            {
                "matched_features": "|".join(features),
                "primary_feature": primary,
                "candidate_group": FEATURE_TO_GROUP[primary],
                "source_feature_field": first_match["source_feature_field"],
                "source_feature_value": first_match["source_feature_value"],
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def filter_points_within_boundary(df: pd.DataFrame, boundary_gdf) -> pd.DataFrame:
    """Keep only candidate points inside the supplied boundary."""

    import geopandas as gpd

    if df.empty:
        return df.copy()
    geometry = gpd.points_from_xy(df["longitude"], df["latitude"])
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=WGS84_CRS)
    boundary = boundary_gdf.to_crs(WGS84_CRS)
    inside = gdf.geometry.within(boundary.geometry.unary_union) | gdf.geometry.touches(boundary.geometry.unary_union)
    return pd.DataFrame(gdf.loc[inside].drop(columns=["geometry"]))


def deduplicate_candidates(
    candidate_df: pd.DataFrame,
    *,
    threshold_m: float = DEDUP_DISTANCE_THRESHOLD_M,
    projected_crs: str = PROJECTED_CRS,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Merge candidate points whose projected distance is within threshold_m."""

    if candidate_df.empty:
        return candidate_df.copy()

    import geopandas as gpd
    from scipy.spatial import cKDTree

    df = candidate_df.reset_index(drop=True).copy()
    geometry = gpd.points_from_xy(df["longitude"], df["latitude"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=WGS84_CRS).to_crs(projected_crs)
    coords = np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])
    tree = cKDTree(coords)
    pairs = tree.query_pairs(float(threshold_m))

    parent = list(range(len(df)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in pairs:
        union(i, j)

    groups: dict[int, list[int]] = {}
    for idx in range(len(df)):
        groups.setdefault(find(idx), []).append(idx)

    records = []
    group_values = groups.values()
    if show_progress:
        group_values = tqdm(group_values, total=len(groups), desc="Building deduplicated records", unit="candidate", leave=False, dynamic_ncols=True)
    for member_indices in group_values:
        members = df.iloc[member_indices].copy()
        members = members.sort_values(["primary_feature", "name"], na_position="last")
        base = members.iloc[0].to_dict()

        features = _ordered_unique(
            feature
            for value in members["matched_features"].dropna()
            for feature in str(value).split("|")
            if feature
        )
        source_fields = _ordered_unique(members["source_feature_field"].dropna().astype(str))
        source_values = _ordered_unique(members["source_feature_value"].dropna().astype(str))
        source_ids = _ordered_unique(members["id"].dropna().astype(str)) if "id" in members.columns else []

        primary = features[0] if features else base["primary_feature"]
        base.update(
            {
                "candidate_id": _make_candidate_id(members, source_ids),
                "candidate_group": FEATURE_TO_GROUP.get(primary, base["candidate_group"]),
                "primary_feature": primary,
                "matched_features": "|".join(features),
                "source_feature_field": "|".join(source_fields),
                "source_feature_value": "|".join(source_values),
                "source_row_count": len(members),
                "merged_source_ids": "|".join(source_ids),
            }
        )
        records.append(base)

    out = pd.DataFrame(records).drop(columns=["geometry"], errors="ignore")
    out = out.sort_values("candidate_id").reset_index(drop=True)
    return out


def sample_candidates(
    candidate_df: pd.DataFrame,
    *,
    max_candidates: int | None = DEFAULT_MAX_CANDIDATES,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Sample candidates reproducibly after cleaning and deduplication."""

    df = candidate_df.sort_values("candidate_id").reset_index(drop=True)
    if max_candidates is None:
        return df
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive or None.")
    if len(df) <= max_candidates:
        return df
    sampled = df.sample(n=max_candidates, random_state=random_seed)
    return sampled.sort_values("candidate_id").reset_index(drop=True)


def validate_candidates(candidate_df: pd.DataFrame, *, max_candidates: int | None = None) -> None:
    """Raise AssertionError when candidate whitelist invariants fail."""

    assert candidate_df["primary_feature"].isin(ALLOWED_FEATURES).all()
    for value in candidate_df["matched_features"].fillna(""):
        features = [feature for feature in str(value).split("|") if feature]
        assert features
        assert all(feature in ALLOWED_FEATURES for feature in features)
    assert candidate_df["candidate_id"].is_unique
    if max_candidates is not None:
        assert len(candidate_df) <= max_candidates


def write_candidate_outputs(
    candidates: pd.DataFrame,
    summary_df: pd.DataFrame,
    excluded_feature_counts: pd.DataFrame,
    *,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Write candidate pipeline outputs."""

    output_dir = ensure_output_dir(output_dir)
    candidates.to_csv(output_dir / CANDIDATE_WHITELIST_PATH.name, index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "yixing_poi_cleaned_summary.csv", index=False, encoding="utf-8-sig")
    excluded_feature_counts.to_csv(output_dir / "yixing_excluded_feature_counts.csv", index=False, encoding="utf-8-sig")
    feature_counts = (
        candidates.groupby(["candidate_group", "primary_feature"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["candidate_group", "count"], ascending=[True, False])
    )
    feature_counts.to_csv(output_dir / "yixing_candidate_feature_counts.csv", index=False, encoding="utf-8-sig")


def _wgs84_columns(df: pd.DataFrame) -> tuple[str, str]:
    if "wgs84_蝏漲" in df.columns and "wgs84_蝥砍漲" in df.columns:
        return "wgs84_蝏漲", "wgs84_蝥砍漲"
    if "wgs84_lon" in df.columns and "wgs84_lat" in df.columns:
        return "wgs84_lon", "wgs84_lat"
    if "longitude" in df.columns and "latitude" in df.columns:
        return "longitude", "latitude"
    raise KeyError("Expected WGS84 longitude/latitude columns were not found.")


def _stable_hash(values: Iterable[object], length: int = 16) -> str:
    text = "|".join("" if value is None or pd.isna(value) else str(value) for value in values)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def _excluded_feature_counts(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        raw_features = []
        for source_field in FEATURE_SOURCE_COLUMNS:
            if source_field in row.index:
                raw_features.extend(normalize_feature_value(row[source_field]))
        matched = [feature for feature in raw_features if feature in ALLOWED_FEATURES]
        if matched:
            continue
        feature_label = raw_features[0] if raw_features else "<missing>"
        records.append(
            {
                "source_feature": feature_label,
                "source_feature_field": _first_nonempty_feature_field(row),
            }
        )
    if not records:
        return pd.DataFrame(columns=["source_feature", "source_feature_field", "count"])
    return (
        pd.DataFrame(records)
        .groupby(["source_feature", "source_feature_field"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


def _first_nonempty_feature_field(row: pd.Series) -> str:
    for source_field in FEATURE_SOURCE_COLUMNS:
        if source_field in row.index and normalize_feature_value(row[source_field]):
            return source_field
    return "<missing>"


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    out.sort(key=lambda item: FEATURE_PRIORITY.get(item, len(FEATURE_PRIORITY)))
    return out


def _make_candidate_id(members: pd.DataFrame, source_ids: list[str]) -> str:
    lon = round(float(members["longitude"].mean()), 7)
    lat = round(float(members["latitude"].mean()), 7)
    names = "|".join(_ordered_unique(members.get("name", pd.Series(dtype=str)).dropna().astype(str)))
    features = "|".join(_ordered_unique(members.get("primary_feature", pd.Series(dtype=str)).dropna().astype(str)))
    readable_source = source_ids[0] if len(source_ids) == 1 else "merged"
    token = _stable_hash([",".join(source_ids), names, features, lon, lat], length=12)
    return f"yx_{readable_source}_{token}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Yixing whitelist candidate locations.")
    parser.add_argument("--input", type=Path, default=MAPPED_POI_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-boundary-filter", action="store_true")
    parser.add_argument("--dedup-distance-m", type=float, default=DEDUP_DISTANCE_THRESHOLD_M)
    parser.add_argument("--max-candidates", type=parse_candidate_limit, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> CandidateBuildResult:
    args = parse_args(argv)
    result = build_whitelist_candidates(
        input_path=args.input,
        output_dir=args.output_dir,
        run_boundary_filter=not args.no_boundary_filter,
        dedup_distance_m=args.dedup_distance_m,
        max_candidates=args.max_candidates,
        random_seed=args.random_seed,
    )
    print("candidate_count =", len(result.candidates))
    print("output =", result.output_path)
    return result


if __name__ == "__main__":
    main()
