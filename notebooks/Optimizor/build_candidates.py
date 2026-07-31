"""Build whitelist candidates for the Yixing optimizer from mapped POI data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import hashlib
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import (
    CANDIDATE_FEATURE_GROUPS,
    CANDIDATE_WHITELIST_PATH,
    DEDUP_DISTANCE_THRESHOLD_M,
    MAPPED_POI_PATH,
    OUTPUT_DIR,
    PROJECTED_CRS,
    WGS84_CRS,
)
from yixing_optimizer_utils import add_h3_index, ensure_output_dir, read_csv_flexible


OSM_TAG_COLUMN = "osm_tag"
WGS84_LON_COLUMNS = ("wgs84_经度", "wgs84_lon", "longitude")
WGS84_LAT_COLUMNS = ("wgs84_纬度", "wgs84_lat", "latitude")

ALLOWED_OSM_TAGS: set[str] = {
    feature
    for features in CANDIDATE_FEATURE_GROUPS.values()
    for feature in features
}

OSM_TAG_PRIORITY: dict[str, int] = {
    feature: index
    for index, feature in enumerate(
        feature
        for features in CANDIDATE_FEATURE_GROUPS.values()
        for feature in features
    )
}


@dataclass(frozen=True)
class PipelineSummary:
    """Row-count summary for candidate generation."""

    raw_poi_count: int
    valid_geometry_count: int
    whitelist_candidate_count: int
    deduplicated_candidate_count: int


@dataclass(frozen=True)
class CandidateBuildResult:
    """In-memory result for the candidate build step."""

    candidates: pd.DataFrame
    summary: PipelineSummary
    summary_df: pd.DataFrame
    excluded_feature_counts: pd.DataFrame
    output_path: Path


def normalize_osm_tag(value: object) -> str | None:
    """Return a normalized OSM tag or None for missing values."""

    if value is None or pd.isna(value):
        return None
    tag = str(value).strip().lower()
    if not tag or tag in {"nan", "none", "null", "[]"}:
        return None
    return tag


def load_candidate_source(path: Path = MAPPED_POI_PATH) -> pd.DataFrame:
    """Read the mapped Yixing POI source file."""

    return read_csv_flexible(path)


def build_whitelist_candidates(
    *,
    input_path: Path = MAPPED_POI_PATH,
    output_dir: Path = OUTPUT_DIR,
    dedup_distance_m: float = DEDUP_DISTANCE_THRESHOLD_M,
) -> CandidateBuildResult:
    """Build and write whitelist candidates from mapped Yixing POI data."""

    with tqdm(total=4, desc="Building whitelist candidates", unit="step", dynamic_ncols=True) as progress:
        raw_poi_df = load_candidate_source(Path(input_path))
        progress.set_postfix_str(f"loaded {len(raw_poi_df):,} POI rows")
        progress.update()

        candidates, summary, summary_df, excluded_feature_counts = build_candidate_points(
            raw_poi_df,
            dedup_distance_m=dedup_distance_m,
            show_progress=True,
        )
        progress.set_postfix_str(f"deduplicated {len(candidates):,} candidates")
        progress.update()

        validate_candidates(candidates)
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
    *,
    dedup_distance_m: float = DEDUP_DISTANCE_THRESHOLD_M,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, PipelineSummary, pd.DataFrame, pd.DataFrame]:
    """Build Yixing candidates from raw mapped POI rows."""

    ensure_output_dir()
    lon_col, lat_col = _wgs84_columns(raw_df)
    df = raw_df.copy()
    df["source_longitude"] = pd.to_numeric(df["longitude"], errors="coerce") if "longitude" in df.columns else np.nan
    df["source_latitude"] = pd.to_numeric(df["latitude"], errors="coerce") if "latitude" in df.columns else np.nan
    df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
    df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
    df[OSM_TAG_COLUMN] = df[OSM_TAG_COLUMN].map(normalize_osm_tag)

    valid_coord = (
        df["longitude"].between(-180, 180)
        & df["latitude"].between(-90, 90)
        & df["longitude"].notna()
        & df["latitude"].notna()
    )
    valid_df = df.loc[valid_coord].copy()

    excluded = _excluded_osm_tag_counts(valid_df)
    candidate_mask = valid_df[OSM_TAG_COLUMN].isin(ALLOWED_OSM_TAGS)
    candidate_df = valid_df.loc[candidate_mask].copy()
    if show_progress:
        tqdm.write(f"Filtering whitelist POIs: {len(candidate_df):,} rows")

    if not candidate_df.empty:
        candidate_df = add_h3_index(candidate_df)

    deduped = deduplicate_candidates(candidate_df, threshold_m=dedup_distance_m, show_progress=show_progress)
    summary = PipelineSummary(
        raw_poi_count=len(raw_df),
        valid_geometry_count=len(valid_df),
        whitelist_candidate_count=len(candidate_df),
        deduplicated_candidate_count=len(deduped),
    )
    summary_df = pd.DataFrame([summary.__dict__])
    return deduped, summary, summary_df, excluded


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
        group_values = tqdm(
            group_values,
            total=len(groups),
            desc="Building deduplicated records",
            unit="candidate",
            leave=False,
            dynamic_ncols=True,
        )
    for member_indices in group_values:
        members = df.iloc[member_indices].copy()
        source_ids = _ordered_unique(members["id"].dropna().astype(str)) if "id" in members.columns else []
        members = members.sort_values([OSM_TAG_COLUMN, "name"], na_position="last")
        tags = _ordered_osm_tags(members[OSM_TAG_COLUMN].dropna().astype(str))
        representative_tag = tags[0]
        base = members.loc[members[OSM_TAG_COLUMN] == representative_tag].iloc[0].to_dict()

        base.update(
            {
                OSM_TAG_COLUMN: representative_tag,
                "candidate_id": _make_candidate_id(members, source_ids, tags),
                "source_row_count": len(members),
                "merged_source_ids": "|".join(source_ids),
            }
        )
        records.append(base)

    out = pd.DataFrame(records).drop(columns=["geometry"], errors="ignore")
    return out.sort_values("candidate_id").reset_index(drop=True)


def validate_candidates(candidate_df: pd.DataFrame) -> None:
    """Raise AssertionError when candidate whitelist invariants fail."""

    assert OSM_TAG_COLUMN in candidate_df.columns
    assert candidate_df[OSM_TAG_COLUMN].isin(ALLOWED_OSM_TAGS).all()
    assert candidate_df["candidate_id"].is_unique


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
        candidates.groupby(OSM_TAG_COLUMN, dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values([OSM_TAG_COLUMN])
    )
    feature_counts.to_csv(output_dir / "yixing_candidate_feature_counts.csv", index=False, encoding="utf-8-sig")


def _wgs84_columns(df: pd.DataFrame) -> tuple[str, str]:
    lon_col = next((column for column in WGS84_LON_COLUMNS if column in df.columns), None)
    lat_col = next((column for column in WGS84_LAT_COLUMNS if column in df.columns), None)
    if lon_col is None or lat_col is None:
        raise KeyError("Expected WGS84 longitude/latitude columns were not found.")
    if OSM_TAG_COLUMN not in df.columns:
        raise KeyError(f"Expected {OSM_TAG_COLUMN!r} column was not found.")
    return lon_col, lat_col


def _stable_hash(values: Iterable[object], length: int = 16) -> str:
    text = "|".join("" if value is None or pd.isna(value) else str(value) for value in values)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def _excluded_osm_tag_counts(df: pd.DataFrame) -> pd.DataFrame:
    excluded = df.loc[~df[OSM_TAG_COLUMN].isin(ALLOWED_OSM_TAGS), OSM_TAG_COLUMN].fillna("<missing>")
    if excluded.empty:
        return pd.DataFrame(columns=[OSM_TAG_COLUMN, "count"])
    return excluded.value_counts().rename_axis(OSM_TAG_COLUMN).reset_index(name="count")


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _ordered_osm_tags(values: Iterable[str]) -> list[str]:
    tags = _ordered_unique(values)
    tags.sort(key=lambda tag: OSM_TAG_PRIORITY.get(tag, len(OSM_TAG_PRIORITY)))
    return tags


def _make_candidate_id(members: pd.DataFrame, source_ids: list[str], tags: list[str]) -> str:
    lon = round(float(members["longitude"].mean()), 7)
    lat = round(float(members["latitude"].mean()), 7)
    names = "|".join(_ordered_unique(members.get("name", pd.Series(dtype=str)).dropna().astype(str)))
    readable_source = source_ids[0] if len(source_ids) == 1 else "merged"
    token = _stable_hash([",".join(source_ids), names, "|".join(tags), lon, lat], length=12)
    return f"yx_{readable_source}_{token}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Yixing whitelist candidate locations.")
    parser.add_argument("--input", type=Path, default=MAPPED_POI_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dedup-distance-m", type=float, default=DEDUP_DISTANCE_THRESHOLD_M)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> CandidateBuildResult:
    args = parse_args(argv)
    result = build_whitelist_candidates(
        input_path=args.input,
        output_dir=args.output_dir,
        dedup_distance_m=args.dedup_distance_m,
    )
    print("candidate_count =", len(result.candidates))
    print("output =", result.output_path)
    return result


if __name__ == "__main__":
    main()


