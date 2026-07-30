from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "Optimizor"))

import yixing_optimizer_utils as yxu


def make_raw_rows(count=3, features=None, lon_start=119.8, lat=31.3, step=0.0002):
    features = features or ["hospital", "restaurant", "bank"]
    rows = []
    for index in range(count):
        rows.append(
            {
                "id": f"src_{index}",
                "name": f"poi_{index}",
                "wgs84_经度": lon_start + step * index,
                "wgs84_纬度": lat,
                "osm_tag": features[index % len(features)],
            }
        )
    return pd.DataFrame(rows)


def test_extract_candidate_features_uses_whitelist_only():
    row = pd.Series({"osm_tag": " Hospital | restaurant ", "amenity": "bank"})

    assert yxu.extract_candidate_features(row) == ["hospital", "bank"]
    assert "restaurant" not in yxu.ALLOWED_FEATURES


def test_candidate_pipeline_deduplicates_within_10m_and_merges_features():
    raw = pd.DataFrame(
        [
            {"id": "a", "name": "same place", "wgs84_经度": 119.8, "wgs84_纬度": 31.3, "osm_tag": "hospital"},
            {"id": "b", "name": "same place", "wgs84_经度": 119.80002, "wgs84_纬度": 31.3, "osm_tag": "clinic"},
            {"id": "c", "name": "far place", "wgs84_经度": 119.801, "wgs84_纬度": 31.3, "osm_tag": "bank"},
            {"id": "d", "name": "excluded", "wgs84_经度": 119.802, "wgs84_纬度": 31.3, "osm_tag": "restaurant"},
        ]
    )

    candidates, summary, _, excluded = yxu.build_candidate_points(
        raw,
        boundary_gdf=None,
        max_candidates=10,
        random_seed=yxu.RANDOM_SEED,
    )

    yxu.validate_candidates(candidates)
    assert summary.raw_poi_count == 4
    assert summary.whitelist_candidate_count == 3
    assert summary.deduplicated_candidate_count == 2
    assert len(candidates) == 2
    merged = candidates[candidates["source_row_count"] == 2].iloc[0]
    assert merged["matched_features"] == "hospital|clinic"
    assert merged["primary_feature"] == "hospital"
    assert set(excluded["source_feature"]) == {"restaurant"}

def test_candidate_ids_stay_unique_when_source_ids_repeat():
    raw = pd.DataFrame(
        [
            {"id": "dup", "name": "place a", "longitude": 119.8, "latitude": 31.3, "osm_tag": "hospital"},
            {"id": "dup", "name": "place b", "longitude": 119.9, "latitude": 31.3, "osm_tag": "bank"},
        ]
    )

    candidates, _, _, _ = yxu.build_candidate_points(raw, boundary_gdf=None, max_candidates=None)

    yxu.validate_candidates(candidates)
    assert candidates["candidate_id"].is_unique

def test_candidate_sampling_is_reproducible_when_cap_is_explicit():
    raw = make_raw_rows(count=2005, features=["hospital", "bank", "school"], step=0.0002)

    full_candidates, _, _, _ = yxu.build_candidate_points(raw, boundary_gdf=None)
    candidates_a, _, _, _ = yxu.build_candidate_points(
        raw,
        boundary_gdf=None,
        max_candidates=yxu.MAX_CANDIDATES,
    )
    candidates_b, _, _, _ = yxu.build_candidate_points(
        raw,
        boundary_gdf=None,
        max_candidates=yxu.MAX_CANDIDATES,
    )

    assert len(full_candidates) == 2005
    assert len(candidates_a) == yxu.MAX_CANDIDATES
    assert candidates_a["candidate_id"].tolist() == candidates_b["candidate_id"].tolist()
    yxu.validate_candidates(full_candidates)
    yxu.validate_candidates(candidates_a, max_candidates=yxu.MAX_CANDIDATES)


def test_compute_total_scores_smoke_has_finite_ranked_scores():
    raw = make_raw_rows(count=5, features=["hospital", "bank"], step=0.0003)
    candidates, _, _, _ = yxu.build_candidate_points(raw, boundary_gdf=None, max_candidates=5)
    candidates["component_score_smoke"] = np.linspace(1.0, 2.0, len(candidates))

    scored = yxu.compute_total_scores(
        candidates,
        point_score_col="component_score_smoke",
        total_score_col="total_score",
    )

    assert scored["total_score"].notna().all()
    assert np.isfinite(scored["total_score"]).all()
    assert scored["total_score"].is_monotonic_decreasing
    assert set(["component_score_smoke", "total_score", "score_rank"]).issubset(scored.columns)


def test_distance_conflict_pairs_use_20m_threshold():
    df = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "longitude": [119.8, 119.80005, 119.801],
            "latitude": [31.3, 31.3, 31.3],
            "total_score": [3.0, 2.0, 1.0],
        }
    )

    pairs = yxu.build_distance_conflict_pairs(
        df,
        threshold_m=yxu.OPTIMIZATION_DISTANCE_THRESHOLD_M,
    )

    assert pairs == [(0, 1)]