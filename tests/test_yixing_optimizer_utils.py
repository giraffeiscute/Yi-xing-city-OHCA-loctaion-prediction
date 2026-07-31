from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "Optimizor"))

import build_candidates as bc
import calculate_total_score as cts
import build_conflict_pairs as conflict_module
import yixing_optimizer_utils as yxu
from config import OPTIMIZATION_DISTANCE_THRESHOLD_M


def make_raw_rows(count=3, features=None, lon_start=119.8, lat=31.3, step=0.0002):
    features = features or ["hospital", "restaurant", "bank"]
    rows = []
    for index in range(count):
        rows.append(
            {
                "id": f"src_{index}",
                "name": f"poi_{index}",
                "longitude": lon_start + step * index,
                "latitude": lat,
                "osm_tag": features[index % len(features)],
            }
        )
    return pd.DataFrame(rows)


def test_normalize_osm_tag_and_whitelist_filter():
    raw = make_raw_rows(count=3, features=[" Hospital ", "restaurant", "bank"])

    candidates, summary, _, excluded = bc.build_candidate_points(raw, dedup_distance_m=0.0)

    assert set(candidates["osm_tag"]) == {"hospital", "bank"}
    assert summary.raw_poi_count == 3
    assert summary.valid_geometry_count == 3
    assert summary.whitelist_candidate_count == 2
    assert set(excluded["osm_tag"]) == {"restaurant"}
    bc.validate_candidates(candidates)


def test_candidate_pipeline_deduplicates_within_10m_and_keeps_one_osm_tag():
    raw = pd.DataFrame(
        [
            {"id": "a", "name": "same place", "longitude": 119.8, "latitude": 31.3, "osm_tag": "hospital"},
            {"id": "b", "name": "same place", "longitude": 119.80002, "latitude": 31.3, "osm_tag": "clinic"},
            {"id": "c", "name": "far place", "longitude": 119.801, "latitude": 31.3, "osm_tag": "bank"},
            {"id": "d", "name": "excluded", "longitude": 119.802, "latitude": 31.3, "osm_tag": "restaurant"},
        ]
    )

    candidates, summary, _, excluded = bc.build_candidate_points(raw)

    bc.validate_candidates(candidates)
    assert summary.raw_poi_count == 4
    assert summary.whitelist_candidate_count == 3
    assert summary.deduplicated_candidate_count == 2
    assert len(candidates) == 2
    merged = candidates[candidates["source_row_count"] == 2].iloc[0]
    assert merged["osm_tag"] == "hospital"
    assert merged["merged_source_ids"] == "a|b"
    assert set(excluded["osm_tag"]) == {"restaurant"}


def test_candidate_ids_stay_unique_when_source_ids_repeat():
    raw = pd.DataFrame(
        [
            {"id": "dup", "name": "place a", "longitude": 119.8, "latitude": 31.3, "osm_tag": "hospital"},
            {"id": "dup", "name": "place b", "longitude": 119.9, "latitude": 31.3, "osm_tag": "bank"},
        ]
    )

    candidates, _, _, _ = bc.build_candidate_points(raw)

    bc.validate_candidates(candidates)
    assert candidates["candidate_id"].is_unique


def test_candidate_pipeline_keeps_all_deduplicated_whitelist_rows():
    raw = make_raw_rows(count=25, features=["hospital", "bank", "school"], step=0.0002)

    candidates, summary, _, _ = bc.build_candidate_points(raw)

    assert len(candidates) == 25
    assert summary.whitelist_candidate_count == 25
    assert summary.deduplicated_candidate_count == 25
    bc.validate_candidates(candidates)


def test_compute_total_scores_smoke_has_finite_ranked_scores():
    raw = make_raw_rows(count=5, features=["hospital", "bank"], step=0.0003)
    candidates, _, _, _ = bc.build_candidate_points(raw)
    candidates["component_score_smoke"] = np.linspace(1.0, 2.0, len(candidates))

    scored = cts.compute_total_scores(
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

    pairs = conflict_module.build_distance_conflict_pairs(
        df,
        threshold_m=OPTIMIZATION_DISTANCE_THRESHOLD_M,
    )

    assert pairs == [(0, 1)]


def test_scalar_haversine_distance_matches_vectorized_helper():
    scalar_distance = yxu.haversine_distance_km(31.3, 119.8, 31.3, 119.8001)
    array_distance = yxu.haversine_array(31.3, 119.8, [31.3], [119.8001])[0]

    assert scalar_distance > 0
    assert np.isclose(scalar_distance, array_distance)