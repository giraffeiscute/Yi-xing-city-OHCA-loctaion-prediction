from contextlib import contextmanager
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OPTIMIZOR_DIR = PROJECT_ROOT / "notebooks" / "Optimizor"
sys.path.insert(0, str(OPTIMIZOR_DIR))

import calculate_total_score as cts
import run_all
import yixing_optimizer_utils as yxu


@contextmanager
def temp_score_paths():
    paths = [
        OPTIMIZOR_DIR / "tests" / "_candidate_input.csv",
        OPTIMIZOR_DIR / "tests" / "_total_score.csv",
    ]
    try:
        yield paths[0], paths[1]
    finally:
        for path in paths:
            path.unlink(missing_ok=True)


def candidate_frame() -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "candidate_id": "yx_a",
                "name": "candidate a",
                "longitude": 119.8,
                "latitude": 31.3,
                "candidate_group": "healthcare",
                "primary_feature": "hospital",
                "matched_features": "hospital",
            },
            {
                "candidate_id": "yx_b",
                "name": "candidate b",
                "longitude": 119.86,
                "latitude": 31.3,
                "candidate_group": "public_services",
                "primary_feature": "bank",
                "matched_features": "bank",
            },
        ]
    )
    return yxu.add_h3_index(df)


def test_full_whitelist_mode_keeps_all_deduplicated_candidates():
    raw = pd.DataFrame(
        {
            "id": [f"src_{index}" for index in range(yxu.MAX_CANDIDATES + 5)],
            "name": [f"poi_{index}" for index in range(yxu.MAX_CANDIDATES + 5)],
            "longitude": [119.8 + 0.0002 * index for index in range(yxu.MAX_CANDIDATES + 5)],
            "latitude": [31.3 for _ in range(yxu.MAX_CANDIDATES + 5)],
            "osm_tag": ["hospital" for _ in range(yxu.MAX_CANDIDATES + 5)],
        }
    )

    full_candidates, full_summary, _, _ = yxu.build_candidate_points(
        raw,
        boundary_gdf=None,
        max_candidates=None,
    )
    sampled_candidates, _, _, _ = yxu.build_candidate_points(
        raw,
        boundary_gdf=None,
        max_candidates=10,
    )

    assert len(full_candidates) == yxu.MAX_CANDIDATES + 5
    assert full_summary.sampled_candidate_count == yxu.MAX_CANDIDATES + 5
    yxu.validate_candidates(full_candidates)
    assert len(sampled_candidates) == 10
    yxu.validate_candidates(sampled_candidates, max_candidates=10)


def test_calculate_total_score_writes_xgb_mlp_and_svr_columns(monkeypatch):
    candidates = candidate_frame()
    h3_ids = candidates["h3_l7"].drop_duplicates().astype(str).tolist()

    def fake_prepare_h3_training_frames():
        source = pd.DataFrame({"id": ["source_h3"], "hospital": [1.0], "bank": [0.0], "ohca": [1.0]})
        yixing = pd.DataFrame(
            {
                "id": h3_ids,
                "hospital": [1.0 for _ in h3_ids],
                "bank": [0.5 for _ in h3_ids],
            }
        )
        return source, yixing, ["hospital", "bank"]

    def fake_compute_xgb_h3_scores(source_h3_df, yixing_h3_df, feature_cols):
        score_df = pd.DataFrame(
            {
                "id": h3_ids,
                "hospital": [1.5 for _ in h3_ids],
                "bank": [0.0 for _ in h3_ids],
            }
        )
        prediction_df = pd.DataFrame({"id": h3_ids, "predicted_ohca_xgb": [0.5 for _ in h3_ids]})
        return score_df, prediction_df

    def fake_compute_mlp_h3_scores(source_h3_df, yixing_h3_df, feature_cols, *, iter_num):
        score_df = pd.DataFrame(
            {
                "id": h3_ids,
                "hospital": [0.2 for _ in h3_ids],
                "bank": [2.0 for _ in h3_ids],
            }
        )
        prediction_df = pd.DataFrame({"id": h3_ids, "predicted_ohca_mlp": [2.0 for _ in h3_ids]})
        return score_df, prediction_df

    def fake_compute_svr_h3_scores(source_h3_df, yixing_h3_df, feature_cols):
        score_df = pd.DataFrame(
            {
                "id": h3_ids,
                "hospital": [0.8 for _ in h3_ids],
                "bank": [0.4 for _ in h3_ids],
            }
        )
        prediction_df = pd.DataFrame({"id": h3_ids, "predicted_ohca_svr": [1.0 for _ in h3_ids]})
        return score_df, prediction_df

    monkeypatch.setattr(cts, "prepare_h3_training_frames", fake_prepare_h3_training_frames)
    monkeypatch.setattr(cts, "compute_xgb_h3_scores", fake_compute_xgb_h3_scores)
    monkeypatch.setattr(cts, "compute_mlp_h3_scores", fake_compute_mlp_h3_scores)
    monkeypatch.setattr(cts, "compute_svr_h3_scores", fake_compute_svr_h3_scores)

    with temp_score_paths() as (input_path, output_path):
        candidates.to_csv(input_path, index=False, encoding="utf-8-sig")
        result = cts.calculate_total_scores(
            input_path=input_path,
            output_path=output_path,
            mlp_iter_num=1,
        )
        written = pd.read_csv(output_path, encoding="utf-8-sig")

        expected_columns = {
            "h3_shap_score_xgb",
            "h3_shap_score_mlp",
            "h3_shap_score_svr",
            "predicted_ohca_xgb",
            "predicted_ohca_mlp",
            "predicted_ohca_svr",
            "total_score_xgb",
            "total_score_mlp",
            "total_score_svr",
            "score_rank_xgb",
            "score_rank_mlp",
            "score_rank_svr",
        }
        assert result.output_path == output_path
        assert expected_columns.issubset(written.columns)
        for score_col in ("total_score_xgb", "total_score_mlp", "total_score_svr"):
            assert np.isfinite(written[score_col]).all()


def test_run_all_optimization_uses_xgb_and_mlp_score_columns(monkeypatch):
    calls = []

    def fake_optimize_candidates(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"optimization_source_index": [0]}), 1.0, [(0, 1)]

    monkeypatch.setattr(run_all, "optimize_candidates", fake_optimize_candidates)

    output_dir = OPTIMIZOR_DIR / "tests"
    result = run_all.run_optimizations(
        total_score_path=output_dir / "_total_score.csv",
        output_dir=output_dir,
        loc_num=3,
        distance_threshold_m=20.0,
        time_limit_seconds=30,
        build_num=50,
    )

    assert calls[0]["score_col"] == "total_score_xgb"
    assert calls[0]["use_mlp"] is False
    assert calls[0]["output_path"] == output_dir / "yixing_selected_locations_xgb.csv"
    assert calls[1]["score_col"] == "total_score_mlp"
    assert calls[1]["use_mlp"] is True
    assert calls[1]["output_path"] == output_dir / "yixing_selected_locations_mlp.csv"
    assert len(result.xgb_selected) == 1
    assert len(result.mlp_selected) == 1