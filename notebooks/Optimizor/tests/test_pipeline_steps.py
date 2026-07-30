from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OPTIMIZOR_DIR = PROJECT_ROOT / "notebooks" / "Optimizor"
sys.path.insert(0, str(OPTIMIZOR_DIR))

import build_candidates as bc
import calculate_total_score as cts
import Data as data_module
import run_this
import yixing_optimizer_utils as yxu
from config import MAX_CANDIDATES


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
            "id": [f"src_{index}" for index in range(MAX_CANDIDATES + 5)],
            "name": [f"poi_{index}" for index in range(MAX_CANDIDATES + 5)],
            "longitude": [119.8 + 0.0002 * index for index in range(MAX_CANDIDATES + 5)],
            "latitude": [31.3 for _ in range(MAX_CANDIDATES + 5)],
            "osm_tag": ["hospital" for _ in range(MAX_CANDIDATES + 5)],
        }
    )

    full_candidates, full_summary, _, _ = bc.build_candidate_points(
        raw,
        boundary_gdf=None,
        max_candidates=None,
    )
    sampled_candidates, _, _, _ = bc.build_candidate_points(
        raw,
        boundary_gdf=None,
        max_candidates=10,
    )

    assert len(full_candidates) == MAX_CANDIDATES + 5
    assert full_summary.sampled_candidate_count == MAX_CANDIDATES + 5
    bc.validate_candidates(full_candidates)
    assert len(sampled_candidates) == 10
    bc.validate_candidates(sampled_candidates, max_candidates=10)


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
            "total_score_xgb",
            "total_score_mlp",
            "total_score_svr",
            "score_rank_xgb",
            "score_rank_mlp",
            "score_rank_svr",
        }
        omitted_columns = {
            "h3_shap_score_xgb",
            "h3_shap_score_mlp",
            "h3_shap_score_svr",
            "predicted_ohca_xgb",
            "predicted_ohca_mlp",
            "predicted_ohca_svr",
        }
        assert result.output_path == output_path
        assert expected_columns.issubset(written.columns)
        assert omitted_columns.isdisjoint(written.columns)
        for score_col in ("total_score_xgb", "total_score_mlp", "total_score_svr"):
            assert np.isfinite(written[score_col]).all()


def conflict_candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"candidate_id": "near_a", "longitude": 119.8, "latitude": 31.3, "total_score": 3.0},
            {"candidate_id": "near_b", "longitude": 119.8001, "latitude": 31.3, "total_score": 2.0},
            {"candidate_id": "far_c", "longitude": 119.802, "latitude": 31.3, "total_score": 1.0},
        ]
    )


def test_build_indicator_matrix_is_dense_bool_symmetric():
    indicator = data_module.build_indicator_matrix(conflict_candidate_frame(), threshold_m=20.0)

    assert indicator.dtype == np.bool_
    assert indicator.shape == (3, 3)
    assert not indicator.diagonal().any()
    assert np.array_equal(indicator, indicator.T)
    assert indicator[0, 1]
    assert not indicator[0, 2]
    assert not indicator[1, 2]


def test_data_read_dataframe_uses_valid_indicator_cache_and_slices_build_num():
    candidate_df = conflict_candidate_frame()
    indicator_path = OPTIMIZOR_DIR / "tests" / "_indicator_cache_test.npy"
    try:
        data_module.write_indicator_cache(candidate_df, indicator_path=indicator_path, threshold_m=20.0)

        data = data_module.Data().read_dataframe(
            candidate_df,
            loc_num=1,
            dist_limit_m=20.0,
            build_num=2,
            score_col="total_score",
            indicator_path=indicator_path,
        )

        assert data.indicator_cache_used is True
        assert data.build_num == 2
        assert data.conflict_pairs == [(0, 1)]
    finally:
        indicator_path.unlink(missing_ok=True)
        data_module.indicator_metadata_path(indicator_path).unlink(missing_ok=True)


def test_data_read_dataframe_falls_back_when_indicator_cache_missing(monkeypatch):
    calls = []

    def fake_build_distance_conflict_pairs(candidate_df, *, threshold_m, projected_crs=data_module.PROJECTED_CRS):
        calls.append((len(candidate_df), threshold_m, projected_crs))
        return [(0, 1)]

    monkeypatch.setattr(data_module, "build_distance_conflict_pairs", fake_build_distance_conflict_pairs)
    missing_indicator_path = OPTIMIZOR_DIR / "tests" / "_missing_indicator_i_j_20m.npy"

    data = data_module.Data().read_dataframe(
        conflict_candidate_frame(),
        loc_num=1,
        dist_limit_m=20.0,
        score_col="total_score",
        indicator_path=missing_indicator_path,
    )

    assert data.indicator_cache_used is False
    assert data.conflict_pairs == [(0, 1)]
    assert calls == [(3, 20.0, data_module.PROJECTED_CRS)]


def test_run_this_optimization_uses_xgb_mlp_and_svr_score_columns(monkeypatch):
    calls = []

    def fake_optimize_candidates(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"optimization_source_index": [0]}), 1.0, [(0, 1)]

    monkeypatch.setattr(run_this, "optimize_candidates", fake_optimize_candidates)

    output_dir = OPTIMIZOR_DIR / "tests"
    result = run_this.run_optimizations(
        total_score_path=output_dir / "_total_score.csv",
        output_dir=output_dir,
        loc_num=3,
        distance_threshold_m=20.0,
        time_limit_seconds=30,
        build_num=50,
    )

    expected_indicator_path = output_dir / "indicator_i_j_20m.npy"

    assert calls[0]["score_col"] == "total_score_xgb"
    assert calls[0]["use_mlp"] is False
    assert calls[0]["output_path"] == output_dir / "yixing_selected_locations_xgb.csv"
    assert calls[1]["score_col"] == "total_score_mlp"
    assert calls[1]["use_mlp"] is True
    assert calls[1]["output_path"] == output_dir / "yixing_selected_locations_mlp.csv"
    assert calls[2]["score_col"] == "total_score_svr"
    assert calls[2]["use_mlp"] is False
    assert calls[2]["output_path"] == output_dir / "yixing_selected_locations_svr.csv"
    assert [call["indicator_path"] for call in calls] == [expected_indicator_path] * 3
    assert [call["use_indicator_cache"] for call in calls] == [True, True, True]
    assert len(result.xgb_selected) == 1
    assert len(result.mlp_selected) == 1
    assert len(result.svr_selected) == 1


def test_run_this_pipeline_wires_build_score_and_optimization(monkeypatch):
    calls = []
    output_dir = OPTIMIZOR_DIR / "tests"
    candidate_result = SimpleNamespace(
        candidates=pd.DataFrame({"candidate_id": ["yx_a"]}),
        output_path=output_dir / "yixing_candidates_whitelist.csv",
    )
    total_score_result = SimpleNamespace(
        scored_candidates=pd.DataFrame({"candidate_id": ["yx_a"]}),
        output_path=output_dir / "total_score.csv",
    )
    optimization_result = run_this.OptimizationRunResult(
        xgb_selected=pd.DataFrame({"candidate_id": ["yx_a"]}),
        xgb_objective_value=1.0,
        xgb_conflict_pairs=[],
        mlp_selected=pd.DataFrame({"candidate_id": ["yx_a"]}),
        mlp_objective_value=1.0,
        mlp_conflict_pairs=[],
        svr_selected=pd.DataFrame({"candidate_id": ["yx_a"]}),
        svr_objective_value=1.0,
        svr_conflict_pairs=[],
    )

    def fake_build_whitelist_candidates(**kwargs):
        calls.append(("build", kwargs))
        return candidate_result

    def fake_calculate_total_scores(**kwargs):
        calls.append(("score", kwargs))
        return total_score_result

    def fake_run_optimizations(**kwargs):
        calls.append(("optimize", kwargs))
        return optimization_result

    monkeypatch.setattr(run_this, "build_whitelist_candidates", fake_build_whitelist_candidates)
    monkeypatch.setattr(run_this, "calculate_total_scores", fake_calculate_total_scores)
    monkeypatch.setattr(run_this, "run_optimizations", fake_run_optimizations)

    result = run_this.run_pipeline(output_dir=output_dir, max_candidates=10, loc_num=3)

    assert result.candidate_result is candidate_result
    assert result.total_score_result is total_score_result
    assert result.optimization_result is optimization_result
    assert calls[0][0] == "build"
    assert calls[0][1]["max_candidates"] == 10
    assert calls[1] == (
        "score",
        {
            "input_path": candidate_result.output_path,
            "output_path": output_dir / "total_score.csv",
            "mlp_iter_num": run_this.MLP_ITER_NUM,
        },
    )
    assert calls[2][0] == "optimize"
    assert calls[2][1]["loc_num"] == 3
