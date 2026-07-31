from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import shutil
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OPTIMIZOR_DIR = PROJECT_ROOT / "notebooks" / "Optimizor"
sys.path.insert(0, str(OPTIMIZOR_DIR))

import build_candidates as bc
import calculate_total_score as cts
import build_conflict_pairs as conflict_module
import Data as data_module
import run_this
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
                "osm_tag": "hospital",
            },
            {
                "candidate_id": "yx_b",
                "name": "candidate b",
                "longitude": 119.86,
                "latitude": 31.3,
                "osm_tag": "bank",
            },
        ]
    )
    return yxu.add_h3_index(df)


def test_attach_candidate_component_scores_uses_osm_tag_only():
    candidates = candidate_frame().iloc[[0]].copy().reset_index(drop=True)
    h3_id = str(candidates.loc[0, "h3_l7"])
    h3_feature_scores = pd.DataFrame(
        {
            "id": [h3_id],
            "hospital": [1.25],
            "bank": [2.75],
        }
    )

    scored = cts.attach_candidate_component_scores(
        candidates,
        h3_feature_scores,
        score_col="point_shap_score_test",
    )

    assert np.isclose(scored.loc[0, "point_shap_score_test"], 1.25)


def test_build_candidate_points_filters_osm_tag_and_prefers_wgs84_coordinates():
    raw = pd.DataFrame(
        {
            "id": ["src_hospital", "src_bank", "src_restaurant"],
            "name": ["hospital", "bank", "restaurant"],
            "longitude": [119.9, 119.95, 120.0],
            "latitude": [31.4, 31.45, 31.5],
            "wgs84_经度": [119.8, 119.85, 119.9],
            "wgs84_纬度": [31.3, 31.35, 31.4],
            "osm_tag": [" Hospital ", "bank", "restaurant"],
        }
    )

    candidates, summary, _, excluded = bc.build_candidate_points(raw, dedup_distance_m=0.0)

    assert candidates["osm_tag"].tolist() == ["bank", "hospital"] or candidates["osm_tag"].tolist() == ["hospital", "bank"]
    hospital = candidates.loc[candidates["osm_tag"] == "hospital"].iloc[0]
    assert np.isclose(hospital["longitude"], 119.8)
    assert np.isclose(hospital["latitude"], 31.3)
    assert np.isclose(hospital["source_longitude"], 119.9)
    assert summary.whitelist_candidate_count == 2
    assert set(excluded["osm_tag"]) == {"restaurant"}
    bc.validate_candidates(candidates)


def test_deduplicate_candidate_points_tracks_source_rows():
    raw = pd.DataFrame(
        {
            "id": ["src_a", "src_b"],
            "name": ["poi a", "poi b"],
            "longitude": [119.8, 119.800001],
            "latitude": [31.3, 31.300001],
            "osm_tag": ["hospital", "hospital"],
        }
    )

    candidates, summary, _, _ = bc.build_candidate_points(raw, dedup_distance_m=10.0)

    assert len(candidates) == 1
    assert summary.whitelist_candidate_count == 2
    assert summary.deduplicated_candidate_count == 1
    assert candidates.loc[0, "source_row_count"] == 2
    assert candidates.loc[0, "merged_source_ids"] == "src_a|src_b"
    assert candidates["candidate_id"].is_unique


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
            "point_shap_score_xgb",
            "point_shap_score_mlp",
            "point_shap_score_svr",
            "predicted_ohca_xgb",
            "predicted_ohca_mlp",
            "predicted_ohca_svr",
            "center_lat",
            "center_lon",
        }
        assert result.output_path == output_path
        assert expected_columns.issubset(written.columns)
        assert omitted_columns.isdisjoint(written.columns)
        assert written.columns[-6:].tolist() == [
            "total_score_xgb",
            "total_score_mlp",
            "total_score_svr",
            "score_rank_xgb",
            "score_rank_mlp",
            "score_rank_svr",
        ]
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
    indicator = conflict_module.build_indicator_matrix(conflict_candidate_frame(), threshold_m=20.0)

    assert indicator.dtype == np.bool_
    assert indicator.shape == (3, 3)
    assert not indicator.diagonal().any()
    assert np.array_equal(indicator, indicator.T)
    assert indicator[0, 1]
    assert not indicator[0, 2]
    assert not indicator[1, 2]


def test_sparse_conflict_pair_cache_round_trip_and_sample_filter():
    candidate_df = conflict_candidate_frame()
    temp_dir = OPTIMIZOR_DIR / "tests" / "_conflict_pair_cache"
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    source_path = temp_dir / "total_score.csv"
    conflict_pairs_path = temp_dir / "conflict_pairs_20m.npy"
    candidate_df.to_csv(source_path, index=False, encoding="utf-8-sig")

    try:
        _, metadata_path, pairs, metadata = conflict_module.write_conflict_pair_cache(
            candidate_df,
            conflict_pairs_path=conflict_pairs_path,
            threshold_m=20.0,
            input_path=source_path,
        )
        loaded = conflict_module.load_valid_conflict_pair_cache(
            candidate_df,
            conflict_pairs_path=conflict_pairs_path,
            threshold_m=20.0,
            input_path=source_path,
        )
        sample_pairs = conflict_module.filter_conflict_pairs_for_sample(
            loaded,
            np.array([1, 0]),
            candidate_count=len(candidate_df),
        )

        assert conflict_pairs_path.exists()
        assert metadata_path.exists()
        assert pairs.shape == (1, 2)
        assert np.array_equal(loaded, pairs)
        assert metadata["cache_type"] == "sparse_conflict_pairs"
        assert sample_pairs == [(0, 1)]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_data_read_dataframe_uses_valid_indicator_cache_and_slices_build_num():
    candidate_df = conflict_candidate_frame()
    indicator_path = OPTIMIZOR_DIR / "tests" / "_indicator_cache_test.npy"
    try:
        conflict_module.write_indicator_cache(candidate_df, indicator_path=indicator_path, threshold_m=20.0)

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
        conflict_module.indicator_metadata_path(indicator_path).unlink(missing_ok=True)


def test_data_read_dataframe_falls_back_when_indicator_cache_missing(monkeypatch):
    calls = []

    def fake_build_distance_conflict_pairs(candidate_df, *, threshold_m, projected_crs=conflict_module.PROJECTED_CRS):
        calls.append((len(candidate_df), threshold_m, projected_crs))
        return [(0, 1)]

    monkeypatch.setattr(conflict_module, "build_distance_conflict_pairs", fake_build_distance_conflict_pairs)
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
    assert calls == [(3, 20.0, conflict_module.PROJECTED_CRS)]


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

    expected_indicator_path = output_dir / "cache" / "indicator_i_j_20m.npy"

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

    result = run_this.run_pipeline(output_dir=output_dir, loc_num=3)

    assert result.candidate_result is candidate_result
    assert result.total_score_result is total_score_result
    assert result.optimization_result is optimization_result
    assert calls[0][0] == "build"
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

def test_generate_candidate_samples_are_reproducible_and_unique_within_rows():
    samples_a = run_this.generate_candidate_samples(25, sample_size=20, sample_count=2, sample_seed=277)
    samples_b = run_this.generate_candidate_samples(25, sample_size=20, sample_count=2, sample_seed=277)

    assert samples_a.shape == (2, 20)
    assert np.array_equal(samples_a, samples_b)
    assert all(len(np.unique(row)) == 20 for row in samples_a)
    assert len(set(samples_a[0].tolist()) & set(samples_a[1].tolist())) > 0

def test_sampled_default_artifact_paths_are_grouped_under_output_subdirs():
    output_dir = OPTIMIZOR_DIR / "tests" / "_layout_base"

    assert run_this.default_sample_path(
        output_dir=output_dir,
        sample_size=2000,
        sample_count=10,
        sample_seed=277,
    ) == output_dir / "samples" / "candidate_sample_2000x10_seed277.npy"
    assert conflict_module.default_conflict_pairs_path(
        output_dir=output_dir,
        threshold_m=50.0,
    ) == output_dir / "cache" / "conflict_pairs_50m.npy"
    assert run_this.default_sampled_run_dir(
        output_dir=output_dir,
        sample_size=2000,
        sample_count=10,
        sample_seed=277,
        distance_threshold_m=50.0,
        loc_num=5,
    ) == output_dir / "runs" / "sampled_2000x10_seed277_d50m_loc5"


def test_candidate_sample_file_and_metadata_round_trip():
    candidate_df = pd.DataFrame(
        {
            "candidate_id": [f"yx_{index}" for index in range(30)],
            "longitude": np.linspace(119.8, 119.9, 30),
            "latitude": np.linspace(31.3, 31.4, 30),
            "total_score_xgb": np.linspace(1.0, 2.0, 30),
        }
    )
    temp_dir = OPTIMIZOR_DIR / "tests" / "_candidate_sample_round_trip"
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    source_path = temp_dir / "total_score.csv"
    sample_path = temp_dir / "candidate_sample_5x3_seed277.npy"
    candidate_df.to_csv(source_path, index=False, encoding="utf-8-sig")

    try:
        samples, metadata_path = run_this.load_or_create_candidate_samples(
            candidate_df,
            source_path=source_path,
            sample_path=sample_path,
            sample_size=5,
            sample_count=3,
            sample_seed=277,
        )
        loaded = run_this.load_valid_candidate_samples(
            candidate_df,
            source_path=source_path,
            sample_path=sample_path,
            sample_size=5,
            sample_count=3,
            sample_seed=277,
        )
        metadata = pd.read_json(metadata_path, typ="series")

        assert sample_path.exists()
        assert metadata_path.exists()
        assert np.array_equal(samples, loaded)
        assert samples.shape == (3, 5)
        assert bool(metadata["replace_within_sample"]) is False
        assert metadata["candidate_id_hash"] == conflict_module.candidate_id_hash(candidate_df)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_data_read_dataframe_sample_subset_adds_trace_columns():
    data = data_module.Data().read_dataframe(
        conflict_candidate_frame(),
        loc_num=1,
        dist_limit_m=20.0,
        score_col="total_score",
        sample_indices=np.array([2, 0]),
        sample_id=7,
    )

    assert data.build_num == 2
    assert data.candidate_df["sample_id"].tolist() == [7, 7]
    assert data.candidate_df["source_candidate_index"].tolist() == [2, 0]
    assert data.candidate_df["candidate_id"].tolist() == ["far_c", "near_a"]

def test_data_read_dataframe_filters_full_conflict_pairs_for_sample():
    data = data_module.Data().read_dataframe(
        conflict_candidate_frame(),
        loc_num=1,
        dist_limit_m=20.0,
        score_col="total_score",
        sample_indices=np.array([1, 0]),
        sample_id=3,
        full_conflict_pairs=np.array([[0, 1], [0, 2]], dtype=np.int64),
    )

    assert data.conflict_pair_cache_used is True
    assert data.indicator_cache_used is False
    assert data.conflict_pairs == [(0, 1)]
    assert data.has_distance_conflict(0, 1) is True


def test_run_sampled_optimizations_writes_three_model_outputs_per_sample(monkeypatch):
    candidate_df = pd.DataFrame(
        {
            "candidate_id": [f"yx_{index}" for index in range(8)],
            "longitude": np.linspace(119.8, 119.9, 8),
            "latitude": np.linspace(31.3, 31.4, 8),
            "total_score_xgb": np.linspace(8.0, 1.0, 8),
            "total_score_mlp": np.linspace(1.0, 8.0, 8),
            "total_score_svr": np.linspace(4.0, 5.0, 8),
        }
    )
    temp_dir = OPTIMIZOR_DIR / "tests" / "_sampled_optimization"
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    total_score_path = temp_dir / "total_score.csv"
    candidate_df.to_csv(total_score_path, index=False, encoding="utf-8-sig")
    calls = []
    cache_calls = []
    full_conflict_pairs = np.array([[0, 1], [1, 2], [4, 5]], dtype=np.int64)

    def fake_load_or_create_conflict_pair_cache(*args, **kwargs):
        cache_calls.append((args, kwargs))
        return full_conflict_pairs, temp_dir / "cache" / "conflict_pairs_50m.json"

    def fake_optimize_candidates(**kwargs):
        calls.append(kwargs)
        selected = pd.DataFrame({"optimization_source_index": [0], "candidate_id": ["yx_fake"]})
        selected.to_csv(kwargs["output_path"], index=False, encoding="utf-8-sig")
        return selected, 10.0, [(0, 1), (1, 2)]
    monkeypatch.setattr(run_this, "load_or_create_conflict_pair_cache", fake_load_or_create_conflict_pair_cache)
    monkeypatch.setattr(run_this, "optimize_candidates", fake_optimize_candidates)

    try:
        result = run_this.run_sampled_optimizations(
            total_score_path=total_score_path,
            output_dir=temp_dir,
            loc_num=2,
            distance_threshold_m=50.0,
            time_limit_seconds=5,
            sample_size=3,
            sample_count=2,
            sample_seed=277,
        )

        assert len(cache_calls) == 1
        assert len(cache_calls[0][0][0]) == len(candidate_df)
        assert cache_calls[0][1]["threshold_m"] == 50.0
        assert cache_calls[0][1]["conflict_pairs_path"] == temp_dir / "cache" / "conflict_pairs_50m.npy"
        assert len(calls) == 6
        assert result.run_dir == temp_dir / "runs" / "sampled_3x2_seed277_d50m_loc2"
        assert result.selected_dir == result.run_dir / "selected"
        assert result.summary_path == result.run_dir / "yixing_sampled_optimization_summary.csv"
        assert result.sample_path == temp_dir / "samples" / "candidate_sample_3x2_seed277.npy"
        assert result.summary_path.exists()
        assert result.summary_df.shape[0] == 6
        assert set(result.summary_df["model"]) == {"xgb", "mlp", "svr"}
        assert calls[0]["sample_id"] == 1
        assert calls[3]["sample_id"] == 2
        assert all(call["use_indicator_cache"] is False for call in calls)
        assert all(call["full_conflict_pairs"] is full_conflict_pairs for call in calls)
        assert all(len(np.unique(call["sample_indices"])) == 3 for call in calls)
        assert (result.selected_dir / "yixing_selected_locations_xgb_sample01.csv").exists()
        assert (result.selected_dir / "yixing_selected_locations_mlp_sample02.csv").exists()
        assert not (temp_dir / "yixing_selected_locations_xgb_sample01.csv").exists()
        assert result.summary_df["sample_path"].iloc[0] == str(result.sample_path)
        assert result.summary_df["output_path"].str.contains("selected").all()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)