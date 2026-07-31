"""Run the Yixing AED candidate pipeline and optimization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_candidates import build_whitelist_candidates
from build_conflict_pairs import (
    candidate_id_hash,
    default_conflict_pairs_path,
    default_indicator_cache_path,
    format_distance_threshold_m,
    load_or_create_conflict_pair_cache,
)
from calculate_total_score import calculate_total_scores
from config import (
    DEFAULT_LOC_NUM,
    DEFAULT_OPTIMIZATION_TIME_LIMIT_SECONDS,
    DEDUP_DISTANCE_THRESHOLD_M,
    MAPPED_POI_PATH,
    MLP_ITER_NUM,
    MLP_SELECTED_PATH,
    OPTIMIZATION_DISTANCE_THRESHOLD_M,
    SVR_SELECTED_PATH,
    OUTPUT_DIR,
    RANDOM_SEED,
    TOTAL_SCORE_PATH,
    XGB_SELECTED_PATH,
)
from Data import Data
from ModelBuilder import ModelBuilder
from yixing_optimizer_utils import read_csv_flexible


DEFAULT_INPUT = TOTAL_SCORE_PATH
DEFAULT_OUTPUT = XGB_SELECTED_PATH
SAMPLED_SUMMARY_NAME = "yixing_sampled_optimization_summary.csv"


@dataclass(frozen=True)
class OptimizationRunResult:
    """Selected-location outputs for all model score columns."""

    xgb_selected: object
    xgb_objective_value: float
    xgb_conflict_pairs: list[tuple[int, int]]
    mlp_selected: object
    mlp_objective_value: float
    mlp_conflict_pairs: list[tuple[int, int]]
    svr_selected: object
    svr_objective_value: float
    svr_conflict_pairs: list[tuple[int, int]]


@dataclass(frozen=True)
class SampledOptimizationRunResult:
    """Outputs produced by sampled optimization."""

    summary_df: pd.DataFrame
    sample_path: Path
    sample_metadata_path: Path
    summary_path: Path
    conflict_pairs_path: Path
    conflict_pairs_metadata_path: Path
    run_dir: Path
    selected_dir: Path


@dataclass(frozen=True)
class FullPipelineResult:
    """Outputs produced by the full pipeline controller."""

    candidate_result: object
    total_score_result: object
    optimization_result: OptimizationRunResult


def optimize_candidates(
    input_path=DEFAULT_INPUT,
    output_path=DEFAULT_OUTPUT,
    loc_num=DEFAULT_LOC_NUM,
    score_col="total_score_xgb",
    distance_threshold_m=OPTIMIZATION_DISTANCE_THRESHOLD_M,
    time_limit_seconds=DEFAULT_OPTIMIZATION_TIME_LIMIT_SECONDS,
    build_num=None,
    use_mlp=False,
    indicator_path=None,
    use_indicator_cache=True,
    sample_indices=None,
    sample_id=None,
    full_conflict_pairs=None,
):
    """Run one optimization pass against an already-scored candidate CSV."""

    data = Data()
    data.read_data(
        input_path,
        loc_num=loc_num,
        dist_limit_m=distance_threshold_m,
        build_num=build_num,
        score_col=score_col,
        indicator_path=indicator_path,
        use_indicator_cache=use_indicator_cache,
        sample_indices=sample_indices,
        sample_id=sample_id,
        full_conflict_pairs=full_conflict_pairs,
    )

    candidate_loc_id = range(data.build_num)
    model_handler = ModelBuilder()
    if use_mlp:
        model_handler.build_IP_mlp(
            data,
            candidate_loc_id,
            loc_num,
            time_limit_seconds=time_limit_seconds,
        )
        selected_indices = model_handler.deploy_decision_mlp
        obj_val = model_handler.obj_val_mlp
    else:
        model_handler.build_IP(
            data,
            candidate_loc_id,
            loc_num,
            time_limit_seconds=time_limit_seconds,
        )
        selected_indices = model_handler.deploy_decision
        obj_val = model_handler.obj_val

    rank_score_col = "total_score_mlp" if use_mlp and "total_score_mlp" in data.candidate_df.columns else score_col
    selected = data.candidate_df.loc[selected_indices].copy()
    selected["optimization_source_index"] = selected.index
    selected = selected.sort_values(rank_score_col, ascending=False)
    selected["optimization_rank"] = range(1, len(selected) + 1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False, encoding="utf-8-sig")
    return selected.reset_index(drop=True), obj_val, data.conflict_pairs


def run_optimizations(
    *,
    total_score_path: Path = TOTAL_SCORE_PATH,
    output_dir: Path = OUTPUT_DIR,
    loc_num: int = DEFAULT_LOC_NUM,
    distance_threshold_m: float = OPTIMIZATION_DISTANCE_THRESHOLD_M,
    time_limit_seconds: int = DEFAULT_OPTIMIZATION_TIME_LIMIT_SECONDS,
    build_num: int | None = None,
    indicator_path: Path | None = None,
    use_indicator_cache: bool = True,
) -> OptimizationRunResult:
    """Optimize selected locations once for XGB, MLP, and SVR scores."""

    output_dir = Path(output_dir)
    total_score_path = Path(total_score_path)
    if indicator_path is None and use_indicator_cache:
        indicator_path = default_indicator_cache_path(input_path=total_score_path, threshold_m=distance_threshold_m)
    indicator_path = Path(indicator_path) if indicator_path is not None else None
    xgb_selected, xgb_obj_val, xgb_conflict_pairs = optimize_candidates(
        input_path=total_score_path,
        output_path=output_dir / XGB_SELECTED_PATH.name,
        loc_num=loc_num,
        score_col="total_score_xgb",
        distance_threshold_m=distance_threshold_m,
        time_limit_seconds=time_limit_seconds,
        build_num=build_num,
        use_mlp=False,
        indicator_path=indicator_path,
        use_indicator_cache=use_indicator_cache,
    )
    mlp_selected, mlp_obj_val, mlp_conflict_pairs = optimize_candidates(
        input_path=total_score_path,
        output_path=output_dir / MLP_SELECTED_PATH.name,
        loc_num=loc_num,
        score_col="total_score_mlp",
        distance_threshold_m=distance_threshold_m,
        time_limit_seconds=time_limit_seconds,
        build_num=build_num,
        use_mlp=True,
        indicator_path=indicator_path,
        use_indicator_cache=use_indicator_cache,
    )
    svr_selected, svr_obj_val, svr_conflict_pairs = optimize_candidates(
        input_path=total_score_path,
        output_path=output_dir / SVR_SELECTED_PATH.name,
        loc_num=loc_num,
        score_col="total_score_svr",
        distance_threshold_m=distance_threshold_m,
        time_limit_seconds=time_limit_seconds,
        build_num=build_num,
        use_mlp=False,
        indicator_path=indicator_path,
        use_indicator_cache=use_indicator_cache,
    )
    return OptimizationRunResult(
        xgb_selected=xgb_selected,
        xgb_objective_value=xgb_obj_val,
        xgb_conflict_pairs=xgb_conflict_pairs,
        mlp_selected=mlp_selected,
        mlp_objective_value=mlp_obj_val,
        mlp_conflict_pairs=mlp_conflict_pairs,
        svr_selected=svr_selected,
        svr_objective_value=svr_obj_val,
        svr_conflict_pairs=svr_conflict_pairs,
    )


def run_sampled_optimizations(
    *,
    total_score_path: Path = TOTAL_SCORE_PATH,
    output_dir: Path = OUTPUT_DIR,
    loc_num: int = DEFAULT_LOC_NUM,
    distance_threshold_m: float = OPTIMIZATION_DISTANCE_THRESHOLD_M,
    time_limit_seconds: int = DEFAULT_OPTIMIZATION_TIME_LIMIT_SECONDS,
    sample_size: int = 2_000,
    sample_count: int = 10,
    sample_seed: int = RANDOM_SEED,
    sample_path: Path | None = None,
    force_resample: bool = False,
    conflict_pairs_path: Path | None = None,
    force_conflict_pairs: bool = False,
    run_dir: Path | None = None,
) -> SampledOptimizationRunResult:
    """Run XGB, MLP, and SVR optimization for each saved random sample."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_score_path = Path(total_score_path)
    run_dir = default_sampled_run_dir(
        output_dir=output_dir,
        sample_size=sample_size,
        sample_count=sample_count,
        sample_seed=sample_seed,
        distance_threshold_m=distance_threshold_m,
        loc_num=loc_num,
    ) if run_dir is None else Path(run_dir)
    selected_dir = run_dir / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    candidate_df = read_csv_flexible(total_score_path)
    sample_path = default_sample_path(
        output_dir=output_dir,
        sample_size=sample_size,
        sample_count=sample_count,
        sample_seed=sample_seed,
    ) if sample_path is None else Path(sample_path)
    sample_matrix, metadata_path = load_or_create_candidate_samples(
        candidate_df,
        source_path=total_score_path,
        sample_path=sample_path,
        sample_size=sample_size,
        sample_count=sample_count,
        sample_seed=sample_seed,
        force_resample=force_resample,
    )
    conflict_pairs_path = default_conflict_pairs_path(
        output_dir=output_dir,
        threshold_m=distance_threshold_m,
    ) if conflict_pairs_path is None else Path(conflict_pairs_path)
    full_conflict_pairs, conflict_pairs_metadata_path = load_or_create_conflict_pair_cache(
        candidate_df,
        source_path=total_score_path,
        conflict_pairs_path=conflict_pairs_path,
        threshold_m=distance_threshold_m,
        force_rebuild=force_conflict_pairs,
    )

    model_specs = [
        ("xgb", "total_score_xgb", False),
        ("mlp", "total_score_mlp", True),
        ("svr", "total_score_svr", False),
    ]
    records: list[dict[str, object]] = []
    for sample_zero_index, sample_indices in enumerate(sample_matrix):
        sample_id = sample_zero_index + 1
        for model_name, score_col, use_mlp in model_specs:
            output_path = selected_dir / f"yixing_selected_locations_{model_name}_sample{sample_id:02d}.csv"
            selected, objective_value, conflict_pairs = optimize_candidates(
                input_path=total_score_path,
                output_path=output_path,
                loc_num=loc_num,
                score_col=score_col,
                distance_threshold_m=distance_threshold_m,
                time_limit_seconds=time_limit_seconds,
                build_num=None,
                use_mlp=use_mlp,
                indicator_path=None,
                use_indicator_cache=False,
                sample_indices=sample_indices,
                sample_id=sample_id,
                full_conflict_pairs=full_conflict_pairs,
            )
            records.append(
                {
                    "sample_id": sample_id,
                    "model": model_name,
                    "score_col": score_col,
                    "objective_value": float(objective_value),
                    "selected_count": int(len(selected)),
                    "conflict_pair_count": int(len(conflict_pairs)),
                    "unique_candidate_count": int(len(np.unique(sample_indices))),
                    "sample_size": int(len(sample_indices)),
                    "distance_threshold_m": float(distance_threshold_m),
                    "loc_num": int(loc_num),
                    "full_conflict_pair_count": int(len(full_conflict_pairs)),
                    "sample_path": str(sample_path),
                    "conflict_pairs_path": str(conflict_pairs_path),
                    "run_dir": str(run_dir),
                    "output_path": str(output_path),
                }
            )

    summary_df = pd.DataFrame(records)
    summary_path = run_dir / SAMPLED_SUMMARY_NAME
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return SampledOptimizationRunResult(
        summary_df=summary_df,
        sample_path=sample_path,
        sample_metadata_path=metadata_path,
        summary_path=summary_path,
        conflict_pairs_path=conflict_pairs_path,
        conflict_pairs_metadata_path=conflict_pairs_metadata_path,
        run_dir=run_dir,
        selected_dir=selected_dir,
    )


def generate_candidate_samples(
    candidate_count: int,
    *,
    sample_size: int = 2_000,
    sample_count: int = 10,
    sample_seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Generate sample_count no-replacement samples; groups may overlap."""

    candidate_count = int(candidate_count)
    sample_size = int(sample_size)
    sample_count = int(sample_count)
    if sample_size <= 0 or sample_count <= 0:
        raise ValueError("sample_size and sample_count must be positive.")
    if sample_size > candidate_count:
        raise ValueError("sample_size cannot exceed candidate_count when sampling without replacement.")

    rng = np.random.default_rng(int(sample_seed))
    rows = [rng.choice(candidate_count, size=sample_size, replace=False) for _ in range(sample_count)]
    return np.vstack(rows).astype(np.int64)


def default_sample_path(
    *,
    output_dir: Path = OUTPUT_DIR,
    sample_size: int = 2_000,
    sample_count: int = 10,
    sample_seed: int = RANDOM_SEED,
) -> Path:
    """Return the default saved candidate-sample matrix path."""

    return Path(output_dir) / "samples" / f"candidate_sample_{int(sample_size)}x{int(sample_count)}_seed{int(sample_seed)}.npy"



def default_sampled_run_dir(
    *,
    output_dir: Path = OUTPUT_DIR,
    sample_size: int = 2_000,
    sample_count: int = 10,
    sample_seed: int = RANDOM_SEED,
    distance_threshold_m: float = OPTIMIZATION_DISTANCE_THRESHOLD_M,
    loc_num: int = DEFAULT_LOC_NUM,
) -> Path:
    """Return the default folder for one sampled optimization batch."""

    threshold_token = format_distance_threshold_m(distance_threshold_m)
    return (
        Path(output_dir)
        / "runs"
        / f"sampled_{int(sample_size)}x{int(sample_count)}_seed{int(sample_seed)}_d{threshold_token}_loc{int(loc_num)}"
    )
def sample_metadata_path(sample_path: Path) -> Path:
    """Return the JSON metadata path paired with a sample matrix."""

    return Path(sample_path).with_suffix(".json")


def load_or_create_candidate_samples(
    candidate_df: pd.DataFrame,
    *,
    source_path: Path,
    sample_path: Path,
    sample_size: int = 2_000,
    sample_count: int = 10,
    sample_seed: int = RANDOM_SEED,
    force_resample: bool = False,
) -> tuple[np.ndarray, Path]:
    """Load a valid saved sample matrix or create a new one."""

    sample_path = Path(sample_path)
    metadata_path = sample_metadata_path(sample_path)
    if sample_path.exists() and metadata_path.exists() and not force_resample:
        sample_matrix = load_valid_candidate_samples(
            candidate_df,
            source_path=source_path,
            sample_path=sample_path,
            sample_size=sample_size,
            sample_count=sample_count,
            sample_seed=sample_seed,
        )
        if sample_matrix is None:
            raise ValueError(f"Saved candidate sample does not match current input: {sample_path}")
        return sample_matrix, metadata_path

    sample_matrix = generate_candidate_samples(
        len(candidate_df),
        sample_size=sample_size,
        sample_count=sample_count,
        sample_seed=sample_seed,
    )
    write_candidate_samples(
        sample_matrix,
        candidate_df,
        source_path=source_path,
        sample_path=sample_path,
        sample_seed=sample_seed,
        sample_size=sample_size,
        sample_count=sample_count,
    )
    return sample_matrix, metadata_path


def load_valid_candidate_samples(
    candidate_df: pd.DataFrame,
    *,
    source_path: Path,
    sample_path: Path,
    sample_size: int,
    sample_count: int,
    sample_seed: int,
) -> np.ndarray | None:
    """Load a saved sample matrix only when metadata matches current candidates."""

    sample_path = Path(sample_path)
    metadata_path = sample_metadata_path(sample_path)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    expected = _sample_metadata(
        candidate_df,
        source_path=source_path,
        sample_seed=sample_seed,
        sample_size=sample_size,
        sample_count=sample_count,
    )
    fields = [
        "source_total_score_path",
        "candidate_count",
        "candidate_id_hash",
        "sample_size",
        "sample_count",
        "seed",
        "replace_within_sample",
    ]
    for field in fields:
        if metadata.get(field) != expected.get(field):
            return None

    sample_matrix = np.load(sample_path)
    if sample_matrix.shape != (int(sample_count), int(sample_size)):
        return None
    if sample_matrix.dtype.kind not in {"i", "u"}:
        return None
    if sample_matrix.size and (sample_matrix.min() < 0 or sample_matrix.max() >= len(candidate_df)):
        return None
    if any(len(np.unique(row)) != len(row) for row in sample_matrix):
        return None
    return sample_matrix.astype(np.int64, copy=False)


def write_candidate_samples(
    sample_matrix: np.ndarray,
    candidate_df: pd.DataFrame,
    *,
    source_path: Path,
    sample_path: Path,
    sample_seed: int,
    sample_size: int | None = None,
    sample_count: int | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    """Write a candidate sample matrix and metadata."""

    sample_path = Path(sample_path)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_matrix = np.asarray(sample_matrix, dtype=np.int64)
    np.save(sample_path, sample_matrix)
    metadata = _sample_metadata(
        candidate_df,
        source_path=source_path,
        sample_seed=sample_seed,
        sample_size=sample_matrix.shape[1] if sample_size is None else sample_size,
        sample_count=sample_matrix.shape[0] if sample_count is None else sample_count,
        include_created_at=True,
    )
    metadata_path = sample_metadata_path(sample_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return sample_path, metadata_path, metadata


def _sample_metadata(
    candidate_df: pd.DataFrame,
    *,
    source_path: Path,
    sample_seed: int,
    sample_size: int,
    sample_count: int,
    include_created_at: bool = False,
) -> dict[str, object]:
    metadata = {
        "source_total_score_path": str(Path(source_path).resolve()),
        "candidate_count": int(len(candidate_df)),
        "candidate_id_hash": candidate_id_hash(candidate_df.reset_index(drop=True)),
        "sample_size": int(sample_size),
        "sample_count": int(sample_count),
        "seed": int(sample_seed),
        "replace_within_sample": False,
    }
    if include_created_at:
        metadata["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    return metadata


def run_pipeline(
    *,
    input_path: Path = MAPPED_POI_PATH,
    output_dir: Path = OUTPUT_DIR,
    dedup_distance_m: float = DEDUP_DISTANCE_THRESHOLD_M,
    mlp_iter_num: int = MLP_ITER_NUM,
    loc_num: int = DEFAULT_LOC_NUM,
    distance_threshold_m: float = OPTIMIZATION_DISTANCE_THRESHOLD_M,
    time_limit_seconds: int = DEFAULT_OPTIMIZATION_TIME_LIMIT_SECONDS,
    build_num: int | None = None,
    indicator_path: Path | None = None,
    use_indicator_cache: bool = True,
) -> FullPipelineResult:
    """Run candidate generation, total-score calculation, and all optimization passes."""

    candidate_result = build_whitelist_candidates(
        input_path=input_path,
        output_dir=output_dir,
        dedup_distance_m=dedup_distance_m,
    )
    total_score_path = Path(output_dir) / TOTAL_SCORE_PATH.name
    total_score_result = calculate_total_scores(
        input_path=candidate_result.output_path,
        output_path=total_score_path,
        mlp_iter_num=mlp_iter_num,
    )
    optimization_result = run_optimizations(
        total_score_path=total_score_result.output_path,
        output_dir=output_dir,
        loc_num=loc_num,
        distance_threshold_m=distance_threshold_m,
        time_limit_seconds=time_limit_seconds,
        build_num=build_num,
        indicator_path=indicator_path,
        use_indicator_cache=use_indicator_cache,
    )
    return FullPipelineResult(
        candidate_result=candidate_result,
        total_score_result=total_score_result,
        optimization_result=optimization_result,
    )


def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=MAPPED_POI_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dedup-distance-m", type=float, default=DEDUP_DISTANCE_THRESHOLD_M)
    parser.add_argument("--mlp-iter-num", type=int, default=MLP_ITER_NUM)
    parser.add_argument("--loc-num", type=int, default=DEFAULT_LOC_NUM)
    parser.add_argument("--distance-threshold-m", type=float, default=OPTIMIZATION_DISTANCE_THRESHOLD_M)
    parser.add_argument("--time-limit-seconds", type=int, default=DEFAULT_OPTIMIZATION_TIME_LIMIT_SECONDS)
    parser.add_argument("--build-num", type=int, default=None)
    parser.add_argument("--indicator-path", type=Path, default=None)
    parser.add_argument("--no-indicator-cache", action="store_true")


def add_optimize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--loc-num", type=int, default=DEFAULT_LOC_NUM)
    parser.add_argument("--score-col", default="total_score_xgb")
    parser.add_argument("--distance-threshold-m", type=float, default=OPTIMIZATION_DISTANCE_THRESHOLD_M)
    parser.add_argument("--time-limit-seconds", type=int, default=DEFAULT_OPTIMIZATION_TIME_LIMIT_SECONDS)
    parser.add_argument("--build-num", type=int, default=None)
    parser.add_argument("--indicator-path", type=Path, default=None)
    parser.add_argument("--no-indicator-cache", action="store_true")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--sample-path", type=Path, default=None)
    parser.add_argument("--force-resample", action="store_true")
    parser.add_argument("--conflict-pairs-path", type=Path, default=None)
    parser.add_argument("--force-conflict-pairs", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    score_group = parser.add_mutually_exclusive_group()
    score_group.add_argument("--mlp", action="store_true")
    score_group.add_argument("--svr", action="store_true")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Yixing AED optimizer pipeline.")
    add_pipeline_args(parser)
    subparsers = parser.add_subparsers(dest="command")

    pipeline_parser = subparsers.add_parser("pipeline", help="Run the full candidate, scoring, and optimization pipeline.")
    add_pipeline_args(pipeline_parser)

    optimize_parser = subparsers.add_parser("optimize", help="Run optimization only against total_score.csv.")
    add_optimize_args(optimize_parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    if args.command == "optimize":
        if args.sample_size is not None:
            result = run_sampled_optimizations(
                total_score_path=args.input,
                output_dir=args.output_dir,
                loc_num=args.loc_num,
                distance_threshold_m=args.distance_threshold_m,
                time_limit_seconds=args.time_limit_seconds,
                sample_size=args.sample_size,
                sample_count=args.sample_count,
                sample_seed=args.sample_seed,
                sample_path=args.sample_path,
                force_resample=args.force_resample,
                conflict_pairs_path=args.conflict_pairs_path,
                force_conflict_pairs=args.force_conflict_pairs,
                run_dir=args.run_dir,
            )
            print("sample_path =", result.sample_path)
            print("sample_metadata_path =", result.sample_metadata_path)
            print("run_dir =", result.run_dir)
            print("selected_dir =", result.selected_dir)
            print("summary_output =", result.summary_path)
            print("conflict_pairs_path =", result.conflict_pairs_path)
            print("conflict_pairs_metadata_path =", result.conflict_pairs_metadata_path)
            print("sampled_optimization_rows =", len(result.summary_df))
            return result

        score_col = args.score_col
        if args.mlp and args.score_col == "total_score_xgb":
            score_col = "total_score_mlp"
        elif args.svr and args.score_col == "total_score_xgb":
            score_col = "total_score_svr"
        output_path = args.output
        if output_path is None:
            if args.mlp:
                output_path = MLP_SELECTED_PATH
            elif args.svr:
                output_path = SVR_SELECTED_PATH
            else:
                output_path = XGB_SELECTED_PATH
        selected_df, objective_value, conflict_pairs = optimize_candidates(
            input_path=args.input,
            output_path=output_path,
            loc_num=args.loc_num,
            score_col=score_col,
            distance_threshold_m=args.distance_threshold_m,
            time_limit_seconds=args.time_limit_seconds,
            build_num=args.build_num,
            use_mlp=args.mlp,
            indicator_path=args.indicator_path,
            use_indicator_cache=not args.no_indicator_cache,
        )
        print("obj_val =", objective_value)
        print("conflict_pair_count =", len(conflict_pairs))
        print("deployment_decision =", selected_df["optimization_source_index"].tolist())
        print("output =", output_path)
        return selected_df, objective_value, conflict_pairs

    result = run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        dedup_distance_m=args.dedup_distance_m,
        mlp_iter_num=args.mlp_iter_num,
        loc_num=args.loc_num,
        distance_threshold_m=args.distance_threshold_m,
        time_limit_seconds=args.time_limit_seconds,
        build_num=args.build_num,
        indicator_path=args.indicator_path,
        use_indicator_cache=not args.no_indicator_cache,
    )
    print("candidate_count =", len(result.candidate_result.candidates))
    print("scored_candidate_count =", len(result.total_score_result.scored_candidates))
    print("xgb_selected_count =", len(result.optimization_result.xgb_selected))
    print("mlp_selected_count =", len(result.optimization_result.mlp_selected))
    print("svr_selected_count =", len(result.optimization_result.svr_selected))
    print("total_score_output =", result.total_score_result.output_path)
    return result


if __name__ == "__main__":
    main()