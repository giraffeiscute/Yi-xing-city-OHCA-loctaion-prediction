"""Run the full Yixing Python pipeline from candidates to optimization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from build_candidates import build_whitelist_candidates, parse_candidate_limit
from calculate_total_score import calculate_total_scores
from config import (
    DEFAULT_LOC_NUM,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_OPTIMIZATION_TIME_LIMIT_SECONDS,
    DEDUP_DISTANCE_THRESHOLD_M,
    MAPPED_POI_PATH,
    MLP_ITER_NUM,
    MLP_SELECTED_PATH,
    OPTIMIZATION_DISTANCE_THRESHOLD_M,
    OUTPUT_DIR,
    RANDOM_SEED,
    TOTAL_SCORE_PATH,
    XGB_SELECTED_PATH,
)
from run_this import optimize_candidates


@dataclass(frozen=True)
class OptimizationRunResult:
    """Selected-location outputs for both model score columns."""

    xgb_selected: object
    xgb_objective_value: float
    xgb_conflict_pairs: list[tuple[int, int]]
    mlp_selected: object
    mlp_objective_value: float
    mlp_conflict_pairs: list[tuple[int, int]]


@dataclass(frozen=True)
class FullPipelineResult:
    """Outputs produced by the full pipeline controller."""

    candidate_result: object
    total_score_result: object
    optimization_result: OptimizationRunResult


def run_optimizations(
    *,
    total_score_path: Path = TOTAL_SCORE_PATH,
    output_dir: Path = OUTPUT_DIR,
    loc_num: int = DEFAULT_LOC_NUM,
    distance_threshold_m: float = OPTIMIZATION_DISTANCE_THRESHOLD_M,
    time_limit_seconds: int = DEFAULT_OPTIMIZATION_TIME_LIMIT_SECONDS,
    build_num: int | None = None,
) -> OptimizationRunResult:
    """Optimize selected locations once for XGB scores and once for MLP scores."""

    output_dir = Path(output_dir)
    xgb_selected, xgb_obj_val, xgb_conflict_pairs = optimize_candidates(
        input_path=Path(total_score_path),
        output_path=output_dir / XGB_SELECTED_PATH.name,
        loc_num=loc_num,
        score_col="total_score_xgb",
        distance_threshold_m=distance_threshold_m,
        time_limit_seconds=time_limit_seconds,
        build_num=build_num,
        use_mlp=False,
    )
    mlp_selected, mlp_obj_val, mlp_conflict_pairs = optimize_candidates(
        input_path=Path(total_score_path),
        output_path=output_dir / MLP_SELECTED_PATH.name,
        loc_num=loc_num,
        score_col="total_score_mlp",
        distance_threshold_m=distance_threshold_m,
        time_limit_seconds=time_limit_seconds,
        build_num=build_num,
        use_mlp=True,
    )
    return OptimizationRunResult(
        xgb_selected=xgb_selected,
        xgb_objective_value=xgb_obj_val,
        xgb_conflict_pairs=xgb_conflict_pairs,
        mlp_selected=mlp_selected,
        mlp_objective_value=mlp_obj_val,
        mlp_conflict_pairs=mlp_conflict_pairs,
    )


def run_pipeline(
    *,
    input_path: Path = MAPPED_POI_PATH,
    output_dir: Path = OUTPUT_DIR,
    run_boundary_filter: bool = True,
    dedup_distance_m: float = DEDUP_DISTANCE_THRESHOLD_M,
    max_candidates: int | None = DEFAULT_MAX_CANDIDATES,
    random_seed: int = RANDOM_SEED,
    mlp_iter_num: int = MLP_ITER_NUM,
    loc_num: int = DEFAULT_LOC_NUM,
    distance_threshold_m: float = OPTIMIZATION_DISTANCE_THRESHOLD_M,
    time_limit_seconds: int = DEFAULT_OPTIMIZATION_TIME_LIMIT_SECONDS,
    build_num: int | None = None,
) -> FullPipelineResult:
    """Run candidate generation, total scoring, and both optimization passes."""

    candidate_result = build_whitelist_candidates(
        input_path=input_path,
        output_dir=output_dir,
        run_boundary_filter=run_boundary_filter,
        dedup_distance_m=dedup_distance_m,
        max_candidates=max_candidates,
        random_seed=random_seed,
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
    )
    return FullPipelineResult(
        candidate_result=candidate_result,
        total_score_result=total_score_result,
        optimization_result=optimization_result,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Yixing optimizer pipeline.")
    parser.add_argument("--input", type=Path, default=MAPPED_POI_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-boundary-filter", action="store_true")
    parser.add_argument("--dedup-distance-m", type=float, default=DEDUP_DISTANCE_THRESHOLD_M)
    parser.add_argument("--max-candidates", type=parse_candidate_limit, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--mlp-iter-num", type=int, default=MLP_ITER_NUM)
    parser.add_argument("--loc-num", type=int, default=DEFAULT_LOC_NUM)
    parser.add_argument("--distance-threshold-m", type=float, default=OPTIMIZATION_DISTANCE_THRESHOLD_M)
    parser.add_argument("--time-limit-seconds", type=int, default=DEFAULT_OPTIMIZATION_TIME_LIMIT_SECONDS)
    parser.add_argument("--build-num", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> FullPipelineResult:
    args = parse_args(argv)
    result = run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        run_boundary_filter=not args.no_boundary_filter,
        dedup_distance_m=args.dedup_distance_m,
        max_candidates=args.max_candidates,
        random_seed=args.random_seed,
        mlp_iter_num=args.mlp_iter_num,
        loc_num=args.loc_num,
        distance_threshold_m=args.distance_threshold_m,
        time_limit_seconds=args.time_limit_seconds,
        build_num=args.build_num,
    )
    print("candidate_count =", len(result.candidate_result.candidates))
    print("scored_candidate_count =", len(result.total_score_result.scored_candidates))
    print("xgb_selected_count =", len(result.optimization_result.xgb_selected))
    print("mlp_selected_count =", len(result.optimization_result.mlp_selected))
    print("total_score_output =", result.total_score_result.output_path)
    return result


if __name__ == "__main__":
    main()
