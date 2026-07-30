"""Run the Yixing AED candidate pipeline and optimization."""

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
    SVR_SELECTED_PATH,
    OUTPUT_DIR,
    RANDOM_SEED,
    TOTAL_SCORE_PATH,
    XGB_SELECTED_PATH,
)
from Data import Data, default_indicator_cache_path
from ModelBuilder import ModelBuilder


DEFAULT_INPUT = TOTAL_SCORE_PATH
DEFAULT_OUTPUT = XGB_SELECTED_PATH


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
    indicator_path: Path | None = None,
    use_indicator_cache: bool = True,
) -> FullPipelineResult:
    """Run candidate generation, total-score calculation, and all optimization passes."""

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
    parser.add_argument("--no-boundary-filter", action="store_true")
    parser.add_argument("--dedup-distance-m", type=float, default=DEDUP_DISTANCE_THRESHOLD_M)
    parser.add_argument("--max-candidates", type=parse_candidate_limit, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
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
    parser.add_argument("--loc-num", type=int, default=DEFAULT_LOC_NUM)
    parser.add_argument("--score-col", default="total_score_xgb")
    parser.add_argument("--distance-threshold-m", type=float, default=OPTIMIZATION_DISTANCE_THRESHOLD_M)
    parser.add_argument("--time-limit-seconds", type=int, default=DEFAULT_OPTIMIZATION_TIME_LIMIT_SECONDS)
    parser.add_argument("--build-num", type=int, default=None)
    parser.add_argument("--indicator-path", type=Path, default=None)
    parser.add_argument("--no-indicator-cache", action="store_true")
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
        run_boundary_filter=not args.no_boundary_filter,
        dedup_distance_m=args.dedup_distance_m,
        max_candidates=args.max_candidates,
        random_seed=args.random_seed,
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
