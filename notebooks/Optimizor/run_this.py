"""Run Yixing AED candidate optimization.

This is the Yixing equivalent of ``notebooks/Optimizor_DSS/run_this_20250720.py``:
load scored candidates, build the Data object, solve the IP, and write selected
locations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from Data import Data
from ModelBuilder import ModelBuilder
from yixing_optimizer_utils import OPTIMIZATION_DISTANCE_THRESHOLD_M, OUTPUT_DIR


DEFAULT_INPUT = OUTPUT_DIR / "yixing_candidates_ranked.csv"
DEFAULT_OUTPUT = OUTPUT_DIR / "yixing_selected_locations.csv"


def optimize_candidates(
    input_path=DEFAULT_INPUT,
    output_path=DEFAULT_OUTPUT,
    loc_num=5,
    score_col="total_score",
    distance_threshold_m=OPTIMIZATION_DISTANCE_THRESHOLD_M,
    time_limit_seconds=600,
    build_num=None,
    use_mlp=False,
):
    data = Data()
    data.read_data(
        input_path,
        loc_num=loc_num,
        dist_limit_m=distance_threshold_m,
        build_num=build_num,
        score_col=score_col,
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run Yixing AED candidate optimization.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--loc-num", type=int, default=5)
    parser.add_argument("--score-col", default="total_score")
    parser.add_argument("--distance-threshold-m", type=float, default=OPTIMIZATION_DISTANCE_THRESHOLD_M)
    parser.add_argument("--time-limit-seconds", type=int, default=600)
    parser.add_argument("--build-num", type=int, default=None)
    parser.add_argument("--mlp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    score_col = "total_score_mlp" if args.mlp and args.score_col == "total_score" else args.score_col
    selected_df, objective_value, conflict_pairs = optimize_candidates(
        input_path=args.input,
        output_path=args.output,
        loc_num=args.loc_num,
        score_col=score_col,
        distance_threshold_m=args.distance_threshold_m,
        time_limit_seconds=args.time_limit_seconds,
        build_num=args.build_num,
        use_mlp=args.mlp,
    )
    print("obj_val =", objective_value)
    print("conflict_pair_count =", len(conflict_pairs))
    print("deployment_decision =", selected_df["optimization_source_index"].tolist())
    print("output =", args.output)