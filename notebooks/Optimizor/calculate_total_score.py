"""Calculate model SHAP total scores for Yixing whitelist candidates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from config import (
    CANDIDATE_WHITELIST_PATH,
    MLP_ITER_NUM,
    TOTAL_SCORE_PATH,
)
from yixing_optimizer_utils import (
    aggregate_h3_feature_scores,
    compute_h3_prediction_area_scores,
    compute_mlp_h3_scores,
    compute_svr_h3_scores,
    compute_xgb_h3_scores,
    prepare_h3_training_frames,
    read_csv_flexible,
)


@dataclass(frozen=True)
class TotalScoreResult:
    """In-memory result for the total-score step."""

    scored_candidates: pd.DataFrame
    xgb_prediction_df: pd.DataFrame
    mlp_prediction_df: pd.DataFrame
    svr_prediction_df: pd.DataFrame
    output_path: Path


def add_h3_frames(candidates: pd.DataFrame, *h3_frames: pd.DataFrame) -> pd.DataFrame:
    """Attach per-H3 score and prediction columns to each candidate row."""

    out = candidates.copy()
    for frame in h3_frames:
        out = out.merge(
            frame.rename(columns={"id": "h3_l7"}),
            how="left",
            on="h3_l7",
        )
    return out


def add_score_ranks(scored: pd.DataFrame) -> pd.DataFrame:
    """Attach rank columns for each total-score variant."""

    out = scored.copy()
    for model_name in ("xgb", "mlp", "svr"):
        score_col = f"total_score_{model_name}"
        rank_col = f"score_rank_{model_name}"
        out[rank_col] = out[score_col].rank(method="first", ascending=False).astype(int)
    return out


def write_total_score_csv(scored: pd.DataFrame, output_path: Path = TOTAL_SCORE_PATH) -> Path:
    """Write the main total-score CSV used by the optimizer."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def calculate_total_scores(
    *,
    input_path: Path = CANDIDATE_WHITELIST_PATH,
    output_path: Path = TOTAL_SCORE_PATH,
    mlp_iter_num: int = MLP_ITER_NUM,
) -> TotalScoreResult:
    """Compute XGB, MLP, and SVR total scores for every whitelist candidate."""

    with tqdm(total=11, desc="Calculating total_score.csv", unit="step", dynamic_ncols=True) as progress:
        candidates = read_csv_flexible(Path(input_path))
        progress.set_postfix_str(f"loaded {len(candidates):,} candidates")
        progress.update()

        source_h3_df, yixing_h3_df, feature_cols = prepare_h3_training_frames()
        progress.set_postfix_str(f"loaded {len(yixing_h3_df):,} Yixing H3 cells")
        progress.update()

        xgb_h3_feature_df, xgb_prediction_df = compute_xgb_h3_scores(
            source_h3_df,
            yixing_h3_df,
            feature_cols,
        )
        xgb_h3_score_df = aggregate_h3_feature_scores(xgb_h3_feature_df, score_col="h3_shap_score_xgb")
        progress.set_postfix_str("XGB H3 SHAP scores ready")
        progress.update()

        scored = compute_h3_prediction_area_scores(
            candidates,
            xgb_h3_score_df,
            prediction_col="h3_shap_score_xgb",
            total_score_col="total_score_xgb",
            show_progress=True,
            progress_desc="Calculating total_score_xgb",
        )
        progress.set_postfix_str("total_score_xgb ready")
        progress.update()

        mlp_h3_feature_df, mlp_prediction_df = compute_mlp_h3_scores(
            source_h3_df,
            yixing_h3_df,
            feature_cols,
            iter_num=mlp_iter_num,
        )
        mlp_h3_score_df = aggregate_h3_feature_scores(mlp_h3_feature_df, score_col="h3_shap_score_mlp")
        progress.set_postfix_str("MLP H3 SHAP scores ready")
        progress.update()

        scored = compute_h3_prediction_area_scores(
            scored,
            mlp_h3_score_df,
            prediction_col="h3_shap_score_mlp",
            total_score_col="total_score_mlp",
            show_progress=True,
            progress_desc="Calculating total_score_mlp",
        )
        progress.set_postfix_str("total_score_mlp ready")
        progress.update()

        svr_h3_feature_df, svr_prediction_df = compute_svr_h3_scores(
            source_h3_df,
            yixing_h3_df,
            feature_cols,
        )
        svr_h3_score_df = aggregate_h3_feature_scores(svr_h3_feature_df, score_col="h3_shap_score_svr")
        progress.set_postfix_str("SVR H3 SHAP scores ready")
        progress.update()

        scored = compute_h3_prediction_area_scores(
            scored,
            svr_h3_score_df,
            prediction_col="h3_shap_score_svr",
            total_score_col="total_score_svr",
            show_progress=True,
            progress_desc="Calculating total_score_svr",
        )
        progress.set_postfix_str("total_score_svr ready")
        progress.update()

        scored = add_h3_frames(
            scored,
            xgb_h3_score_df,
            mlp_h3_score_df,
            svr_h3_score_df,
            xgb_prediction_df,
            mlp_prediction_df,
            svr_prediction_df,
        )
        progress.set_postfix_str("H3 scores and predictions attached")
        progress.update()

        scored = add_score_ranks(scored)
        scored = scored.sort_values("total_score_xgb", ascending=False).reset_index(drop=True)
        progress.set_postfix_str("rank columns ready")
        progress.update()

        written_path = write_total_score_csv(scored, output_path)
        progress.set_postfix_str("CSV written")
        progress.update()

    return TotalScoreResult(
        scored_candidates=scored,
        xgb_prediction_df=xgb_prediction_df,
        mlp_prediction_df=mlp_prediction_df,
        svr_prediction_df=svr_prediction_df,
        output_path=written_path,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate Yixing candidate total scores.")
    parser.add_argument("--input", type=Path, default=CANDIDATE_WHITELIST_PATH)
    parser.add_argument("--output", type=Path, default=TOTAL_SCORE_PATH)
    parser.add_argument("--mlp-iter-num", type=int, default=MLP_ITER_NUM)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> TotalScoreResult:
    args = parse_args(argv)
    result = calculate_total_scores(
        input_path=args.input,
        output_path=args.output,
        mlp_iter_num=args.mlp_iter_num,
    )
    print("scored_candidate_count =", len(result.scored_candidates))
    print("output =", result.output_path)
    return result


if __name__ == "__main__":
    main()