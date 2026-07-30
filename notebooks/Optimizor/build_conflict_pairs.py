"""Build a cached DSS-style distance indicator matrix for Yixing optimization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import OPTIMIZATION_DISTANCE_THRESHOLD_M, TOTAL_SCORE_PATH
from Data import default_indicator_cache_path, write_indicator_cache
from yixing_optimizer_utils import read_csv_flexible


@dataclass(frozen=True)
class ConflictIndicatorBuildResult:
    """Outputs from building a dense indicator matrix."""

    indicator_path: Path
    metadata_path: Path
    candidate_count: int
    conflict_pair_count: int
    distance_threshold_m: float


def build_conflict_indicator(
    *,
    input_path: Path = TOTAL_SCORE_PATH,
    output_path: Path | None = None,
    distance_threshold_m: float = OPTIMIZATION_DISTANCE_THRESHOLD_M,
    build_num: int | None = None,
) -> ConflictIndicatorBuildResult:
    """Build and write a dense bool indicator matrix from scored candidates."""

    input_path = Path(input_path)
    output_path = default_indicator_cache_path(input_path=input_path, threshold_m=distance_threshold_m) if output_path is None else Path(output_path)

    with tqdm(total=4, desc="Building conflict indicator", unit="step", dynamic_ncols=True) as progress:
        candidate_df = read_csv_flexible(input_path)
        if build_num is not None:
            candidate_df = candidate_df.head(int(build_num)).copy()
        candidate_df = candidate_df.reset_index(drop=True)
        progress.set_postfix_str(f"loaded {len(candidate_df):,} candidates")
        progress.update()

        indicator_path, metadata_path, indicator, _ = write_indicator_cache(
            candidate_df,
            indicator_path=output_path,
            threshold_m=distance_threshold_m,
            input_path=input_path,
        )
        progress.set_postfix_str(f"matrix {indicator.shape[0]:,} x {indicator.shape[1]:,}")
        progress.update()

        conflict_pair_count = int(np.triu(indicator, k=1).sum())
        progress.set_postfix_str(f"{conflict_pair_count:,} conflict pairs")
        progress.update()

        progress.set_postfix_str("cache written")
        progress.update()

    return ConflictIndicatorBuildResult(
        indicator_path=indicator_path,
        metadata_path=metadata_path,
        candidate_count=len(candidate_df),
        conflict_pair_count=conflict_pair_count,
        distance_threshold_m=float(distance_threshold_m),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Yixing dense distance-conflict indicator matrix.")
    parser.add_argument("--input", type=Path, default=TOTAL_SCORE_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--distance-threshold-m", type=float, default=OPTIMIZATION_DISTANCE_THRESHOLD_M)
    parser.add_argument("--build-num", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> ConflictIndicatorBuildResult:
    args = parse_args(argv)
    result = build_conflict_indicator(
        input_path=args.input,
        output_path=args.output,
        distance_threshold_m=args.distance_threshold_m,
        build_num=args.build_num,
    )
    print("candidate_count =", result.candidate_count)
    print("conflict_pair_count =", result.conflict_pair_count)
    print("indicator_output =", result.indicator_path)
    print("metadata_output =", result.metadata_path)
    return result


if __name__ == "__main__":
    main()