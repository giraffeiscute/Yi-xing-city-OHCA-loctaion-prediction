"""Yixing optimizer data loader.

This mirrors the DSS ``Data`` role: read scored candidate locations, expose
arrays used by the optimization model, and prepare distance-conflict indicators.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import build_conflict_pairs as conflict_utils
from config import OPTIMIZATION_DISTANCE_THRESHOLD_M
from yixing_optimizer_utils import haversine_distance_km, read_csv_flexible


class Data(object):
    def __init__(self):
        self.loc_lat = None
        self.loc_lon = None
        self.loc_score = None
        self.loc_score_mlp = None

        self.dist_i_j = {}
        self.indicator_i_j = {}
        self.conflict_pairs = []
        self.indicator_cache_path = None
        self.indicator_cache_used = False
        self.conflict_pair_cache_used = False
        self.infinite = 0

        self.build_num = 0
        self.loc_num = 0
        self.dist_limit = 0.0
        self.dist_limit_m = 0.0

        self.candidate_df = pd.DataFrame()
        self._full_candidate_count = 0
        self._source_candidate_indices = None

    def read_data(
        self,
        file_name,
        loc_num=5,
        dist_limit_m=OPTIMIZATION_DISTANCE_THRESHOLD_M,
        build_num=None,
        score_col="total_score",
        mlp_score_col="total_score_mlp",
        indicator_path=None,
        use_indicator_cache=True,
        sample_indices=None,
        sample_id=None,
        full_conflict_pairs=None,
    ):
        input_path = Path(file_name)
        df = read_csv_flexible(input_path)
        if sample_indices is not None:
            use_indicator_cache = False
            indicator_path = None
        elif indicator_path is None and use_indicator_cache:
            indicator_path = conflict_utils.default_indicator_cache_path(input_path=input_path, threshold_m=dist_limit_m)
        return self.read_dataframe(
            df,
            loc_num=loc_num,
            dist_limit_m=dist_limit_m,
            build_num=build_num,
            score_col=score_col,
            mlp_score_col=mlp_score_col,
            indicator_path=indicator_path,
            use_indicator_cache=use_indicator_cache,
            input_path=input_path,
            sample_indices=sample_indices,
            sample_id=sample_id,
            full_conflict_pairs=full_conflict_pairs,
        )

    def read_dataframe(
        self,
        candidate_df,
        loc_num=5,
        dist_limit_m=OPTIMIZATION_DISTANCE_THRESHOLD_M,
        build_num=None,
        score_col="total_score",
        mlp_score_col="total_score_mlp",
        indicator_path=None,
        use_indicator_cache=True,
        input_path=None,
        sample_indices=None,
        sample_id=None,
        full_conflict_pairs=None,
    ):
        full_df = candidate_df.reset_index(drop=True).copy()
        cached_indicator = None
        cached_conflict_pairs = None if full_conflict_pairs is None else np.asarray(full_conflict_pairs, dtype=np.int64)
        self.indicator_cache_path = Path(indicator_path) if indicator_path is not None else None
        self.indicator_cache_used = False
        self.conflict_pair_cache_used = False
        self._full_candidate_count = len(full_df)

        if sample_indices is not None:
            df = self._sample_dataframe(full_df, sample_indices=sample_indices, sample_id=sample_id)
            self._source_candidate_indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
        else:
            df = full_df
            if build_num is not None:
                df = df.head(int(build_num)).copy()
            self._source_candidate_indices = np.arange(len(df), dtype=np.int64)
            if use_indicator_cache and self.indicator_cache_path is not None:
                cached_indicator = conflict_utils.load_valid_indicator_cache(
                    full_df,
                    indicator_path=self.indicator_cache_path,
                    threshold_m=dist_limit_m,
                    input_path=input_path,
                )

        lat_col, lon_col = self._lat_lon_columns(df)
        if score_col not in df.columns:
            raise KeyError(f"Missing score column for optimization: {score_col}")

        self.candidate_df = df.reset_index(drop=True)
        self.loc_lat = pd.to_numeric(self.candidate_df[lat_col], errors="coerce").to_numpy(dtype=float)
        self.loc_lon = pd.to_numeric(self.candidate_df[lon_col], errors="coerce").to_numpy(dtype=float)
        self.loc_score = pd.to_numeric(self.candidate_df[score_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        if mlp_score_col in self.candidate_df.columns:
            self.loc_score_mlp = pd.to_numeric(self.candidate_df[mlp_score_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        else:
            self.loc_score_mlp = None

        self.loc_num = int(loc_num)
        self.build_num = len(self.candidate_df)
        self.dist_limit_m = float(dist_limit_m)
        self.dist_limit = self.dist_limit_m / 1000.0
        self._build_distance_indicator(cached_indicator=cached_indicator, cached_conflict_pairs=cached_conflict_pairs)
        return self

    def has_distance_conflict(self, i, j):
        if self.infinite != 0:
            return False
        return bool(self.indicator_i_j.get((i, j), self.indicator_i_j.get((j, i), 0)))

    def _build_distance_indicator(self, cached_indicator=None, cached_conflict_pairs=None):
        self.dist_i_j = {}
        self.indicator_i_j = {}
        self.conflict_pairs = []

        if self.dist_limit_m <= 0 or self.candidate_df.empty:
            self.infinite = 1
            return

        self.infinite = 0
        if cached_conflict_pairs is not None:
            self.conflict_pairs = conflict_utils.filter_conflict_pairs_for_sample(
                cached_conflict_pairs,
                self._source_candidate_indices,
                candidate_count=self._full_candidate_count,
            )
            self.conflict_pair_cache_used = True
            self.indicator_cache_used = False
        elif cached_indicator is not None:
            indicator_slice = np.array(cached_indicator[: self.build_num, : self.build_num], dtype=bool, copy=True)
            np.fill_diagonal(indicator_slice, False)
            self.conflict_pairs = conflict_utils.indicator_matrix_to_pairs(indicator_slice)
            self.indicator_cache_used = True
            self.conflict_pair_cache_used = False
        else:
            self.conflict_pairs = conflict_utils.build_distance_conflict_pairs(self.candidate_df, threshold_m=self.dist_limit_m)
            self.indicator_cache_used = False
            self.conflict_pair_cache_used = False

        for i, j in self.conflict_pairs:
            distance_km = haversine_distance_km(self.loc_lat[i], self.loc_lon[i], self.loc_lat[j], self.loc_lon[j])
            self.dist_i_j[i, j] = distance_km
            self.dist_i_j[j, i] = distance_km
            self.indicator_i_j[i, j] = 1
            self.indicator_i_j[j, i] = 1

    @staticmethod
    def _sample_dataframe(candidate_df, *, sample_indices, sample_id=None):
        indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
        if len(indices) == 0:
            raise ValueError("sample_indices must not be empty.")
        if indices.min() < 0 or indices.max() >= len(candidate_df):
            raise IndexError("sample_indices contains out-of-range candidate indices.")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("sample_indices must be unique within each sample.")

        out = candidate_df.iloc[indices].copy().reset_index(drop=True)
        out.insert(0, "source_candidate_index", indices.astype(int))
        if sample_id is not None:
            out.insert(0, "sample_id", int(sample_id))
        return out

    @staticmethod
    def _lat_lon_columns(df):
        if "latitude" in df.columns and "longitude" in df.columns:
            return "latitude", "longitude"
        if "lat" in df.columns and "lon" in df.columns:
            return "lat", "lon"
        raise KeyError("Expected candidate latitude/longitude columns were not found.")