"""Yixing optimizer data loader.

This mirrors the DSS ``Data`` role: read scored candidate locations, expose
arrays used by the optimization model, and build distance-conflict indicators.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from yixing_optimizer_utils import (
    OPTIMIZATION_DISTANCE_THRESHOLD_M,
    build_distance_conflict_pairs,
    read_csv_flexible,
)


class Data(object):
    def __init__(self):
        self.loc_lat = None
        self.loc_lon = None
        self.loc_score = None
        self.loc_score_mlp = None

        self.dist_i_j = {}
        self.indicator_i_j = {}
        self.conflict_pairs = []
        self.infinite = 0

        self.build_num = 0
        self.loc_num = 0
        self.dist_limit = 0.0
        self.dist_limit_m = 0.0

        self.candidate_df = pd.DataFrame()

    def read_data(
        self,
        file_name,
        loc_num=5,
        dist_limit_m=OPTIMIZATION_DISTANCE_THRESHOLD_M,
        build_num=None,
        score_col="total_score",
        mlp_score_col="total_score_mlp",
    ):
        df = read_csv_flexible(Path(file_name))
        return self.read_dataframe(
            df,
            loc_num=loc_num,
            dist_limit_m=dist_limit_m,
            build_num=build_num,
            score_col=score_col,
            mlp_score_col=mlp_score_col,
        )

    def read_dataframe(
        self,
        candidate_df,
        loc_num=5,
        dist_limit_m=OPTIMIZATION_DISTANCE_THRESHOLD_M,
        build_num=None,
        score_col="total_score",
        mlp_score_col="total_score_mlp",
    ):
        df = candidate_df.reset_index(drop=True).copy()
        if build_num is not None:
            df = df.head(int(build_num)).copy()

        lat_col, lon_col = self._lat_lon_columns(df)
        if score_col not in df.columns:
            raise KeyError(f"Missing score column for optimization: {score_col}")

        self.candidate_df = df
        self.loc_lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=float)
        self.loc_lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=float)
        self.loc_score = pd.to_numeric(df[score_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        if mlp_score_col in df.columns:
            self.loc_score_mlp = pd.to_numeric(df[mlp_score_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        else:
            self.loc_score_mlp = None

        self.loc_num = int(loc_num)
        self.build_num = len(df)
        self.dist_limit_m = float(dist_limit_m)
        self.dist_limit = self.dist_limit_m / 1000.0
        self._build_distance_indicator()
        return self

    def haversine(self, lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        d_lat = lat2 - lat1
        d_lon = lon2 - lon1
        a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return c * 6371.0

    def has_distance_conflict(self, i, j):
        if self.infinite != 0:
            return False
        return bool(self.indicator_i_j.get((i, j), self.indicator_i_j.get((j, i), 0)))

    def _build_distance_indicator(self):
        self.dist_i_j = {}
        self.indicator_i_j = {}
        self.conflict_pairs = []

        if self.dist_limit_m <= 0 or self.candidate_df.empty:
            self.infinite = 1
            return

        self.infinite = 0
        self.conflict_pairs = build_distance_conflict_pairs(self.candidate_df, threshold_m=self.dist_limit_m)
        for i, j in self.conflict_pairs:
            distance_km = self.haversine(self.loc_lat[i], self.loc_lon[i], self.loc_lat[j], self.loc_lon[j])
            self.dist_i_j[i, j] = distance_km
            self.dist_i_j[j, i] = distance_km
            self.indicator_i_j[i, j] = 1
            self.indicator_i_j[j, i] = 1

    @staticmethod
    def _lat_lon_columns(df):
        if "latitude" in df.columns and "longitude" in df.columns:
            return "latitude", "longitude"
        if "lat" in df.columns and "lon" in df.columns:
            return "lat", "lon"
        raise KeyError("Expected candidate latitude/longitude columns were not found.")