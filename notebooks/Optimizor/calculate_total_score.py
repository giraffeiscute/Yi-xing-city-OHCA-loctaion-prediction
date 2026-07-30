"""Calculate model SHAP total scores for Yixing whitelist candidates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import (
    AED_RANGE_KM,
    CANDIDATE_FEATURE_GROUPS,
    CANDIDATE_WHITELIST_PATH,
    H3_CENTER_RADIUS_KM,
    MLP_ITER_NUM,
    OUTPUT_DIR,
    RANDOM_SEED,
    SOURCE_H3_PATH,
    TOTAL_SCORE_PATH,
    WGS84_CRS,
    YIXING_H3_PATH,
)
from yixing_optimizer_utils import (
    cache_h3_centers,
    ensure_output_dir,
    h3_center_latlon,
    haversine_array,
    intersection_area,
    read_csv_flexible,
)


FEATURE_PRIORITY: dict[str, int] = {
    feature: index
    for index, feature in enumerate(
        feature
        for features in CANDIDATE_FEATURE_GROUPS.values()
        for feature in features
    )
}


@dataclass(frozen=True)
class TotalScoreResult:
    """In-memory result for the total-score step."""

    scored_candidates: pd.DataFrame
    xgb_prediction_df: pd.DataFrame
    mlp_prediction_df: pd.DataFrame
    svr_prediction_df: pd.DataFrame
    output_path: Path


def prepare_h3_training_frames(
    source_h3_path: Path = SOURCE_H3_PATH,
    yixing_h3_path: Path = YIXING_H3_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load source and Yixing H3 matrices with aligned feature columns."""

    source = read_csv_flexible(source_h3_path)
    yixing = read_csv_flexible(yixing_h3_path)
    if "commercial;yes" in source.columns:
        source = source.drop(columns=["commercial;yes"])
    feature_cols = [col for col in source.columns if col not in {"id", "ohca"}]
    missing = [col for col in feature_cols if col not in yixing.columns]
    if missing:
        raise KeyError(f"Yixing H3 matrix is missing feature columns: {missing[:10]}")
    return source, yixing, feature_cols


def source_minmax_normalize(
    target_features: pd.DataFrame,
    source_features: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize target features with the source-city Min-Max range."""

    source_min = source_features.min()
    source_range = source_features.max() - source_min
    zero_range = source_range == 0
    adjusted_range = source_range.mask(zero_range, 1)
    normalized = (target_features - source_min) / adjusted_range
    normalized.loc[:, zero_range] = 0
    return normalized


def train_xgb_source_model(source_h3_df: pd.DataFrame, feature_cols: list[str]):
    """Train the XGB model using the existing project hyperparameters."""

    import xgboost as xgb

    normalized = source_minmax_normalize(
        source_h3_df[feature_cols + ["ohca"]],
        source_h3_df[feature_cols + ["ohca"]],
    )
    centers = [h3_center_latlon(str(hid)) for hid in source_h3_df["id"]]
    train_index = [idx for idx, (_, lon) in enumerate(centers) if lon > -76.05]
    test_index = [idx for idx, (_, lon) in enumerate(centers) if lon <= -76.05]
    spatial_data = normalized.to_numpy(dtype=np.float64)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        early_stopping_rounds=10,
        max_depth=3,
        learning_rate=0.2,
        n_estimators=100,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=0,
        reg_alpha=0.1,
        reg_lambda=1,
        min_child_weight=3,
    )
    model.fit(
        spatial_data[train_index, :-1],
        spatial_data[train_index, -1],
        eval_set=[(spatial_data[test_index, :-1], spatial_data[test_index, -1])],
        verbose=False,
    )
    ohca_min = float(source_h3_df["ohca"].min())
    ohca_range = float(source_h3_df["ohca"].max() - ohca_min)
    return model, ohca_min, ohca_range


def compute_xgb_h3_scores(
    source_h3_df: pd.DataFrame,
    yixing_h3_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute DSS-style XGB H3 feature scores and predicted OHCA for Yixing."""

    import shap

    model, ohca_min, ohca_range = train_xgb_source_model(source_h3_df, feature_cols)
    source_features = source_h3_df[feature_cols]
    yixing_input = source_minmax_normalize(yixing_h3_df[feature_cols], source_features)
    x_yixing = yixing_input.to_numpy(dtype=np.float64)
    pred_norm = model.predict(x_yixing)
    predicted_ohca = pred_norm * ohca_range + ohca_min

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_yixing) * ohca_range
    score_df = _feature_score_from_shap(yixing_h3_df["id"], yixing_h3_df[feature_cols], shap_values, feature_cols)
    prediction_df = pd.DataFrame({"id": yixing_h3_df["id"].astype(str), "predicted_ohca_xgb": predicted_ohca})
    return score_df, prediction_df


def train_mlp_source_model(
    source_h3_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    iter_num: int = 5_000,
    hidden_multiplier: int = 2,
):
    """Train the MLP model using the existing project architecture."""

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    class Regressor(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, output_size: int = 1):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size)
            nn.init.normal_(self.fc1.weight, std=0.02)
            nn.init.constant_(self.fc1.bias, 0)
            nn.init.normal_(self.fc2.weight, std=0.02)
            nn.init.constant_(self.fc2.bias, 0)
            nn.init.normal_(self.fc3.weight, std=0.02)
            nn.init.constant_(self.fc3.bias, 0)

        def forward(self, input_tensor):
            output = F.relu(self.fc1(input_tensor))
            output = F.relu(self.fc2(output))
            return self.fc3(output)

    normalized = source_minmax_normalize(
        source_h3_df[feature_cols + ["ohca"]],
        source_h3_df[feature_cols + ["ohca"]],
    )
    centers = [h3_center_latlon(str(hid)) for hid in source_h3_df["id"]]
    train_index = [idx for idx, (_, lon) in enumerate(centers) if lon > -76.05]
    spatial_data = normalized.to_numpy(dtype=np.float64)
    train_spatial_data = spatial_data[train_index]

    torch.manual_seed(123)
    np.random.seed(123)
    model = Regressor(
        input_size=len(feature_cols),
        hidden_size=len(feature_cols) * hidden_multiplier,
        output_size=1,
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    for _ in range(iter_num):
        h3_l7_id = np.random.choice(train_spatial_data.shape[0] - 1, 1)
        ohca = torch.autograd.Variable(torch.FloatTensor(train_spatial_data[h3_l7_id, -1].reshape(-1, 1)))
        pred = model(torch.autograd.Variable(torch.FloatTensor(train_spatial_data[h3_l7_id, :-1]))).reshape(-1, 1)
        loss = torch.nn.MSELoss(reduction="sum")(pred, ohca)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ohca_min = float(source_h3_df["ohca"].min())
    ohca_range = float(source_h3_df["ohca"].max() - ohca_min)
    return model, ohca_min, ohca_range


def to_2d_shap_values(raw_shap, sample_count: int, feature_count: int) -> np.ndarray:
    """Normalize SHAP output from different explainers to a sample-by-feature array."""

    if isinstance(raw_shap, list):
        raw_shap = raw_shap[0]
    shap_values = np.asarray(raw_shap)
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[:, :, 0]
    return shap_values.reshape(sample_count, feature_count)


def compute_mlp_h3_scores(
    source_h3_df: pd.DataFrame,
    yixing_h3_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    iter_num: int = 5_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute MLP H3 feature SHAP scores and predicted OHCA for Yixing."""

    import shap
    import torch

    model, ohca_min, ohca_range = train_mlp_source_model(
        source_h3_df,
        feature_cols,
        iter_num=iter_num,
    )
    yixing_input = source_minmax_normalize(yixing_h3_df[feature_cols], source_h3_df[feature_cols])
    x_yixing = yixing_input.to_numpy(dtype=np.float32)
    yixing_tensor = torch.FloatTensor(x_yixing)

    model.eval()
    with torch.no_grad():
        pred_norm = model(yixing_tensor).detach().numpy().flatten()
    predicted = np.clip(pred_norm * ohca_range + ohca_min, a_min=0, a_max=None)

    explainer = shap.GradientExplainer(model, yixing_tensor)
    raw_shap = explainer.shap_values(yixing_tensor)
    shap_values = to_2d_shap_values(raw_shap, len(yixing_h3_df), len(feature_cols)) * ohca_range
    score_df = _feature_score_from_shap(yixing_h3_df["id"], yixing_h3_df[feature_cols], shap_values, feature_cols)
    prediction_df = pd.DataFrame({"id": yixing_h3_df["id"].astype(str), "predicted_ohca_mlp": predicted})
    return score_df, prediction_df


def compute_mlp_h3_predictions(
    source_h3_df: pd.DataFrame,
    yixing_h3_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    iter_num: int = 5_000,
) -> pd.DataFrame:
    """Predict Yixing H3 OHCA values with the MLP flow from the project."""

    import torch

    model, ohca_min, ohca_range = train_mlp_source_model(
        source_h3_df,
        feature_cols,
        iter_num=iter_num,
    )
    yixing_input = source_minmax_normalize(yixing_h3_df[feature_cols], source_h3_df[feature_cols])
    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.FloatTensor(yixing_input.to_numpy(dtype=np.float32))).detach().numpy().flatten()
    predicted = np.clip(pred_norm * ohca_range + ohca_min, a_min=0, a_max=None)
    return pd.DataFrame({"id": yixing_h3_df["id"].astype(str), "predicted_ohca_mlp": predicted})


def train_svr_source_model(source_h3_df: pd.DataFrame, feature_cols: list[str]):
    """Train the linear SVR model used by the project notebooks."""

    from sklearn.svm import SVR

    normalized = source_minmax_normalize(
        source_h3_df[feature_cols + ["ohca"]],
        source_h3_df[feature_cols + ["ohca"]],
    )
    centers = [h3_center_latlon(str(hid)) for hid in source_h3_df["id"]]
    train_index = [idx for idx, (_, lon) in enumerate(centers) if lon > -76.05]
    spatial_data = normalized.to_numpy(dtype=np.float64)
    train_spatial_data = spatial_data[train_index]

    model = SVR(kernel="linear", C=1, epsilon=0.05)
    model.fit(train_spatial_data[:, :-1], train_spatial_data[:, -1])

    ohca_min = float(source_h3_df["ohca"].min())
    ohca_range = float(source_h3_df["ohca"].max() - ohca_min)
    return model, ohca_min, ohca_range


def compute_svr_h3_scores(
    source_h3_df: pd.DataFrame,
    yixing_h3_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute SVR H3 feature SHAP scores and predicted OHCA for Yixing."""

    import shap

    model, ohca_min, ohca_range = train_svr_source_model(source_h3_df, feature_cols)
    yixing_input = source_minmax_normalize(yixing_h3_df[feature_cols], source_h3_df[feature_cols])
    x_yixing = yixing_input.to_numpy(dtype=np.float64)
    pred_norm = model.predict(x_yixing)
    predicted_ohca = pred_norm * ohca_range + ohca_min

    masker = shap.maskers.Independent(x_yixing, max_samples=len(yixing_h3_df))
    explainer = shap.LinearExplainer(model, masker)
    shap_values = to_2d_shap_values(explainer.shap_values(x_yixing), len(yixing_h3_df), len(feature_cols)) * ohca_range
    score_df = _feature_score_from_shap(yixing_h3_df["id"], yixing_h3_df[feature_cols], shap_values, feature_cols)
    prediction_df = pd.DataFrame({"id": yixing_h3_df["id"].astype(str), "predicted_ohca_svr": predicted_ohca})
    return score_df, prediction_df


def aggregate_h3_feature_scores(
    h3_feature_score_df: pd.DataFrame,
    *,
    score_col: str,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate whitelist feature SHAP components into one H3 scalar score."""

    if "id" not in h3_feature_score_df.columns:
        raise ValueError("h3_feature_score_df must include an id column.")
    if feature_cols is None:
        feature_cols = [feature for feature in FEATURE_PRIORITY if feature in h3_feature_score_df.columns]
    else:
        feature_cols = [feature for feature in feature_cols if feature in h3_feature_score_df.columns]

    out = pd.DataFrame({"id": h3_feature_score_df["id"].astype(str)})
    if not feature_cols:
        out[score_col] = 0.0
        return out

    values = h3_feature_score_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    out[score_col] = values.sum(axis=1)
    return out


def attach_candidate_component_scores(
    candidates: pd.DataFrame,
    h3_feature_score_df: pd.DataFrame,
    *,
    score_col: str = "component_score",
) -> pd.DataFrame:
    """Join per-H3/per-feature component scores to candidate rows."""

    long_score = h3_feature_score_df.melt(id_vars="id", var_name="primary_feature", value_name=score_col)
    out = candidates.merge(
        long_score,
        how="left",
        left_on=["h3_l7", "primary_feature"],
        right_on=["id", "primary_feature"],
        suffixes=("", "_score_h3"),
    )
    return out.drop(columns=["id_score_h3"], errors="ignore")


def area_score_from_points(
    aed_lat: float,
    aed_lon: float,
    score_points: pd.DataFrame,
    *,
    point_score_col: str,
    aed_range_km: float = AED_RANGE_KM,
    center_radius_km: float = H3_CENTER_RADIUS_KM,
) -> float:
    """Compute DSS-style area weighted score around one candidate point."""

    valid = score_points[score_points[point_score_col].notna()].copy()
    if valid.empty:
        return 0.0
    distances = haversine_array(aed_lat, aed_lon, valid["latitude"], valid["longitude"])
    mask = distances <= aed_range_km
    subset = valid.loc[mask].copy()
    if subset.empty:
        return 0.0
    center_distances = haversine_array(
        aed_lat,
        aed_lon,
        subset["center_lat"],
        subset["center_lon"],
    )
    proportions = np.array(
        [intersection_area(aed_range_km, center_radius_km, d) for d in center_distances],
        dtype=float,
    ) / (math.pi * center_radius_km**2)
    return float((subset[point_score_col].to_numpy(dtype=float) * proportions).sum())


def compute_total_scores(
    candidates: pd.DataFrame,
    *,
    point_score_col: str = "component_score",
    total_score_col: str = "total_score",
    show_progress: bool = False,
    progress_desc: str = "Calculating total scores",
) -> pd.DataFrame:
    """Compute total_score for sampled candidates using the DSS area logic."""

    score_points = cache_h3_centers(candidates)
    rows = score_points.itertuples(index=False)
    if show_progress:
        rows = tqdm(rows, total=len(score_points), desc=progress_desc, unit="candidate", leave=False, dynamic_ncols=True)
    totals = [
        area_score_from_points(row.latitude, row.longitude, score_points, point_score_col=point_score_col)
        for row in rows
    ]
    out = score_points.copy()
    out[total_score_col] = totals
    out["score_rank"] = out[total_score_col].rank(method="first", ascending=False).astype(int)
    return out.sort_values(total_score_col, ascending=False).reset_index(drop=True)


def compute_h3_prediction_area_scores(
    candidates: pd.DataFrame,
    h3_prediction_df: pd.DataFrame,
    *,
    prediction_col: str,
    total_score_col: str,
    aed_range_km: float = AED_RANGE_KM,
    center_radius_km: float = H3_CENTER_RADIUS_KM,
    show_progress: bool = False,
    progress_desc: str = "Calculating H3 prediction area scores",
) -> pd.DataFrame:
    """Compute DSS-style area scores from one scalar value per H3 cell."""

    h3_points = h3_prediction_df.copy()
    centers = [h3_center_latlon(str(hid)) for hid in h3_points["id"]]
    h3_points["center_lat"] = [lat for lat, _ in centers]
    h3_points["center_lon"] = [lon for _, lon in centers]

    totals = []
    rows = candidates.itertuples(index=False)
    if show_progress:
        rows = tqdm(rows, total=len(candidates), desc=progress_desc, unit="candidate", leave=False, dynamic_ncols=True)
    for row in rows:
        distances = haversine_array(row.latitude, row.longitude, h3_points["center_lat"], h3_points["center_lon"])
        subset = h3_points.loc[distances <= aed_range_km * 2].copy()
        if subset.empty:
            totals.append(0.0)
            continue
        center_distances = haversine_array(
            row.latitude,
            row.longitude,
            subset["center_lat"],
            subset["center_lon"],
        )
        proportions = np.array(
            [intersection_area(aed_range_km, center_radius_km, d) for d in center_distances],
            dtype=float,
        ) / (math.pi * center_radius_km**2)
        totals.append(float((subset[prediction_col].to_numpy(dtype=float) * proportions).sum()))

    out = candidates.copy()
    out[total_score_col] = totals
    return out


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
    """Attach total-score ranks kept in total_score.csv."""

    out = scored.copy()
    for model_name in ("xgb", "mlp", "svr"):
        total_score_col = f"total_score_{model_name}"
        rank_col = f"score_rank_{model_name}"
        if total_score_col not in out.columns:
            continue
        rank_values = out[total_score_col].rank(method="first", ascending=False).astype(int)
        if rank_col in out.columns:
            out = out.drop(columns=[rank_col])
        insert_at = out.columns.get_loc(total_score_col) + 1
        out.insert(insert_at, rank_col, rank_values)
    return out


def write_total_score_csv(scored: pd.DataFrame, output_path: Path = TOTAL_SCORE_PATH) -> Path:
    """Write the main total-score CSV used by the optimizer."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def write_scored_outputs(scored: pd.DataFrame, *, output_dir: Path = OUTPUT_DIR, score_col: str | None = None) -> None:
    """Write scored candidate outputs and a ranked GeoJSON when geopandas exists."""

    output_dir = ensure_output_dir(output_dir)
    scored.to_csv(output_dir / "yixing_candidates_scored.csv", index=False, encoding="utf-8-sig")
    if score_col is None:
        for candidate_col in ("total_score", "total_score_xgb", "total_score_mlp"):
            if candidate_col in scored.columns:
                score_col = candidate_col
                break
    if score_col is None or score_col not in scored.columns:
        raise KeyError("Expected a total score column for ranked output.")
    ranked = scored.sort_values(score_col, ascending=False).reset_index(drop=True)
    ranked.to_csv(output_dir / "yixing_candidates_ranked.csv", index=False, encoding="utf-8-sig")
    try:
        import geopandas as gpd

        geometry = gpd.points_from_xy(ranked["longitude"], ranked["latitude"])
        gdf = gpd.GeoDataFrame(ranked, geometry=geometry, crs=WGS84_CRS)
        gdf.to_file(output_dir / "yixing_candidates_ranked.geojson", driver="GeoJSON")
    except ImportError:
        pass


def calculate_total_scores(
    *,
    input_path: Path = CANDIDATE_WHITELIST_PATH,
    output_path: Path = TOTAL_SCORE_PATH,
    mlp_iter_num: int = MLP_ITER_NUM,
) -> TotalScoreResult:
    """Compute XGB, MLP, and SVR total scores for every whitelist candidate."""

    with tqdm(total=10, desc="Calculating total_score.csv", unit="step", dynamic_ncols=True) as progress:
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

        scored = add_score_ranks(scored)
        scored = scored.sort_values("total_score_xgb", ascending=False).reset_index(drop=True)
        progress.set_postfix_str("score ranks ready")
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


def _feature_score_from_shap(
    h3_ids: pd.Series,
    raw_feature_df: pd.DataFrame,
    shap_values: np.ndarray,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Convert H3 feature SHAP values to per-unit feature component scores."""

    denominator = raw_feature_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    numerator = np.asarray(shap_values, dtype=float)[:, : len(feature_cols)]
    component_scores = np.array(numerator, copy=True, dtype=float)
    np.divide(numerator, denominator, out=component_scores, where=denominator != 0)

    score_df = pd.DataFrame(component_scores, columns=feature_cols)
    score_df.insert(0, "id", h3_ids.astype(str).to_numpy())
    return score_df


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
