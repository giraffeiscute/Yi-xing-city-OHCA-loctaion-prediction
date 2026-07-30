"""Shared configuration for the Yixing optimizer Python pipeline.

The values here are intentionally plain module constants so every step script
can be run independently without importing notebook state.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPTIMIZOR_DIR = PROJECT_ROOT / "notebooks" / "Optimizor"
OUTPUT_DIR = OPTIMIZOR_DIR / "outputs"

MAPPED_POI_PATH = PROJECT_ROOT / "data" / "interim" / "mapped_data.csv"
SOURCE_H3_PATH = PROJECT_ROOT / "data" / "processed" / "h3_l7_df_new.csv"
YIXING_H3_PATH = PROJECT_ROOT / "data" / "processed" / "h3_l7_df_yixing_full.csv"

CANDIDATE_WHITELIST_PATH = OUTPUT_DIR / "yixing_candidates_whitelist.csv"
TOTAL_SCORE_PATH = OUTPUT_DIR / "total_score.csv"
XGB_SELECTED_PATH = OUTPUT_DIR / "yixing_selected_locations_xgb.csv"
MLP_SELECTED_PATH = OUTPUT_DIR / "yixing_selected_locations_mlp.csv"

CITY_NAME = "Yixing, Wuxi, Jiangsu, China"
H3_RESOLUTION = 7
RANDOM_SEED = 277
MAX_CANDIDATES = 2_000
DEFAULT_MAX_CANDIDATES = None
DEDUP_DISTANCE_THRESHOLD_M = 10.0
OPTIMIZATION_DISTANCE_THRESHOLD_M = 20.0
WGS84_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:32650"
AED_RANGE_KM = 1.21
H3_CENTER_RADIUS_KM = 1.21
MLP_ITER_NUM = 5_000
DEFAULT_LOC_NUM = 5
DEFAULT_OPTIMIZATION_TIME_LIMIT_SECONDS = 600

# Candidate groups are whitelist categories used before scoring.
CANDIDATE_FEATURE_GROUPS: dict[str, list[str]] = {
    "healthcare": ["hospital", "clinic", "pharmacy", "dentist"],
    "education": [
        "school",
        "kindergarten",
        "university",
        "college",
        "training",
        "library",
    ],
    "transportation": [
        "bus_station",
        "parking",
        "parking_entrance",
        "charging_station",
        "fuel",
    ],
    "public_services": ["marketplace", "bank", "post_office"],
    "culture_recreation_public": [
        "public",
        "sports_centre",
        "cinema",
        "theatre",
        "arts_centre",
        "exhibition_centre",
    ],
}

FEATURE_SOURCE_COLUMNS = (
    "osm_tag",
    "amenity",
    "shop",
    "healthcare",
    "leisure",
    "tourism",
    "office",
    "landuse",
    "building",
)
