# Yixing Candidate Scoring and Optimization

This directory contains the Python-only Yixing AED candidate pipeline. The flow
is split into focused modules so candidate filtering, total-score calculation,
and optimization can be checked separately. `run_this.py` is the main controller.

## Module Roles

1. `build_candidates.py`
   - Input: `data/interim/mapped_data.csv`
   - Output: `outputs/yixing_candidates_whitelist.csv`
   - Behavior: keeps whitelist POI features, optionally filters inside the
     Yixing boundary, deduplicates within `10m`, and keeps all deduplicated
     candidates by default.

2. `calculate_total_score.py`
   - Input: `outputs/yixing_candidates_whitelist.csv`
   - Output: `outputs/total_score.csv`
   - Behavior: computes XGBoost, MLP, and SVR SHAP-derived scoring columns
     for every whitelist candidate.

3. `build_conflict_pairs.py`
   - Input: `outputs/total_score.csv`
   - Output: `outputs/indicator_i_j_20m.npy` and `outputs/indicator_i_j_20m.json`
   - Behavior: builds a DSS-style dense bool conflict matrix where `True`
     means two candidates are within the optimization distance threshold.

4. `Data.py` and `ModelBuilder.py`
   - `Data.py` reads scored candidates, prepares score/coordinate arrays, and
     loads a valid cached indicator matrix before falling back to live
     distance-conflict pair calculation.
   - `ModelBuilder.py` builds and solves the Gurobi binary IP.

5. `run_this.py`
   - Main controller for the full Yixing pipeline.
   - Default behavior: build candidates, calculate total scores, then optimize
     XGB, MLP, and SVR selected locations.
   - `optimize` subcommand: run optimization only against an existing
     `total_score.csv`.

6. `yixing_optimizer_utils.py`
   - Shared CSV, H3, and geometry helpers only.

## Candidate Whitelist

- `healthcare`: `hospital`, `clinic`, `pharmacy`, `dentist`
- `education`: `school`, `kindergarten`, `university`, `college`, `training`, `library`
- `transportation`: `bus_station`, `parking`, `parking_entrance`, `charging_station`, `fuel`
- `public_services`: `marketplace`, `bank`, `post_office`
- `culture_recreation_public`: `public`, `sports_centre`, `cinema`, `theatre`, `arts_centre`, `exhibition_centre`

## Main Outputs

Outputs are written under `notebooks/Optimizor/outputs/`:

- `yixing_poi_cleaned_summary.csv`
- `yixing_candidates_whitelist.csv`
- `yixing_candidate_feature_counts.csv`
- `yixing_excluded_feature_counts.csv`
- `total_score.csv`
- `indicator_i_j_20m.npy`
- `indicator_i_j_20m.json`
- `yixing_selected_locations_xgb.csv`
- `yixing_selected_locations_mlp.csv`
- `yixing_selected_locations_svr.csv`

The main scoring CSV contains Yixing candidate fields plus:

- `total_score_xgb`
- `total_score_mlp`
- `total_score_svr`
- `score_rank_xgb`
- `score_rank_mlp`
- `score_rank_svr`

`total_score_xgb`, `total_score_mlp`, and `total_score_svr` keep their raw area-score scale. H3 SHAP components
and model predictions are kept out of the main CSV.

## Run Commands

Use the project environment, for example:

```powershell
conda run -n yixin_env python notebooks\Optimizor\run_this.py
```

Run the full pipeline, including all three optimization passes:

```powershell
conda run -n yixin_env python notebooks\Optimizor\run_this.py
```

Build whitelist candidates as an independent step:

```powershell
conda run -n yixin_env python notebooks\Optimizor\build_candidates.py
```

Calculate total scores as an independent step:

```powershell
conda run -n yixin_env python notebooks\Optimizor\calculate_total_score.py
```

Build the cached 20m conflict indicator matrix as an independent step:

```powershell
conda run -n yixin_env python notebooks\Optimizor\build_conflict_pairs.py
```

`run_this.py optimize` automatically looks for `indicator_i_j_<threshold>m.npy`
next to the input `total_score.csv`. Use `--indicator-path` to override that
file, or `--no-indicator-cache` to force live distance calculation.

Run optimization only against an existing `total_score.csv`:

```powershell
conda run -n yixin_env python notebooks\Optimizor\run_this.py optimize --input notebooks\Optimizor\outputs\total_score.csv --score-col total_score_xgb --output notebooks\Optimizor\outputs\yixing_selected_locations_xgb.csv
conda run -n yixin_env python notebooks\Optimizor\run_this.py optimize --input notebooks\Optimizor\outputs\total_score.csv --mlp --output notebooks\Optimizor\outputs\yixing_selected_locations_mlp.csv
conda run -n yixin_env python notebooks\Optimizor\run_this.py optimize --input notebooks\Optimizor\outputs\total_score.csv --svr --output notebooks\Optimizor\outputs\yixing_selected_locations_svr.csv
```

Use `--max-candidates 2000` on `build_candidates.py` or `run_this.py` only when
you want the old sampled workflow. The default is all deduplicated whitelist
candidates.

## Tests

```powershell
python -m pytest tests\test_yixing_optimizer_utils.py notebooks\Optimizor\tests -q -o cache_dir=C:\tmp\pytest_cache_yixing_optimizer
```
