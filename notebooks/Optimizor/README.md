# Yixing Candidate Scoring and Optimization

This directory contains the Python-only Yixing AED candidate pipeline. The flow
is split into independent steps so candidate filtering, total-score calculation,
and optimization can be checked separately.

`Data.py`, `ModelBuilder.py`, and `run_this.py` keep the existing optimization
interface. New pipeline scripts prepare their inputs and call that interface.

## Pipeline Steps

1. Build whitelist candidates
   - Script: `build_candidates.py`
   - Input: `data/interim/mapped_data.csv`
   - Output: `outputs/yixing_candidates_whitelist.csv`
   - Behavior: keeps whitelist POI features, optionally filters inside the
     Yixing boundary, deduplicates within `10m`, and keeps all deduplicated
     candidates by default.

2. Calculate total scores
   - Script: `calculate_total_score.py`
   - Input: `outputs/yixing_candidates_whitelist.csv`
   - Output: `outputs/total_score.csv`
   - Behavior: computes XGBoost, MLP, and SVR SHAP-derived scoring columns
     for every whitelist candidate.

3. Optimize selected locations
   - Existing script: `run_this.py`
   - Input: `outputs/total_score.csv`
   - XGB score column: `total_score_xgb`
   - MLP score column: `total_score_mlp`
   - Outputs: `outputs/yixing_selected_locations_xgb.csv` and
     `outputs/yixing_selected_locations_mlp.csv`

4. Full controller
   - Script: `run_all.py`
   - Behavior: runs candidate build, total-score calculation, then XGB and MLP
     optimization.

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
- `yixing_selected_locations_xgb.csv`
- `yixing_selected_locations_mlp.csv`

The main scoring CSV contains Yixing candidate fields plus:

- `h3_shap_score_xgb`
- `h3_shap_score_mlp`
- `h3_shap_score_svr`
- `predicted_ohca_xgb`
- `predicted_ohca_mlp`
- `predicted_ohca_svr`
- `total_score_xgb`
- `total_score_mlp`
- `total_score_svr`
- `score_rank_xgb`
- `score_rank_mlp`
- `score_rank_svr`

## Run Commands

Use the project environment:

```powershell
$py = "C:\Users\Yuan\Desktop\YUAN\OHCA 宜興市\yixin_env\Scripts\python.exe"
```

Build all whitelist candidates:

```powershell
& $py notebooks\Optimizor\build_candidates.py
```

Calculate total scores as an independent step:

```powershell
& $py notebooks\Optimizor\calculate_total_score.py
```

Run the full pipeline, including both optimization passes:

```powershell
& $py notebooks\Optimizor\run_all.py
```

Run optimization only with the existing entrypoint:

```powershell
& $py notebooks\Optimizor\run_this.py --input notebooks\Optimizor\outputs\total_score.csv --score-col total_score_xgb --output notebooks\Optimizor\outputs\yixing_selected_locations_xgb.csv
& $py notebooks\Optimizor\run_this.py --input notebooks\Optimizor\outputs\total_score.csv --score-col total_score_mlp --mlp --output notebooks\Optimizor\outputs\yixing_selected_locations_mlp.csv
```

Use `--max-candidates 2000` on `build_candidates.py` or `run_all.py` only when
you want the old sampled workflow. The default is all deduplicated whitelist
candidates.

## Tests

```powershell
& $py -m pytest tests\test_yixing_optimizer_utils.py notebooks\Optimizor\tests -o cache_dir=C:\tmp\pytest_cache_yixing_optimizer
```