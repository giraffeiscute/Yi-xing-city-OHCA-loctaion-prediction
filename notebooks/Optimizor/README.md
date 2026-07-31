# Yixing Candidate Scoring and Optimization

This directory contains the Python-only Yixing AED candidate pipeline. The flow
is split into focused modules so candidate filtering, total-score calculation,
and optimization can be checked separately. `run_this.py` is the main controller.

## Module Roles

1. `build_candidates.py`
   - Input: `data/interim/mapped_data.csv`
   - Output: `outputs/yixing_candidates_whitelist.csv`
   - Behavior: keeps POIs whose `osm_tag` is in `CANDIDATE_FEATURE_GROUPS`, uses WGS84 coordinates, deduplicates within `10m`, and writes every deduplicated whitelist candidate.

2. `calculate_total_score.py`
   - Input: `outputs/yixing_candidates_whitelist.csv`
   - Output: `outputs/total_score.csv`
   - Behavior: computes XGBoost, MLP, and SVR SHAP-derived scoring columns
     for every whitelist candidate.

3. `build_conflict_pairs.py`
   - Input: `outputs/total_score.csv`
   - Output: `outputs/cache/conflict_pairs_50m.npy` and `outputs/cache/conflict_pairs_50m.json`
   - Behavior: builds a sparse `(n_conflict_pairs, 2)` pair cache where each row
     stores two candidates within the optimization distance threshold.

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
- `transportation`: `bus_station`, `parking`, `charging_station`, `fuel`
- `public_services`: `bank`, `post_office`
- `culture_recreation_public`: `public`, `sports_centre`, `cinema`, `theatre`, `arts_centre`, `exhibition_centre`

## Main Outputs

Outputs are written under `notebooks/Optimizor/outputs/`:

- `yixing_poi_cleaned_summary.csv`
- `yixing_candidates_whitelist.csv`
- `yixing_candidate_feature_counts.csv`
- `yixing_excluded_feature_counts.csv`
- `total_score.csv`
- `cache/conflict_pairs_50m.npy`
- `cache/conflict_pairs_50m.json`
- `yixing_selected_locations_xgb.csv`
- `yixing_selected_locations_mlp.csv`
- `yixing_selected_locations_svr.csv`
- `samples/candidate_sample_2000x10_seed277.npy`
- `samples/candidate_sample_2000x10_seed277.json`
- `runs/sampled_2000x10_seed277_d50m_loc5/yixing_sampled_optimization_summary.csv`
- `runs/sampled_2000x10_seed277_d50m_loc5/selected/yixing_selected_locations_<model>_sampleNN.csv`

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

Use the project environment directly:

```powershell
& "C:\Users\Yuan\Desktop\YUAN\OHCA 宜興市\yixin_env\python.exe" notebooks\Optimizor\run_this.py
```

Run the full pipeline, including all three optimization passes:

```powershell
& "C:\Users\Yuan\Desktop\YUAN\OHCA 宜興市\yixin_env\python.exe" notebooks\Optimizor\run_this.py
```

Build whitelist candidates as an independent step:

```powershell
& "C:\Users\Yuan\Desktop\YUAN\OHCA 宜興市\yixin_env\python.exe" notebooks\Optimizor\build_candidates.py
```

Calculate total scores as an independent step:

```powershell
& "C:\Users\Yuan\Desktop\YUAN\OHCA 宜興市\yixin_env\python.exe" notebooks\Optimizor\calculate_total_score.py
```

Build the cached 50m sparse conflict-pair file as an independent step. The default output is `outputs/cache/conflict_pairs_50m.npy`:

```powershell
& "C:\Users\Yuan\Desktop\YUAN\OHCA 宜興市\yixin_env\python.exe" notebooks\Optimizor\build_conflict_pairs.py
```

`run_this.py optimize --sample-size ...` automatically looks for `cache/conflict_pairs_<threshold>m.npy` under `--output-dir`, builds it once if missing, and filters it for each sample. Sample matrices go under `samples/`; selected CSVs and summary go under `runs/sampled_<size>x<count>_seed<seed>_d<threshold>_loc<loc_num>/`. Use `--sample-path`, `--conflict-pairs-path`, or `--run-dir` to override those defaults. Non-sampled optimization can still use the dense `cache/indicator_i_j_<threshold>m.npy` cache via `--indicator-path`.

Run optimization only against an existing `total_score.csv`:

```powershell
& "C:\Users\Yuan\Desktop\YUAN\OHCA 宜興市\yixin_env\python.exe" notebooks\Optimizor\run_this.py optimize --input notebooks\Optimizor\outputs\total_score.csv --score-col total_score_xgb --output notebooks\Optimizor\outputs\yixing_selected_locations_xgb.csv
& "C:\Users\Yuan\Desktop\YUAN\OHCA 宜興市\yixin_env\python.exe" notebooks\Optimizor\run_this.py optimize --input notebooks\Optimizor\outputs\total_score.csv --mlp --output notebooks\Optimizor\outputs\yixing_selected_locations_mlp.csv
& "C:\Users\Yuan\Desktop\YUAN\OHCA 宜興市\yixin_env\python.exe" notebooks\Optimizor\run_this.py optimize --input notebooks\Optimizor\outputs\total_score.csv --svr --output notebooks\Optimizor\outputs\yixing_selected_locations_svr.csv
```

Run 10 sampled optimization batches at the default 50m spacing. Each sample has 2000 unique candidates; candidates may repeat across different sample groups:

```powershell
& "C:\Users\Yuan\Desktop\YUAN\OHCA 宜興市\yixin_env\python.exe" notebooks\Optimizor\run_this.py optimize --input notebooks\Optimizor\outputs\total_score.csv --distance-threshold-m 50 --sample-size 2000 --sample-count 10 --sample-seed 277
```

This writes `samples/candidate_sample_2000x10_seed277.npy`, matching JSON metadata, `cache/conflict_pairs_50m.npy`, matching conflict-pair metadata, one selected-location CSV per sample/model under the run folder, and `yixing_sampled_optimization_summary.csv` in that same run folder.

Candidate generation does not choose a fixed K. Every POI whose `osm_tag` is in the whitelist is kept after coordinate cleaning and 10m deduplication. `run_this.py --loc-num K` only controls the final optimization stage.

## Tests

```powershell
& "C:\Users\Yuan\Desktop\YUAN\OHCA 宜興市\yixin_env\Scripts\python.exe" -m pytest tests\test_yixing_optimizer_utils.py notebooks\Optimizor\tests -q -p no:cacheprovider
```

