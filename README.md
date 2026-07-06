> [!NOTE]
> **[繁體中文](#繁體中文)** | **[English Version](#english-version)**
>
<a name="繁體中文"></a>
# 宜興市地理特徵預測院外心臟驟停 (OHCA) 分佈研究
# (Predicting OHCA Distribution Using Only Geographic Features in Yixing City)

## 專案簡介
本專案旨在探索僅使用**地理特徵 (Geographic Features, GF)** 來預測宜興市網格層級的院外心臟驟停 (OHCA) 風險。

由於傳統模型高度依賴難以獲取的人口統計學與醫療數據，本研究利用較易獲取的 OpenStreetMap (OSM) 原始數據、POI（興趣點）及建築資訊，來潛在表徵人口與醫療特徵，建立一個低門檻且高可擴展性的預測框架。

---

## 核心方法

本專案架構分為兩個主要模組：

### 1. 預測器 (Predictor)
使用六邊形網格 (Hexagonal Grid) 進行地理特徵預處理，並測試了三種機器學習模型：
* **XGBoost**: 基於樹狀結構的梯度提升模型，用於捕捉非線性關係。
* **MLP (多層感知器)**: 深度學習架構，用於學習複雜的特徵表示。
* **SVR (支持向量回歸)**: 傳統機器學習模型，用於回歸分析。

數據處理流程包含數據標準化 (Min-Max Normalization)，確保各特徵在相同的量級下運算。

### 2. 解釋器 (Interpreter)
利用**可解釋人工智慧 (XAI)** 技術，量化各項地理特徵對預測結果的貢獻度：
* **SHAP**: 提供全局與局部的特徵貢獻分析。

---

## 執行環境與建議順序

所有 Notebook 均使用 `yixin_env` kernel，並預期從 repository 根目錄啟動 Jupyter：

```bash
conda env create -f environment.yml
conda activate yixin_env
jupyter lab
```

建議依下列順序執行：

1. `notebooks/01_download_osm.ipynb`（需要重新下載 OSM 資料時）
2. `notebooks/02_select_poi_sites.ipynb`
3. `notebooks/03_transform_poi_tags.ipynb`
4. `notebooks/04_build_h3_features.ipynb`
5. `notebooks/05_data_summary.ipynb`
6. `notebooks/06_train_xgboost.ipynb`
7. `notebooks/07_train_svr.ipynb`
8. `notebooks/08_train_mlp.ipynb`
9. `notebooks/09_prepare_population.ipynb`
10. `notebooks/10_compare_models_and_population.ipynb`

模型與統計 Notebook 使用的 `h3_l7_df_new.csv` 和中間版本 `h3_l7_df_yixing.csv` 位於 `data/processed/`。人口 Notebook 另需本地檔案 `data/raw/Yixing_data/chn_pop_2024_CN_100m_R2025A_v1.tif`。

## 檔案結構與 Notebook 說明

根據儲存庫內的檔案清單，各程式碼檔案的功能如下：

### 數據獲取與預處理
* **`notebooks/01_download_osm.ipynb`**: 構建數據框架並從 OSM 下載原始地理數據。
* **`notebooks/03_transform_poi_tags.ipynb`**: 進行數據清洗、整理與標籤轉換。
* **`notebooks/02_select_poi_sites.ipynb` & `notebooks/05_data_summary.ipynb`**: 處理站點選擇邏輯與宜興市基礎數據統計。`notebooks/04_build_h3_features.ipynb` 建立 H3 特徵矩陣。

### 模型訓練與視覺化
* **`notebooks/06_train_xgboost.ipynb`**: XGBoost 模型的訓練、OHCA 風險預測及預測結果視覺化。
* **`notebooks/08_train_mlp.ipynb`**: MLP 模型的構建與預測熱力圖強化視覺化。
* **`notebooks/07_train_svr.ipynb`**: SVR 模型訓練與回歸分析視覺化。`notebooks/08_train_mlp.ipynb` 內的跨模型比較會使用其輸出。

### 人口與模型比較
* **`notebooks/09_prepare_population.ipynb`**: 將人口 raster 彙整到 H3 網格。
* **`notebooks/10_compare_models_and_population.ipynb`**: 比較模型預測與人口資料。
* **`archive/virginia_beach_shap.ipynb`**: 封存的 Virginia Beach SHAP 分析，不屬於主要執行流程。

### 數據文件
* **`data/raw/` / `data/interim/`**: 本地原始資料與中間資料；兩者不納入 Git。
* **`data/processed/h3_l7_df_new.csv` / `data/processed/h3_l7_df_yixing.csv`**: 模型訓練來源與宜興市 H3 Level 7 中間特徵矩陣。
* **`data/processed/h3_l7_df_yixing_full.csv`**: 模型使用的宜興市完整特徵矩陣。
* **`data/interim/location_sites.csv` / `data/interim/mapped_data.csv`**: 處理後的地理位置資訊與特徵映射表。

---

## 實驗結果摘要

* **特徵分佈**: 在宜興市中，零售 (Retail)、辦公室 (Office) 與餐廳 (Restaurant) 是數量最顯著的地理特徵。
* **風險地圖**: 專案產出了宜興市全域的 OHCA 預測熱力圖，視覺化展示了高風險區域的分佈。
* **特徵重要性**: 通過 SHAP 分析發現，特定的商業設施與建築密度對 OHCA 的預測有顯著的正向或負向影響。

###  數據視覺化 (Data Visualization)

<p align="center">
  <b>宜興市建築分佈圖</b><br>
  <img src="outputs/figures/Yixing%20building%20distribution.png" width="50%">
</p>

<hr>

<p align="center">
  <b>模型預測結果比較</b>
</p>

<table style="width: 100%; table-layout: fixed;">
  <tr>
    <td align="center">
      <img src="outputs/figures/MLP%20predict.png" style="width:100%"><br>
      <b>MLP Prediction</b>
    </td>
    <td align="center">
      <img src="outputs/figures/SVR%20predict.png" style="width:100%"><br>
      <b>SVR Prediction</b>
    </td>
    <td align="center">
      <img src="outputs/figures/XGB%20predict.png" style="width:100%"><br>
      <b>XGB Prediction</b>
    </td>
  </tr>
</table>
---

## 未來工作 (Future Work)
1.  **功能區識別 (PCA)**: 利用主成分分析對建築分佈進行分類（如住宅區、商業區），以減緩數據分佈偏移 (Distribution Shift) 的問題。
2.  **細粒度數據校準**: 針對「公寓 (Apartment)」類別進行權重校準，以更準確地反映實際的建築分佈與人口密度。
3.  **效能驗證**: 引入 WorldPop 全球人口數據集，進一步驗證模型在不同人口分佈下的預測準確性。


<a name="english-version"></a>
# Predicting Out-of-Hospital Cardiac Arrest (OHCA) Distribution Using Only Geographic Features in Yixing City

## Project Introduction
This project explores the feasibility of predicting grid-level Out-of-Hospital Cardiac Arrest (OHCA) risk in Yixing City using only **Geographic Features (GF)**. 

Traditional predictive models often rely heavily on demographic or medical data, which are frequently difficult to acquire. This research leverages more accessible data—OpenStreetMap (OSM) raw data, Points of Interest (POI), and building information—to potentially represent underlying demographic and medical characteristics.

---

## Core Methodology

The project is structured into two primary modules:

### 1. Predictor
This module focuses on predicting grid-level OHCA risk using geographic data.
* **Preprocessing**: Raw OSM data is processed and aggregated into hexagonal grid-level features.
* **Normalization**: All input features are normalized using Min-Max scaling to a range of [0, 1].
* **Machine Learning Models**: Three models are evaluated for their predictive performance:
    * **XGBoost**: A tree-based gradient boosting model used for capturing non-linear relationships.
    * **MLP (Multi-Layer Perceptron)**: A neural network architecture used for learning complex feature representations.
    * **SVR (Support Vector Regression)**: A regression-based machine learning model.

### 2. Interpreter
This module uses **Explainable AI (XAI)** techniques to quantify the contribution of each geographic feature to the predicted OHCA risk.
* **SHAP**: Used to provide both global and local feature contribution analysis.
* **SP-LIME**: Employed for localized interpretation of model predictions.

---

## Environment and Recommended Execution Order

All notebooks use the `yixin_env` kernel and expect Jupyter to be started from the repository root:

```bash
conda env create -f environment.yml
conda activate yixin_env
jupyter lab
```

Recommended execution order:

1. `notebooks/01_download_osm.ipynb` (when the OSM data must be downloaded again)
2. `notebooks/02_select_poi_sites.ipynb`
3. `notebooks/03_transform_poi_tags.ipynb`
4. `notebooks/04_build_h3_features.ipynb`
5. `notebooks/05_data_summary.ipynb`
6. `notebooks/06_train_xgboost.ipynb`
7. `notebooks/07_train_svr.ipynb`
8. `notebooks/08_train_mlp.ipynb`
9. `notebooks/09_prepare_population.ipynb`
10. `notebooks/10_compare_models_and_population.ipynb`

The model and statistics notebooks read `h3_l7_df_new.csv` and the intermediate `h3_l7_df_yixing.csv` from `data/processed/`. The population notebook additionally requires the local file `data/raw/Yixing_data/chn_pop_2024_CN_100m_R2025A_v1.tif`.

## File Structure and Notebooks

Based on the repository's file list, the notebooks are organized as follows:

### Data Acquisition and Preprocessing
* **`notebooks/01_download_osm.ipynb`**: Framework construction and raw data download from OpenStreetMap.
* **`notebooks/03_transform_poi_tags.ipynb`**: Data cleaning, tidying, and label transformation.
* **`notebooks/02_select_poi_sites.ipynb` & `notebooks/05_data_summary.ipynb`**: Site selection and Yixing statistics. `notebooks/04_build_h3_features.ipynb` builds the H3 feature matrix.

### Model Training and Visualization
* **`notebooks/06_train_xgboost.ipynb`**: Training, prediction, and reinforced visualization for the XGBoost model.
* **`notebooks/08_train_mlp.ipynb`**: Construction and heatmap visualization for the Neural Network (MLP) model.
* **`notebooks/07_train_svr.ipynb`**: SVR training and regression visualization. Its output is used by the cross-model cells in `notebooks/08_train_mlp.ipynb`.

### Population and Model Comparison
* **`notebooks/09_prepare_population.ipynb`**: Aggregates the population raster to H3 grids.
* **`notebooks/10_compare_models_and_population.ipynb`**: Compares model predictions with population data.
* **`archive/virginia_beach_shap.ipynb`**: Archived Virginia Beach SHAP analysis; it is not part of the main workflow.

### Data Files
* **`data/raw/` / `data/interim/`**: Local raw and intermediate data; both are excluded from Git.
* **`data/processed/h3_l7_df_new.csv` / `data/processed/h3_l7_df_yixing.csv`**: Source-model and intermediate Yixing H3 Level 7 feature matrices.
* **`data/processed/h3_l7_df_yixing_full.csv`**: Complete Yixing feature matrix used by the models.
* **`data/interim/location_sites.csv` / `data/interim/mapped_data.csv`**: Processed location information and feature mapping results.

---

## Experimental Results Summary

* **Feature Distribution**: In Yixing City, the most prominent geographic features are Retail, Office, and Restaurant.
* **Risk Mapping**: The project generated heatmaps showing the distribution of predicted OHCA risk across the city, allowing for visual comparison between models.
* **Feature Importance**: SHAP analysis revealed that specific commercial facilities and building densities have significant impacts on the model's output.

### Data Visualization

<p align="center">
  <b>Building Distribution Map of Yixing City</b><br>
  <img src="outputs/figures/Yixing%20building%20distribution.png" width="50%">
</p>

<hr>

<p align="center">
  <b>Comparison of Model Prediction Results</b>
</p>

<table style="width: 100%; table-layout: fixed;">
  <tr>
    <td align="center">
      <img src="outputs/figures/MLP%20predict.png" style="width:100%"><br>
      <b>MLP Prediction</b>
    </td>
    <td align="center">
      <img src="outputs/figures/SVR%20predict.png" style="width:100%"><br>
      <b>SVR Prediction</b>
    </td>
    <td align="center">
      <img src="outputs/figures/XGB%20predict.png" style="width:100%"><br>
      <b>XGB Prediction</b>
    </td>
  </tr>
</table>
---

## Future Work
1. **Functional Zone Identification (PCA)**: Use Principal Component Analysis on building distributions to classify functional zones (e.g., Residential, Commercial) to mitigate distribution shift.
2. **Fine-grained Data Refinement**: Calibrate the "Apartment" category—which currently represents residential complexes—by applying a multiplier to more accurately reflect building density.
3. **Validation**: Use the WorldPop Global 2 dataset to validate prediction accuracy against actual population distributions.
