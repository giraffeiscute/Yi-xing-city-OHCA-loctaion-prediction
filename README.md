> [!NOTE]
> **[繁體中文](#繁體中文)** | **[English Version](#english-version)**
>

<a name="繁體中文"></a>
# 宜興市地理特徵預測院外心臟驟停（OHCA）分佈研究
# Predicting OHCA Distribution Using Only Geographic Features in Yixing City

## 專案簡介

本專案建構一套面向公共安全場景的 AI 決策支援系統，用於在缺乏完整醫療與人口統計資料的城市中，預測院外心臟驟停（Out-of-Hospital Cardiac Arrest, OHCA）高風險區域，並支援後續 AED 資源配置與部署策略規劃。

專案重點不僅在於預測準確率，更著重於提升 AI 決策系統的 可解釋性、可信度與落地可用性。透過 SHAP 等可解釋 AI 方法，分析不同地理特徵對 OHCA 風險預測的影響，協助決策者理解模型判斷依據，降低黑盒模型在公共安全決策中的導入門檻。

## 專案目標

本專案主要解決三個問題：

1. **資料取得困難**：在缺乏完整 OHCA、人口與醫療資料的城市中，利用公開地理特徵建立風險預測模型。
2. **模型可信度不足**：透過 SHAP 分析地理環境、建築密度與 POI 類型對預測結果的影響，提高模型透明度。
3. **AI 難以落地決策**：將模型輸出轉化為風險地圖與決策依據，支援 AED 部署與城市應急管理。

## 技術方法

本專案以六邊形網格作為城市空間分析單元，將 OSM 原始地理資料、POI 資訊與建築特徵聚合至網格層級，形成可用於機器學習的城市特徵矩陣。

在預測模型方面，專案比較了三種機器學習方法：

* **XGBoost**：用於捕捉非線性特徵關係，並提供穩定的基準模型表現。
* **MLP（Multi-Layer Perceptron）**：用於學習複雜的城市空間特徵表示。
* **SVR（Support Vector Regression）**：作為傳統回歸模型，用於比較不同模型在空間風險預測任務上的泛化能力。

在模型解釋方面，專案導入 **SHAP（SHapley Additive exPlanations）**，分析不同地理特徵對 OHCA 風險預測的正向或負向貢獻，協助識別影響風險分佈的城市功能特徵。

## 核心貢獻

### 1. 建立低資料依賴的城市 OHCA 風險預測框架

本專案不依賴高度敏感或難以取得的醫療與人口資料，而是利用公開地理特徵建構城市級風險預測模型，使方法更容易遷移至資料不足的城市。

### 2. 提升 AI 決策系統的可解釋性與可信度

透過 SHAP 分析模型特徵貢獻，將黑盒預測結果轉化為可理解的城市風險因素說明，協助決策者判斷模型是否符合實際城市結構與部署邏輯。

### 3. 支援跨城市、跨單位 AI 應用落地

本專案與深圳市、宜興市政府合作推動 PoC 驗證與部署策略規劃，將 OHCA 風險熱力圖轉化為 AED 資源配置與城市應急管理的決策依據，服務覆蓋約 **1,868 萬名居民**。

### 4. 將模型結果轉化為公共安全決策工具

專案產出宜興市 OHCA 風險預測熱力圖、模型比較結果與特徵重要性分析，協助決策者快速辨識高風險區域，作為 AED 佈建與公共安全規劃的基礎。


## 檔案結構與 Notebook 說明

根據儲存庫內的檔案清單，各程式碼檔案的功能如下：

### 數據獲取與預處理

* **`notebooks/01_download_osm.ipynb`**：構建數據框架並從 OSM 下載原始地理數據。
* **`notebooks/03_transform_poi_tags.ipynb`**：進行數據清洗、整理與標籤轉換。
* **`notebooks/02_select_poi_sites.ipynb` & `notebooks/05_data_summary.ipynb`**：處理站點選擇邏輯與宜興市基礎數據統計。
* **`notebooks/04_build_h3_features.ipynb`**：建立 H3 特徵矩陣。

### 模型訓練與視覺化

* **`notebooks/06_train_xgboost.ipynb`**：XGBoost 模型的訓練、OHCA 風險預測及預測結果視覺化。
* **`notebooks/08_train_mlp.ipynb`**：MLP 模型的構建與預測熱力圖視覺化。
* **`notebooks/07_train_svr.ipynb`**：SVR 模型訓練與回歸分析視覺化；`notebooks/08_train_mlp.ipynb` 內的跨模型比較會使用其輸出。

### 人口與模型比較

* **`notebooks/09_prepare_population.ipynb`**：將人口 raster 彙整到 H3 網格。
* **`notebooks/10_compare_models_and_population.ipynb`**：比較模型預測與人口資料。
* **`archive/virginia_beach_shap.ipynb`**：封存的 Virginia Beach SHAP 分析，不屬於主要執行流程。

### 數據文件

* **`data/raw/` / `data/interim/`**：本地原始資料與中間資料；兩者不納入 Git。
* **`data/processed/h3_l7_df_new.csv` / `data/processed/h3_l7_df_yixing.csv`**：模型訓練來源與宜興市 H3 Level 7 中間特徵矩陣。
* **`data/processed/h3_l7_df_yixing_full.csv`**：模型使用的宜興市完整特徵矩陣。
* **`data/interim/location_sites.csv` / `data/interim/mapped_data.csv`**：處理後的地理位置資訊與特徵映射表。

---

## 實驗結果摘要

* **特徵分佈**：在宜興市中，零售（Retail）、辦公室（Office）與餐廳（Restaurant）是數量最顯著的地理特徵。
* **風險地圖**：專案產出了宜興市全域的 OHCA 預測熱力圖，視覺化展示高風險區域分佈。
* **特徵重要性**：SHAP 分析顯示，特定商業設施與建築密度對 OHCA 風險預測具有明顯影響。

### 數據視覺化（Data Visualization）

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

## 未來工作（Future Work）

1. **功能區識別（PCA）**：利用主成分分析對建築分佈進行分類，例如住宅區、商業區，以減緩數據分佈偏移（Distribution Shift）的問題。
2. **細粒度數據校準**：針對「公寓（Apartment）」類別進行權重校準，以更準確地反映實際建築分佈與人口密度。
3. **效能驗證**：引入 WorldPop 全球人口數據集，進一步驗證模型在不同人口分佈下的預測準確性。


<a name="english-version"></a>
# Predicting Out-of-Hospital Cardiac Arrest (OHCA) Distribution Using Only Geographic Features in Yixing City

## Project Introduction

This project develops an AI decision-support system for public safety applications. It aims to predict high-risk areas of Out-of-Hospital Cardiac Arrest (OHCA) in cities where complete medical and demographic data are difficult to obtain, while also supporting subsequent AED resource allocation and deployment strategy planning.

The project focuses not only on predictive accuracy, but also on improving the explainability, trustworthiness, and practical usability of AI-based decision support. By applying explainable AI methods such as SHAP, the project analyzes how different geographic features influence OHCA risk prediction, helping decision-makers understand the model’s reasoning and reducing the barriers to adopting black-box models in public safety decision-making.

## Project Objectives

This project addresses three main challenges:

1. **Limited data availability**: Build OHCA risk prediction models using public geographic features when local OHCA, demographic, and medical data are incomplete or unavailable.
2. **Lack of model trustworthiness**: Use SHAP to explain how geographic environments, building density, and POI categories influence predicted OHCA risk.
3. **Difficulty in real-world AI deployment**: Convert model outputs into risk maps and decision-support evidence for AED deployment and urban emergency management.

## Technical Approach

The project uses hexagonal grids as spatial analysis units. Raw OSM data, POI information, and building features are aggregated into grid-level urban feature matrices for machine learning.

For prediction, three machine learning models are compared:

* **XGBoost**: Captures non-linear feature relationships and provides a stable baseline.
* **MLP (Multi-Layer Perceptron)**: Learns complex representations of urban spatial features.
* **SVR (Support Vector Regression)**: Serves as a traditional regression model for comparing generalization ability in spatial risk prediction.

For interpretation, the project applies **SHAP (SHapley Additive exPlanations)** to analyze the positive and negative contributions of geographic features to OHCA risk prediction, helping identify urban functional features associated with risk distribution.

## Core Contributions

### 1. A low-data-dependency framework for urban OHCA risk prediction

The project avoids relying on sensitive or hard-to-access medical and demographic data. Instead, it uses public geographic features to build city-level risk prediction models that can be transferred to data-limited cities.

### 2. Improved explainability and trustworthiness for AI decision support

SHAP analysis converts black-box model predictions into interpretable urban risk factors, helping decision-makers assess whether model outputs align with real urban structures and deployment logic.

### 3. Support for cross-city and cross-agency AI deployment

In collaboration with the governments of Shenzhen and Yixing, the project supports PoC validation and deployment strategy planning. The predicted OHCA risk heatmaps are converted into decision evidence for AED resource allocation and urban emergency management, covering approximately **18.68 million residents**.

### 4. Transformation of model outputs into public safety decision tools

The project produces OHCA risk heatmaps, model comparison results, and feature importance analysis for Yixing City, helping decision-makers identify high-risk areas and support AED placement and public safety planning.



## File Structure and Notebooks

Based on the repository's file list, the notebooks are organized as follows:

### Data Acquisition and Preprocessing

* **`notebooks/01_download_osm.ipynb`**: Builds the data framework and downloads raw geographic data from OpenStreetMap.
* **`notebooks/03_transform_poi_tags.ipynb`**: Performs data cleaning, tidying, and tag transformation.
* **`notebooks/02_select_poi_sites.ipynb` & `notebooks/05_data_summary.ipynb`**: Handles site selection logic and basic Yixing statistics.
* **`notebooks/04_build_h3_features.ipynb`**: Builds the H3 feature matrix.

### Model Training and Visualization

* **`notebooks/06_train_xgboost.ipynb`**: Trains the XGBoost model, predicts OHCA risk, and visualizes prediction results.
* **`notebooks/08_train_mlp.ipynb`**: Builds the MLP model and visualizes predicted risk heatmaps.
* **`notebooks/07_train_svr.ipynb`**: Trains the SVR model and visualizes regression results; its output is used by the cross-model comparison cells in `notebooks/08_train_mlp.ipynb`.

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
* **Risk Mapping**: The project generated OHCA prediction heatmaps for Yixing City, visualizing the spatial distribution of high-risk areas.
* **Feature Importance**: SHAP analysis shows that specific commercial facilities and building density features have noticeable impacts on predicted OHCA risk.

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

1. **Functional Zone Identification (PCA)**: Use Principal Component Analysis on building distributions to classify functional zones, such as residential and commercial areas, to mitigate distribution shift.
2. **Fine-grained Data Calibration**: Calibrate the "Apartment" category to more accurately reflect actual building distribution and population density.
3. **Performance Validation**: Introduce the WorldPop global population dataset to further validate prediction accuracy under different population distributions.
