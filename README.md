
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

## 檔案結構與 Notebook 說明

根據儲存庫內的檔案清單，各程式碼檔案的功能如下：

### 數據獲取與預處理
* **`OSM download.ipynb`**: 構建數據框架並從 OSM 下載原始地理數據。
* **`tag_transformation.ipynb`**: 進行數據清洗、整理與標籤轉換。
* **`tag_site_selection.ipynb` & `Yixing_data_stat.ipynb`**: 處理站點選擇邏輯與宜興市基礎數據統計。

### 模型訓練與視覺化
* **`XGB_yixing.ipynb`**: XGBoost 模型的訓練、OHCA 風險預測及預測結果視覺化。
* **`NN_yixing.ipynb`**: MLP 模型的構建與預測熱力圖強化視覺化。
* **`SVR_yixing.ipynb`**: SVR 模型訓練與回歸分析視覺化。

### 數據文件
* **`cache/`**: 存放訓練好的模型與城市基礎數據。
* **`h3_l7_df_yixing.csv`**: 宜興市 H3 Level 7 六邊形網格的特徵矩陣。
* **`location_sites.csv` / `mapped_data.csv`**: 處理後的地理位置資訊與特徵映射表。

---

## 實驗結果摘要

* **特徵分佈**: 在宜興市中，零售 (Retail)、辦公室 (Office) 與餐廳 (Restaurant) 是數量最顯著的地理特徵。
* **風險地圖**: 專案產出了宜興市全域的 OHCA 預測熱力圖，視覺化展示了高風險區域的分佈。
* **特徵重要性**: 通過 SHAP 分析發現，特定的商業設施與建築密度對 OHCA 的預測有顯著的正向或負向影響。

---

## 未來工作 (Future Work)
1.  **功能區識別 (PCA)**: 利用主成分分析對建築分佈進行分類（如住宅區、商業區），以減緩數據分佈偏移 (Distribution Shift) 的問題。
2.  **細粒度數據校準**: 針對「公寓 (Apartment)」類別進行權重校準，以更準確地反映實際的建築分佈與人口密度。
3.  **效能驗證**: 引入 WorldPop 全球人口數據集，進一步驗證模型在不同人口分佈下的預測準確性。

---
**研究機構:** 清華大學 (Tsinghua University)  
**研究員:** Yang Chih Yuan  
**匯報日期:** 2025.12.07
