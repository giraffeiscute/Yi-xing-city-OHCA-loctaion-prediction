"""
AED_deployment_model
Author: Kexin Cao
Institute: Tsinghua University
Date: 20241220 
"""

import ModelBuilder_20250720 as ModelBuilder
import pandas as pd
import math
import numpy as np

class Data(object):
    def __init__(self):
        # 初始化空的地理位置資料與分數變數
        self.loc_lat = None            # 所有候選地點的緯度
        self.loc_lon = None            # 所有候選地點的經度
        self.loc_score = None          # 每個地點的評分（可能與緊急需求相關）
        self.loc_score_mlp = None      # 使用 MLP 模型算出的評分，用於 S_p^{mlp}
        
        # 初始化距離與指標的字典
        self.dist_i_j = {}             # i 到 j 的實際距離（公里）
        self.indicator_i_j = {}        # i 到 j 是否超出距離限制（1 表示超過，0 表示沒超過）
        self.infinite = 0              # 標記是否無法載入指標檔案（未命中特定距離）
        
        # 初始化地點與建築物數目相關參數
        self.build_num = 0             # 建築物數量
        self.loc_num = 0               # 候選地點數量
        self.dist_limit = 0            # 距離閾值限制（公里）
    
    def read_npy(self, file_name, file_name_ohca, build_num):
        # self.loc_num = loc_nums
        self.build_num = build_num
        self.file_name_ohca = file_name_ohca

        # 讀取候選地點的經緯度與兩種評分（原始與 MLP 模型算出來的）
        df = pd.read_csv(file_name)
        self.loc_lat = df['lat'].values
        self.loc_lon = df['lon'].values
        self.loc_score = df['total_score'].values
        self.loc_score_mlp = df['total_score_mlp'].values

        # 讀取實際 OHCA 發生點的經緯度（用於後續比較或分析）
        df_ohca = pd.read_csv(file_name_ohca)
        self.ohca_lat = df_ohca['Latitude'].values
        self.ohca_lon = df_ohca['Longitude'].values
        
        
    def read_data(self, file_name, file_name_ohca, loc_num, dist_limit, build_num):
        self.loc_num = loc_num
        self.dist_limit = dist_limit
        self.build_num = build_num
        self.file_name_ohca = file_name_ohca
        
        # 讀取候選地點的經緯度與評分（這裡用的是 "score"，不是 "total_score"）
        df = pd.read_csv(file_name)
        self.loc_lat = df['lat'].values
        self.loc_lon = df['lon'].values
        self.loc_score = df['score'].values

        # 讀取實際 OHCA 發生點的位置資料
        df_ohca = pd.read_csv(file_name_ohca)
        self.ohca_lat = df_ohca['Latitude'].values
        self.ohca_lon = df_ohca['Longitude'].values
        
        # 計算任意兩個地點 i, j 的距離，並根據是否超出距離限制生成指標
        for i in range(self.build_num):
            for j in range(self.build_num):
                if i != j:
                    lat1, lon1 = self.loc_lat[i], self.loc_lon[i]
                    lat2, lon2 = self.loc_lat[j], self.loc_lon[j]
                    dist = self.haversine(lat1, lon1, lat2, lon2)
                    self.dist_i_j[i, j] = dist

                    # 根據是否超過距離閾值設定 indicator
                    self.indicator_i_j[i, j] = 0 if dist <= self.dist_limit else 1
    
    def haversine(self, lat1, lon1, lat2, lon2):
        # 将角度转换为弧度
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine公式
        d_lat = lat2 - lat1
        d_lon = lon2 - lon1
        a = math.sin(d_lat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # 地球半径（单位：公里）
        r = 6371
        return c * r                    
                    
                    
                    
                    
                    
                    