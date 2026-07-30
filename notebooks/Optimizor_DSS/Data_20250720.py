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

        self.loc_lat = None
        self.loc_lon = None
        self.loc_score = None
        self.loc_score_mlp = None
        

        self.dist_i_j = {}
        self.indicator_i_j = {}
        self.infinite = 0
        

        self.build_num = 0
        self.loc_num = 0
        self.dist_limit = 0
    
    def read_npy(self, file_name, file_name_ohca, dist_limit, build_num):
        # self.loc_num = loc_num
        self.dist_limit = dist_limit
        self.build_num = build_num
        self.file_name_ohca = file_name_ohca


        df = pd.read_csv(file_name)
        self.loc_lat = df['lat'].values
        self.loc_lon = df['lon'].values
        self.loc_score = df['total_score'].values
        self.loc_score_mlp = df['total_score_mlp'].values


        df_ohca = pd.read_csv(file_name_ohca)
        self.ohca_lat = df_ohca['Latitude'].values
        self.ohca_lon = df_ohca['Longitude'].values
        

        if dist_limit == 0.6:
            self.indicator_i_j = self._load_indicator('indicator_i_j_0_6.npy')
        elif dist_limit == 0.8:
            self.indicator_i_j = self._load_indicator('indicator_i_j_0_8.npy')
        elif dist_limit == 0.96:
            self.indicator_i_j = self._load_indicator('indicator_i_j.npy')
        elif dist_limit == 1.2:
            self.indicator_i_j = self._load_indicator('indicator_i_j_1_2.npy')
        elif dist_limit == 1:
            self.indicator_i_j = self._load_indicator('indicator_i_j_1.npy')
        elif dist_limit == 1.4:
            self.indicator_i_j = self._load_indicator('indicator_i_j_1_4.npy')
        elif dist_limit == 1.6:
            self.indicator_i_j = self._load_indicator('indicator_i_j_1_6.npy')
        else:
            self.infinite = 1
        
    def _load_indicator(self, file_name):
        indicator = np.load(file_name)
        if self._indicator_marks_far_locations(indicator):
            return 1 - indicator
        return indicator

    def _indicator_marks_far_locations(self, indicator):
        if self.dist_limit <= 0 or getattr(indicator, 'ndim', 0) < 2:
            return False

        max_index = min(
            self.build_num,
            len(self.loc_lat),
            len(self.loc_lon),
            indicator.shape[0],
            indicator.shape[1],
        )
        if max_index < 2:
            return False

        sample_count = min(max_index, 25)
        sample_ids = np.linspace(0, max_index - 1, sample_count, dtype=int)
        current_evidence = 0
        inverted_evidence = 0

        for pos, i in enumerate(sample_ids[:-1]):
            for j in sample_ids[pos + 1:]:
                value = indicator[i, j]
                is_close = self.haversine(self.loc_lat[i], self.loc_lon[i], self.loc_lat[j], self.loc_lon[j]) <= self.dist_limit
                if (value == 1 and is_close) or (value == 0 and not is_close):
                    current_evidence += 1
                elif (value == 1 and not is_close) or (value == 0 and is_close):
                    inverted_evidence += 1

        return inverted_evidence > current_evidence
    def read_data(self, file_name, file_name_ohca, loc_num, dist_limit, build_num):
        self.loc_num = loc_num
        self.dist_limit = dist_limit
        self.build_num = build_num
        self.file_name_ohca = file_name_ohca
        

        df = pd.read_csv(file_name)
        self.loc_lat = df['lat'].values
        self.loc_lon = df['lon'].values
        self.loc_score = df['score'].values


        df_ohca = pd.read_csv(file_name_ohca)
        self.ohca_lat = df_ohca['Latitude'].values
        self.ohca_lon = df_ohca['Longitude'].values
        

        for i in range(self.build_num):
            for j in range(self.build_num):
                if i != j:
                    lat1, lon1 = self.loc_lat[i], self.loc_lon[i]
                    lat2, lon2 = self.loc_lat[j], self.loc_lon[j]
                    dist = self.haversine(lat1, lon1, lat2, lon2)
                    self.dist_i_j[i, j] = dist


                    self.indicator_i_j[i, j] = 1 if dist <= self.dist_limit else 0
    
    def haversine(self, lat1, lon1, lat2, lon2):

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        

        d_lat = lat2 - lat1
        d_lon = lon2 - lon1
        a = math.sin(d_lat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2)**2
        c = 2 * math.asin(math.sqrt(a))
        

        r = 6371
        return c * r                    
                    
                    
                    
                    
                    
                    
