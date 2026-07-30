"""
AED_deployment_model
Author: Kexin Cao
Institute: Tsinghua University
Date: 20241220 
"""

import Data_20250720 as Data
import ModelBuilder_20250720 as ModelBuilder
from gurobipy import *
import random
import numpy as np
import pandas as pd
import csv

if __name__ == "__main__":
    
    file_name = 'test_poi_df_NNtotal_20250415.csv'
    file_name_ohca = 'ohca_df.csv'


    # loc_num = 5
    loc_num_list = [5,10,20,40,60,80,100]
    # loc_num_list = [100]


    dist_limit = 0.96


    # dist_list = [0.6,0.8,0.96,1.2,0]
    # dist_list = [0.96]
    dist_list = [0,0.6,0.8,0.96,1,1.2,1.4,1.6]
    # dist_list = [0.8, 0.96, 1, 1.2]


    build_num = 5000
    build_num_choose_cnt = 10

    # 總建築物??
    build_sum = 99724


    ohca_cover_random_cnt = 10
    

    # random_candidate_loc_id = []
    # for cnt in range(build_num_choose_cnt):
    #     random_candidate_loc_id.append(random.choices(range(build_sum), k=build_num))
    # df = pd.DataFrame(random_candidate_loc_id).T
    # output_name = 'build_num_choose_' + str(build_num) + '.csv'
    # df.to_csv(output_name, index=False)    


    for dist in dist_list:
        data = Data.Data()
        # data.read_data(file_name,file_name_ohca,loc_num=loc_num,dist_limit=dist_limit,build_num=build_num)   
        data.read_npy(file_name,file_name_ohca,dist_limit=dist,build_num=build_num)
        print('read finishing !')
        # data = np.load('indicator_i_j.npy')


        for loc_num in loc_num_list:

            output_name = 'build_num_choose_100_set_' + str(build_num_choose_cnt) + '.npy'
            random_candidate_loc_id = np.load(output_name).astype(np.int64)

            deploy_decision_coverage_shap = np.zeros(build_num_choose_cnt)
            deploy_decision_shap = np.zeros((build_num_choose_cnt, loc_num))
            deploy_decision_survivalrate_shap = np.zeros((build_num_choose_cnt,len(data.ohca_lat)))
            deploy_decision_survivalrate_average_shap = np.zeros(build_num_choose_cnt)

            deploy_decision_coverage_mlp = np.zeros(build_num_choose_cnt)
            deploy_decision_mlp = np.zeros((build_num_choose_cnt, loc_num))
            deploy_decision_survivalrate_mlp = np.zeros((build_num_choose_cnt,len(data.ohca_lat)))
            deploy_decision_survivalrate_average_mlp = np.zeros(build_num_choose_cnt)

            for cnt in range(build_num_choose_cnt):
                model_handler = ModelBuilder.ModelBuilder()
                candidate_loc_id = random_candidate_loc_id[cnt]


                print('IP-SHAP-dist-',dist,'cnt-',cnt)
                model_handler.build_IP(data,candidate_loc_id,loc_num)


                print('{} = {}'.format('obj_val', model_handler.obj_val), end='')
                print()
                print('{} = {}'.format('run_time', model_handler.run_time), end='')
                print()
                print('{} = {}'.format('deployment_decision', model_handler.deploy_decision), end='')
                print()
                deploy_decision_shap[cnt] = np.array(model_handler.deploy_decision)
                
                ohca_cover_cnt_predict = 0

                for ohca in range(len(data.ohca_lat)):
                    iscover = 0
                    min_dist = np.inf
                    for i in range(len(model_handler.deploy_decision)):
                        loc = model_handler.deploy_decision[i]

                        dist_loc_ohca = data.haversine(data.loc_lat[loc],data.loc_lon[loc],data.ohca_lat[ohca],data.ohca_lon[ohca])
                        if dist_loc_ohca <= min_dist:
                            min_dist = dist_loc_ohca
                        if dist_loc_ohca <= dist:
                            iscover = 1

                    t_loc_ohca = min_dist / 300 * 1000

                    if t_loc_ohca <= 4:
                        s_loc_ohca = (1 + np.exp(-0.26 + 0.106 * t_loc_ohca + 0.139 * 10.5))**(-1)
                    else:
                        s_loc_ohca = 0  
                    deploy_decision_survivalrate_shap[cnt][ohca] = s_loc_ohca
                    if iscover == 1:
                        ohca_cover_cnt_predict += 1
                deploy_decision_coverage_shap[cnt] = ohca_cover_cnt_predict
                print(deploy_decision_survivalrate_shap[cnt])
                deploy_decision_survivalrate_average_shap[cnt] = np.average(deploy_decision_survivalrate_shap[cnt])   


                print('IP-MLP-dist-', dist, 'cnt-', cnt)
                model_handler.build_IP_mlp(data,candidate_loc_id,loc_num)


                print('{} = {}'.format('obj_val', model_handler.obj_val_mlp), end='')
                print()
                print('{} = {}'.format('run_time', model_handler.run_time_mlp), end='')
                print()
                print('{} = {}'.format('deployment_decision', model_handler.deploy_decision_mlp), end='')
                print()
                deploy_decision_mlp[cnt] = np.array(model_handler.deploy_decision_mlp)   
                ohca_cover_cnt_predict = 0
                for ohca in range(len(data.ohca_lat)):
                    iscover = 0
                    min_dist = np.inf
                    for i in range(len(model_handler.deploy_decision_mlp)):
                        loc = model_handler.deploy_decision_mlp[i]

                        dist_loc_ohca = data.haversine(data.loc_lat[loc],data.loc_lon[loc],data.ohca_lat[ohca],data.ohca_lon[ohca])
                        if dist_loc_ohca <= min_dist:
                            min_dist = dist_loc_ohca
                        if dist_loc_ohca <= dist:
                            iscover = 1


                    t_loc_ohca = min_dist / 300 * 1000
                    if t_loc_ohca <= 4:
                        s_loc_ohca = (1 + np.exp(-0.26 + 0.106 * t_loc_ohca + 0.139 * 10.5))**(-1)
                    else:
                        s_loc_ohca = 0  
                    deploy_decision_survivalrate_mlp[cnt][ohca] = s_loc_ohca
                    if iscover == 1:
                        ohca_cover_cnt_predict += 1   
                deploy_decision_coverage_mlp[cnt] = ohca_cover_cnt_predict
                print(deploy_decision_survivalrate_mlp[cnt])
                deploy_decision_survivalrate_average_mlp[cnt] = np.average(deploy_decision_survivalrate_mlp[cnt])
            
 
            output_folder = 'results_mlp_20250720/'
            ##%% np.save for IP-SHAP
            output_name = output_folder+'shap/deployment_decision_' + str(loc_num)+ '_set_' + str(build_num_choose_cnt)  + '_dist_' + str(dist) + '.npy'
            np.save(output_name,deploy_decision_shap)   
            output_name = output_folder+'shap/deployment_decision_coverage_' + str(loc_num)+ '_set_' + str(build_num_choose_cnt) + '_dist_' + str(dist) + '.npy'
            np.save(output_name,deploy_decision_coverage_shap) 
            output_name = output_folder+'shap/deploy_decision_survivalrate_' + str(loc_num)+ '_set_' + str(build_num_choose_cnt) + '_dist_' + str(dist) + '.npy'
            np.save(output_name,deploy_decision_survivalrate_shap)
            output_name = output_folder+'shap/deploy_decision_survivalrate_average_' + str(loc_num)+ '_set_' + str(build_num_choose_cnt) + '_dist_' + str(dist) + '.npy'
            np.save(output_name,deploy_decision_survivalrate_average_shap)

            ##%% np.save for IP-MLP
            output_name = output_folder+'mlp/deployment_decision_' + str(loc_num)+ '_set_' + str(build_num_choose_cnt)  + '_dist_' + str(dist) + '.npy'
            np.save(output_name,deploy_decision_mlp)   
            output_name = output_folder+'mlp/deployment_decision_coverage_' + str(loc_num)+ '_set_' + str(build_num_choose_cnt) + '_dist_' + str(dist) + '.npy'
            np.save(output_name,deploy_decision_coverage_mlp) 
            output_name = output_folder+'mlp/deploy_decision_survivalrate_' + str(loc_num)+ '_set_' + str(build_num_choose_cnt) + '_dist_' + str(dist) + '.npy'
            np.save(output_name,deploy_decision_survivalrate_mlp)
            output_name = output_folder+'mlp/deploy_decision_survivalrate_average_' + str(loc_num)+ '_set_' + str(build_num_choose_cnt) + '_dist_' + str(dist) + '.npy'
            np.save(output_name,deploy_decision_survivalrate_average_mlp)
                                

            print('model finishing _ ',loc_num) 
            

    '''
  (1 + e^{-0.26 + 0.106 \cdot t_{\text{aed}} + 0.139 \cdot t_{\text{cpr}}})^{-1}, & t_{\min} < 4, \\
    0, & t_{\min} \geq 4.
'''        


