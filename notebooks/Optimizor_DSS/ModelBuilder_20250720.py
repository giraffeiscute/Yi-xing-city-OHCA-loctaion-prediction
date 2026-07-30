"""
AED_deployment_model
Author: Kexin Cao
Institute: Tsinghua University
Date: 20241220 
"""


from gurobipy import *
import time

class ModelBuilder(object):
    def __init__(self):
        self.data = None
        
        self.x_i = {}
        self.cons_dist = {}

        self.deploy_decision = []
        
        self.obj_val = 0
        self.run_time = 0
        
        ##%% for self.IP_MLP_model
        self.x_i_mlp = {}
        self.cons_dist_mlp = {}

        self.deploy_decision_mlp = []
        
        self.obj_val_mlp = 0
        self.run_time_mlp = 0
                
    def build_IP(self, data, candidate_loc_id, loc_num):
        """





        """
        start_time = time.time()
        self.data = data
        self.IP_model = Model('IP_model')
        obj = LinExpr()
        lhs = LinExpr()


        for loc_id in candidate_loc_id:
            i = loc_id
            self.x_i[i] = self.IP_model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x_' + str(i))
            obj.addTerms(data.loc_score[i], self.x_i[i])
            lhs.addTerms(1, self.x_i[i])
            # if i % 200 == 0:
            #     print(i) 
        self.IP_model.setObjective(obj, GRB.MAXIMIZE)


        if data.infinite == 0:
            for i in candidate_loc_id:
                for j in candidate_loc_id:
                    if j > i and data.indicator_i_j[i, j] == 1:
                        self.cons_dist[i, j] = self.IP_model.addConstr(
                            self.x_i[i] + self.x_i[j] <= 1,
                            name='cons_dist' + '_' + str(i) + '_' + str(j)
                        )
            # if i % 200 == 0:
            #     print(i) 
        

        self.cons_sum = self.IP_model.addConstr(lhs <= loc_num, name = 'cons_sum')


        self.IP_model.setParam('OutputFlag', 1)
        self.IP_model.setParam('Presolve', 1)








        finish_time = time.time() - start_time 
        print('model construction time:',finish_time)


        self.IP_model.setParam('TimeLimit', 3600*3)


        self.IP_model.optimize()  
        # self.IP_model.write('IP_model.lp')


        self.obj_val = self.IP_model.ObjVal
        self.run_time = self.IP_model.run_time
        for loc_id in candidate_loc_id:
            if self.x_i[loc_id].x >= 0.5:
                self.deploy_decision.append(loc_id)

    def build_IP_mlp(self, data, candidate_loc_id, loc_num):
        """


        """
        start_time = time.time()
        self.data = data
        self.IP_MLP_model = Model('IP_MLP_model')
        obj = LinExpr()
        lhs = LinExpr()


        for loc_id in candidate_loc_id:
            i = loc_id
            self.x_i_mlp[i] = self.IP_MLP_model.addVar(lb = 0, ub = 1, vtype = GRB.BINARY, name = 'x_' + str(i))
            obj.addTerms(data.loc_score_mlp[i], self.x_i_mlp[i])
            lhs.addTerms(1, self.x_i_mlp[i])
            # if i % 200 == 0:
            #     print(i) 
        self.IP_MLP_model.setObjective(obj, GRB.MAXIMIZE)   


        if data.infinite == 0:
            for i in candidate_loc_id:
                for j in candidate_loc_id:
                    if j > i and data.indicator_i_j[i, j] == 1:
                        self.cons_dist_mlp[i, j] = self.IP_MLP_model.addConstr(
                            self.x_i_mlp[i] + self.x_i_mlp[j] <= 1,
                            name=f'cons_dist_{i}_{j}'
                        )
            # if i % 200 == 0:
            #     print(i) 
        

        self.cons_sum = self.IP_MLP_model.addConstr(lhs <= loc_num, name = 'cons_sum')

        self.IP_MLP_model.setParam('OutputFlag', 1) 
        self.IP_MLP_model.setParam('Presolve', 1)   








        finish_time = time.time() - start_time 
        print('model construction time:',finish_time)

        self.IP_MLP_model.setParam('TimeLimit', 3600*3)
        self.IP_MLP_model.optimize()  
        # self.IP_model.write('IP_model.lp')


        self.obj_val_mlp = self.IP_MLP_model.ObjVal
        self.run_time_mlp = self.IP_MLP_model.run_time
        for loc_id in candidate_loc_id:
            if self.x_i_mlp[loc_id].x >= 0.5:
                self.deploy_decision_mlp.append(loc_id) 
        
                   
                    
                    
                    
                    
                    
                    
                    
