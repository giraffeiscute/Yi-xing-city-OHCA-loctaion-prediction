"""Yixing optimizer model builder.

This mirrors the DSS ``ModelBuilder`` role: receive a populated ``Data`` object,
build the binary IP, solve it, and expose deployment decisions/objective values.
"""

from __future__ import annotations

import time


class ModelBuilder(object):
    def __init__(self):
        self.data = None

        self.x_i = {}
        self.cons_dist = {}
        self.deploy_decision = []
        self.obj_val = 0.0
        self.run_time = 0.0

        self.x_i_mlp = {}
        self.cons_dist_mlp = {}
        self.deploy_decision_mlp = []
        self.obj_val_mlp = 0.0
        self.run_time_mlp = 0.0

    def build_IP(self, data, candidate_loc_id=None, loc_num=None, time_limit_seconds=600, output_flag=0):
        return self._build_ip(
            data=data,
            candidate_loc_id=candidate_loc_id,
            loc_num=loc_num,
            score_values=data.loc_score,
            model_name="yixing_candidate_ip",
            x_attr="x_i",
            cons_attr="cons_dist",
            decision_attr="deploy_decision",
            obj_attr="obj_val",
            runtime_attr="run_time",
            time_limit_seconds=time_limit_seconds,
            output_flag=output_flag,
        )

    def build_IP_mlp(self, data, candidate_loc_id=None, loc_num=None, time_limit_seconds=600, output_flag=0):
        if data.loc_score_mlp is None:
            raise KeyError("Missing MLP score column: total_score_mlp")
        return self._build_ip(
            data=data,
            candidate_loc_id=candidate_loc_id,
            loc_num=loc_num,
            score_values=data.loc_score_mlp,
            model_name="yixing_candidate_mlp_ip",
            x_attr="x_i_mlp",
            cons_attr="cons_dist_mlp",
            decision_attr="deploy_decision_mlp",
            obj_attr="obj_val_mlp",
            runtime_attr="run_time_mlp",
            time_limit_seconds=time_limit_seconds,
            output_flag=output_flag,
        )

    def _build_ip(
        self,
        data,
        candidate_loc_id,
        loc_num,
        score_values,
        model_name,
        x_attr,
        cons_attr,
        decision_attr,
        obj_attr,
        runtime_attr,
        time_limit_seconds,
        output_flag,
    ):
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as exc:
            raise RuntimeError("gurobipy is required for optimization but is not installed.") from exc

        start_time = time.time()
        self.data = data
        loc_ids = self._candidate_ids(candidate_loc_id, data.build_num)
        loc_num = data.loc_num if loc_num is None else int(loc_num)
        self._validate_candidate_ids(loc_ids, len(score_values))

        model = gp.Model(model_name)
        x_vars = {}
        cons_dist = {}
        obj = gp.LinExpr()
        lhs = gp.LinExpr()

        for loc_id in loc_ids:
            x_vars[loc_id] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name="x_" + str(loc_id))
            obj.addTerms(float(score_values[loc_id]), x_vars[loc_id])
            lhs.addTerms(1.0, x_vars[loc_id])

        model.setObjective(obj, GRB.MAXIMIZE)

        loc_id_set = set(loc_ids)
        for i, j in data.conflict_pairs:
            if i in loc_id_set and j in loc_id_set:
                cons_dist[i, j] = model.addConstr(
                    x_vars[i] + x_vars[j] <= 1,
                    name="cons_dist_" + str(i) + "_" + str(j),
                )

        self.cons_sum = model.addConstr(lhs <= loc_num, name="cons_sum")
        model.setParam("OutputFlag", output_flag)
        model.setParam("Presolve", 1)
        model.setParam("TimeLimit", time_limit_seconds)
        model.optimize()

        if model.SolCount == 0:
            raise RuntimeError("Gurobi did not produce a feasible solution.")

        decisions = [loc_id for loc_id in loc_ids if x_vars[loc_id].X >= 0.5]
        setattr(self, model_name, model)
        setattr(self, x_attr, x_vars)
        setattr(self, cons_attr, cons_dist)
        setattr(self, decision_attr, decisions)
        setattr(self, obj_attr, float(model.ObjVal))
        setattr(self, runtime_attr, float(getattr(model, "Runtime", time.time() - start_time)))
        return decisions

    @staticmethod
    def _candidate_ids(candidate_loc_id, build_num):
        if candidate_loc_id is None:
            return list(range(int(build_num)))
        return [int(loc_id) for loc_id in candidate_loc_id]

    @staticmethod
    def _validate_candidate_ids(candidate_loc_id, score_count):
        invalid = [loc_id for loc_id in candidate_loc_id if loc_id < 0 or loc_id >= score_count]
        if invalid:
            raise IndexError("candidate_loc_id contains out-of-range ids: " + str(invalid[:10]))