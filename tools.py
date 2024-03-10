import copy
import torch
import types
import numpy as np
import cvxpy as cvx
from options import args_parser
args = args_parser()

def average_weights_weighted(w, avg_weight):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    weight = torch.tensor(avg_weight)
    agg_w = weight/(weight.sum(dim=0))
    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key])
        for i in range(len(w)):
            w_avg[key] += (agg_w[i] * w[i][key]).to(w_avg[key].dtype)
    return w_avg

def get_parameter_values(model):
    parameter = torch.cat([torch.reshape(param.data, (-1,)) for param in model.parameters()]).clone().detach()
    return parameter


