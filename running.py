from svd_tools import get_grads_, set_grads_,pcgrad_svd, pcgrad_hierarchy
import numpy as np
import copy
import torch
import tools
from tools import average_weights_weighted
from tqdm import tqdm
from options import args_parser


def one_round_training(rule):
    # gradient aggregation rule
    Train_Round = {'FedAvg':train_round_fedavg,
                   'FedPAC':train_round_fedpac,
                   'FedProx':train_round_fedprox,
                   'FedBN':train_round_fedbn,
                   'FedGH': train_round_fedgh,
                   'FedLD': train_round_fedld
    }
    return Train_Round[rule]

## training methods -------------------------------------------------------------------
# local training only
def train_round_standalone(args, global_model, local_clients, rnd, **kwargs):
    print(f'\n---- Global Communication Round : {rnd+1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_losses1, local_losses2 = [], []
    local_acc1 = []
    local_acc2 = []

    global_weight = global_model.state_dict()
    
    for idx in idx_users:
        local_client = local_clients[idx]
        local_epoch = args.local_epoch
        w, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch)
        local_losses1.append(copy.deepcopy(loss1))
        local_losses2.append(copy.deepcopy(loss2))
        local_acc1.append(acc1)
        local_acc2.append(acc2)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2


def train_round_fedld(args, global_model, local_clients, rnd, grad_history, **kwargs):
    print(f'\n---- Global Communication Round : {rnd + 1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_acc1 = []
    local_acc2 = []
    agg_weight = []

    global_weight = global_model.state_dict()
    # for idx in tqdm(idx_users):
    for idx in idx_users:
        local_client = local_clients[idx]
        agg_weight.append(local_client.agg_weight)
        local_epoch = args.local_epoch
        local_client.update_local_model(global_weight=global_weight)
        _, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch, round=rnd)
        local_losses1.append(loss1)
        local_losses2.append(loss2)
        local_acc1.append(acc1)
        local_acc2.append(acc2)

    # get global weights
    if args.svd:
        local_clients_grads = []
        local_weights_new = []
        for idx in idx_users:
            local_clients_grads.append(get_grads_(local_clients[idx].local_model, global_model))
        grad_new, grad_history = pcgrad_svd(num_users, local_clients_grads, grad_history)
        for idx in idx_users:
            local_clients[idx].local_model = set_grads_(local_clients[idx].local_model, global_model, grad_new)
        for idx in idx_users:
            local_weights_new.append(copy.deepcopy(local_clients[idx].local_model.state_dict()))
        global_weight = average_weights_weighted(local_weights_new, agg_weight)
    else:
        global_weight = average_weights_weighted(local_weights, agg_weight)
    global_model.load_state_dict(global_weight)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)
    
    torch.cuda.empty_cache()

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2

def train_round_fedavg(args, global_model, local_clients, rnd, train_loader, **kwargs):
    print(f'\n---- Global Communication Round : {rnd+1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_grads = []
    local_acc1 = []
    local_acc2 = []
    agg_weight = []

    global_weight = global_model.state_dict()
    for idx in idx_users:
        local_client = local_clients[idx]
        agg_weight.append(local_client.agg_weight)
        local_epoch = args.local_epoch
        local_client.update_local_model(global_weight=global_weight)
        w, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch, round=rnd)
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(copy.deepcopy(loss1))
        local_losses2.append(copy.deepcopy(loss2))
        local_acc1.append(acc1)
        local_acc2.append(acc2)

    # get global weights
    global_weight = average_weights_weighted(local_weights, agg_weight)
    # update global model
    global_model.load_state_dict(global_weight)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2

def train_round_fedprox(args, global_model, local_clients, rnd, train_loader, **kwargs):
    print(f'\n---- Global Communication Round : {rnd + 1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_acc1 = []
    local_acc2 = []
    agg_weight = []

    global_weight = global_model.state_dict()

    for idx in tqdm(idx_users):
        local_client = local_clients[idx]
        agg_weight.append(local_client.agg_weight)
        local_epoch = args.local_epoch
        local_client.update_local_model(global_weight=global_weight)
        w, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch, global_model = global_model, round=rnd)
        print('idx: {}, loss1: {}, loss2: {}, acc1: {}, acc2: {}'.format(idx, loss1, loss2, acc1, acc2))
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(copy.deepcopy(loss1))
        local_losses2.append(copy.deepcopy(loss2))
        local_acc1.append(acc1)
        local_acc2.append(acc2)

    # get global weights
    global_weight = average_weights_weighted(local_weights, agg_weight)
    # update global model
    global_model.load_state_dict(global_weight)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2

def communication_fedbn(args,server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        #if args.mode.lower() == 'fedbn':
        client_num = args.num_users
        for key in server_model.state_dict().keys():
            if 'bn' not in key:
                temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                for client_idx in range(client_num):
                    temp += client_weights[client_idx] * models[client_idx].local_model.state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)
                for client_idx in range(client_num):
                    models[client_idx].local_model.state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models

def train_round_fedbn(args, global_model, local_clients, rnd, **kwargs):
    print(f'\n---- Global Communication Round : {rnd + 1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_grads = []
    local_acc1 = []
    local_acc2 = []
    agg_weight = []

    global_weight = global_model.state_dict()

    for idx in tqdm(idx_users):
        local_client = local_clients[idx]
        agg_weight.append(local_client.agg_weight)
        local_epoch = args.local_epoch
        local_client.update_except_bn_local_model(global_weight=global_weight) # update parameters except bn layers
        w, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch, round=rnd)
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(copy.deepcopy(loss1))
        local_losses2.append(copy.deepcopy(loss2))
        local_acc1.append(acc1)
        local_acc2.append(acc2)

    # get global weights
    global_weight = average_weights_weighted(local_weights, agg_weight)
    # update global model
    global_model.load_state_dict(global_weight)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return global_model, local_clients, loss_avg1, loss_avg2, acc_avg1, acc_avg2

def train_round_fedpac(args, global_model, local_clients, rnd, **kwargs):
    print(f'\n---- Global Communication Round : {rnd + 1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)

    local_weights, local_losses1, local_losses2 = [], [], []
    local_acc1 = []
    local_acc2 = []
    agg_weight = []  # aggregation weights for f
    avg_weight = []  # aggregation weights for g
    sizes_label = []
    local_protos = []

    Vars = []
    Hs = []

    agg_g = args.agg_g  # conduct classifier aggregation or not

    if rnd <= args.epochs:
        for idx in idx_users:
            local_client = local_clients[idx]
            ## statistics collection
            v, h = local_client.statistics_extraction()
            Vars.append(copy.deepcopy(v))
            Hs.append(copy.deepcopy(h))
            ## local training
            local_epoch = args.local_epoch
            sizes_label.append(local_client.sizes_label)
            w, loss1, loss2, acc1, acc2, protos = local_client.local_training(local_epoch=local_epoch, round=rnd)
            local_weights.append(copy.deepcopy(w))
            local_losses1.append(copy.deepcopy(loss1))
            local_losses2.append(copy.deepcopy(loss2))
            local_acc1.append(acc1)
            local_acc2.append(acc2)
            agg_weight.append(local_client.agg_weight)
            local_protos.append(copy.deepcopy(protos))

        # get weight for feature extractor aggregation
        agg_weight = torch.stack(agg_weight).to(args.device)

        # update global feature extractor
        global_weight_new = average_weights_weighted(local_weights, agg_weight)

        # update global prototype
        global_protos = tools.protos_aggregation(local_protos, sizes_label)

        for idx in range(num_users):
            local_client = local_clients[idx]
            local_client.update_base_model(global_weight=global_weight_new)
            local_client.update_global_protos(global_protos=global_protos)

        # get weight for local classifier aggregation
        if agg_g and rnd < args.epochs:
            avg_weights = tools.get_head_agg_weight(m, Vars, Hs)
            idxx = 0
            for idx in idx_users:
                local_client = local_clients[idx]
                if avg_weights[idxx] is not None:
                    new_cls = tools.agg_classifier_weighted_p(local_weights, avg_weights[idxx],
                                                                  local_client.w_local_keys, idxx)
                else:
                    new_cls = local_weights[idxx]
                local_client.update_local_classifier(new_weight=new_cls)
                idxx += 1

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2

def train_round_fedgh(args, global_model, local_clients, rnd, grad_history, **kwargs):
    print(f'\n---- Global Communication Round : {rnd + 1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_acc1 = []
    local_acc2 = []
    agg_weight = []

    global_weight = global_model.state_dict()

    for idx in tqdm(idx_users):
        local_client = local_clients[idx]
        agg_weight.append(local_client.agg_weight)
        local_epoch = args.local_epoch
        local_client.update_local_model(global_weight=global_weight)
        w, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch, round=rnd)
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(copy.deepcopy(loss1))
        local_losses2.append(copy.deepcopy(loss2))
        local_acc1.append(acc1)
        local_acc2.append(acc2)

    # get global weights
    local_clients_grads = []
    local_weights_new = []
    for idx in idx_users:
        local_clients_grads.append(get_grads_(local_clients[idx].local_model, global_model))
    grad_new, grad_history = pcgrad_hierarchy(num_users, local_clients_grads, grad_history)
    for idx in idx_users:
        local_clients[idx].local_model = set_grads_(local_clients[idx].local_model, global_model, grad_new)
    for idx in idx_users:
        local_weights_new.append(copy.deepcopy(local_clients[idx].local_model.state_dict()))
    global_weight = average_weights_weighted(local_weights_new, agg_weight)
    # update global model
    global_model.load_state_dict(global_weight)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2