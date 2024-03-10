import tools
from svd_tools import get_grads_, set_grads_,pcgrad_svd
import numpy as np
import copy

from tools import average_weights_weighted
from tqdm import tqdm


def one_round_training(rule):
    # gradient aggregation rule
    Train_Round = {'FedAvg':train_round_fedavg,
                   'FedLD':train_round_fedld,
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
    local_grads = []
    local_acc1 = []
    local_acc2 = []
    agg_weight = []

    global_weight = global_model.state_dict()
    if rnd <= args.epochs:
        if rnd <= 2:
            args.lr = 0.0001
        else:
            args.r = 0.01

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
    #global_weight = average_weights_weighted(local_weights, agg_weight)
    local_clients_grads = []
    local_weights_new = []
    for idx in idx_users:
        local_clients_grads.append(get_grads_(local_clients[idx].local_model, global_model))
    print('shape of local_clients_grads in running.py',len(local_clients_grads),' ',local_clients_grads[0].shape)
    #grad_new, grad_history = pcgrad_hierarchy(num_users, local_clients_grads, grad_history)
    grad_new, grad_history = pcgrad_svd(num_users, local_clients_grads, grad_history)
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

# FedAvg
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
    if rnd <= args.epochs:
        if rnd <= 2:
            args.lr = 0.0001
        else:
            args.r = 0.01

    supervised_loss_list = []
    dist_shift_loss_list = []
    all_data_loss_list = []
    for idx in tqdm(idx_users):
        local_client = local_clients[idx]
        agg_weight.append(local_client.agg_weight)
        local_epoch = args.local_epoch
        local_client.update_local_model(global_weight=global_weight)
        w, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch, round=rnd)
        print('idx: {}, loss1: {}, loss2: {}, acc1: {}, acc2: {}'.format(idx, loss1, loss2, acc1, acc2))
        # supervised loss for client idx
        supervised_loss_idx = local_client.local_test(test_loader = train_loader[idx])[0]
        supervised_loss_list.append(supervised_loss_idx)
        # distribution shift loss for client idx
        dist_shift_loss_idx = 0
        all_data_loss_idx = 0
        for j in tqdm(idx_users):
            dist_shift_loss_idx  = dist_shift_loss_idx + local_client.local_test(test_loader = train_loader[j])[0] - local_client.local_test(test_loader = train_loader[idx])[0]
            all_data_loss_idx = all_data_loss_idx + local_client.local_test(test_loader = train_loader[j])[0]
        dist_shift_loss_list.append(dist_shift_loss_idx)
        all_data_loss_list.append(all_data_loss_idx)
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(copy.deepcopy(loss1))
        local_losses2.append(copy.deepcopy(loss2))
        local_acc1.append(acc1)
        local_acc2.append(acc2)

    # get global weights
    global_weight = average_weights_weighted(local_weights, agg_weight)
    # update global model
    global_model.load_state_dict(global_weight)

    temp_client = copy.deepcopy(local_clients[0])
    temp_client.update_local_model(global_weight=global_weight)
    all_data_loss_global = 0
    for idx in idx_users:
        all_data_loss_global += temp_client.local_test(test_loader = train_loader[idx])[0]
    aggregation_loss_list = []
    for idx in idx_users:
        aggregation_loss_list.append(all_data_loss_global - all_data_loss_list[idx])

    supervised_loss = sum(supervised_loss_list)
    dist_shift_loss = sum(dist_shift_loss_list)/len(dist_shift_loss_list)
    aggregation_loss = sum(aggregation_loss_list)/len(aggregation_loss_list)


    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2, supervised_loss, dist_shift_loss, aggregation_loss