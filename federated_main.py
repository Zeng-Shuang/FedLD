import numpy as np
import torch
import torch.nn as nn
from data_loader import get_dataset
from running import one_round_training
from methods import local_update
from models.cnn import ResNet50
import torchvision.models as torch_models
from options import args_parser
import copy
from svd_tools import initialize_grad_len

torch.set_num_threads(4)

if __name__ == '__main__':
    args = args_parser()
    device = args.device
    print(device)
    # load dataset and user groups
    train_loader, test_loader, global_test_loader = get_dataset(args)
    seed = 520
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # construct model
    if args.dataset == 'covid_fl':
        args.num_classes = 3
        args.num_users = 12
        global_model = ResNet50()
        pretrained_resnet50 = torch_models.resnet50(pretrained=True)
        global_model.load_state_dict(pretrained_resnet50.state_dict())
        global_model.fc = nn.Linear(2048, args.num_classes)
        global_model.to(device)
    elif args.dataset == 'retina':
        args.num_classes = 2
        global_model = ResNet50()
        pretrained_resnet50 = torch_models.resnet50(pretrained=True)
        global_model.load_state_dict(pretrained_resnet50.state_dict())
        global_model.fc = nn.Linear(2048, args.num_classes)
        global_model.to(device)
    else:
        raise NotImplementedError()
    

    # Training Rule
    LocalUpdate = local_update(args.train_rule)
    # One Round Training Function
    train_round_parallel = one_round_training(args.train_rule)

    # Training
    train_loss, train_acc = [], []
    test_acc = []
    local_accs1, local_accs2 = [], []
#======================================================================================================#
    local_clients = []
    for idx in range(args.num_users):
        if args.dataset == 'covid_fl' or args.dataset == 'retina':
            local_clients.append(LocalUpdate(idx=idx, args=args, train_set=train_loader[idx], test_set=test_loader,
                                             model=copy.deepcopy(global_model)))
        else:
            local_clients.append(LocalUpdate(idx=idx, args=args, train_set=train_loader[idx], test_set=test_loader[idx],
                                            model=copy.deepcopy(global_model)))

    if args.train_rule == 'FedLD':
        # initialize grad_history and grad_len
        grad_history = {}
        for k in global_model.state_dict().keys():
            grad_history[k] = None
        grad_len = initialize_grad_len(global_model, grad_history)

        grad_history['grad_len'] = grad_len

        for round in range(args.epochs):
            loss1, loss2, local_acc1, local_acc2 = train_round_parallel(args, global_model, local_clients, round,grad_history)
            train_loss.append(loss1)
            print("Train Loss: {}, {}".format(loss1, loss2))
            print("Local Accuracy on Local Data: {}%, {}%".format(local_acc1, local_acc2))
            local_accs1.append(local_acc1)
            local_accs2.append(local_acc2)

    else:
        for round in range(args.epochs):
            loss1, loss2, local_acc1, local_acc2= train_round_parallel(args, global_model, local_clients, round)
            train_loss.append(loss1)
            print("Train Loss: {}, {}".format(loss1, loss2))
            print("Local Accuracy on Local Data: {}%, {}%".format(local_acc1, local_acc2))
            local_accs1.append(local_acc1)
            local_accs2.append(local_acc2)


