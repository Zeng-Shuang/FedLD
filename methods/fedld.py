import torch
from torch.optim.lr_scheduler import LambdaLR
import copy
from torch import nn
from tqdm import tqdm
from options import args_parser
args = args_parser()

class MarginalLogLoss(nn.Module):
    def __init__(self, lambda_value):
        super(MarginalLogLoss, self).__init__()
        self.lambda_value = lambda_value

    def forward(self, output, target):
        log_loss = nn.CrossEntropyLoss()(output, target)
        softmax_output = torch.softmax(output, dim=1)
        magnitude = torch.norm(softmax_output, p=2)
        marginal_log_loss = log_loss + self.lambda_value * torch.log(1 + magnitude ** 2)
        return marginal_log_loss
    # ---------------------------------------------------------------------------- #

class LocalUpdate_FedLD(object):
    def __init__(self, idx, args, train_set, test_set, model):
        self.idx = idx
        self.args = args
        self.train_data = train_set
        self.test_data = test_set
        self.device = args.device
        self.criterion = MarginalLogLoss(args.margin_loss_penalty)
        self.local_model = model
        self.local_model_finetune = copy.deepcopy(model)
        #self.w_local_keys = self.local_model.classifier_weight_keys
        self.agg_weight = self.aggregate_weight()

    def aggregate_weight(self):
        data_size = len(self.train_data.dataset)
        w = torch.tensor(data_size).to(self.device)
        return w

    def local_test(self, test_loader, test_model=None):
        model = self.local_model if test_model is None else test_model
        model.eval()
        device = self.device
        correct = 0
        total = len(test_loader.dataset)
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        acc = 100.0 * correct / total
        return acc

    def update_local_model(self, global_weight):
        self.local_model.load_state_dict(global_weight)

    def local_training(self, local_epoch, round=0):
        model = self.local_model
        model.train()
        iter_loss = []
        model.zero_grad()


        acc1 = self.local_test(self.test_data)

        # Set optimizer for the local updates, default sgd
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                    momentum=0.5, weight_decay=0.0005)
        warmup_epochs = args.warm_up_epochs
        scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (local_epoch * round + ep) / warmup_epochs if (local_epoch * round + ep) < warmup_epochs else 1)

        # multiple local epochs
        if local_epoch > 0:
            for ep in range(local_epoch):
                data_loader = iter(self.train_data)
                iter_num = len(data_loader)
                for it in range(iter_num):
                    images, labels = next(data_loader)
                    images, labels = images.to(self.device), labels.to(self.device)
                    model.zero_grad()
                    _, output = model(images)
                    loss = self.criterion(output, labels)
                    loss.backward()
                    optimizer.step()
                    if args.lr_warm_up:
                        scheduler.step()
                    iter_loss.append(loss.item())
                    torch.cuda.empty_cache()
        # multiple local iterations, but less than 1 epoch
        else:
            data_loader = iter(self.train_data)
            iter_num = self.args.local_iter
            for it in range(iter_num):
                images, labels = next(data_loader)
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                _, output = model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()
                iter_loss.append(loss.item())
        # loss value
        round_loss1 = iter_loss[0]
        round_loss2 = iter_loss[-1]
        acc2 = self.local_test(self.test_data)

        return model.state_dict(), round_loss1, round_loss2, acc1, acc2

    def local_fine_tuning(self, local_epoch, round=0):
        model = self.local_model
        model.train()
        iter_loss = []
        model.zero_grad()

        acc1 = self.local_test(self.test_data)

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr,
                                    momentum=0.5, weight_decay=0.0005)
        # multiple local epochs
        if local_epoch > 0:
            for ep in range(local_epoch):
                data_loader = iter(self.train_data)
                iter_num = len(data_loader)
                for it in range(iter_num):
                    images, labels = next(data_loader)
                    images, labels = images.to(self.device), labels.to(self.device)
                    model.zero_grad()
                    _, output = model(images)
                    loss = self.criterion(output, labels)
                    loss.backward()
                    optimizer.step()
                    iter_loss.append(loss.item())
        # multiple local iterations, but less than 1 epoch
        else:
            data_loader = iter(self.train_data)
            iter_num = self.args.local_iter
            for it in range(iter_num):
                images, labels = next(data_loader)
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                _, output = model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()
                iter_loss.append(loss.item())
        # loss value
        round_loss1 = iter_loss[0]
        round_loss2 = iter_loss[-1]
        acc2 = self.local_test(self.test_data)

        return round_loss1, round_loss2, acc1, acc2
