import torch
import torch.nn.functional as F
from options import args_parser
args = args_parser()


def get_grads_(model, server_model):
    grads = []
    for key in server_model.state_dict().keys():
        grads.append(model.state_dict()[key].data.clone().detach().flatten() - server_model.state_dict()[key].data.clone().detach().flatten())
    return torch.cat(grads)

def set_grads_(model,server_model, new_grads):
    start = 0
    for key in server_model.state_dict().keys():
        dims = model.state_dict()[key].shape
        end = start + dims.numel()
        model.state_dict()[key].data.copy_(server_model.state_dict()[key].data.clone().detach() + new_grads[start:end].reshape(dims).clone())
        start = end
    return model

def initialize_grad_len(server_model, grad_history):
    grad_len = {key:0 for key in server_model.state_dict().keys()}
    for key in server_model.state_dict().keys():
        dims = server_model.state_dict()[key].shape
        grad_len[key] = dims.numel()
    return grad_len

def pcgrad_hierarchy(client_num, client_grads, grad_history):
    """ Projecting conflicting gradients"""
    #print('grad_history', grad_history)
    eplison = 1e-7
    client_grads_ = torch.stack(client_grads)
    grads = []
    grad_len = grad_history['grad_len']
    start = 0
    for key in grad_len.keys():
        g_len = grad_len[key]
        end = start + g_len
        layer_grad_history = grad_history[key]
        if layer_grad_history is not None:
            pc_v = layer_grad_history.unsqueeze(0)
            client_grads_layer = client_grads_[:, start:end]
            while True:
                num = client_grads_layer.size(0)
                if num > 2:
                    inner_prod = torch.mul(client_grads_layer, pc_v).sum(1)
                    project = inner_prod / (pc_v ** 2).sum().sqrt()
                    _, ind = project.sort(descending=True)
                    pair_list = []
                    if num % 2 == 0:
                        for i in range(num // 2):
                            pair_list.append([ind[i], ind[num - i - 1]])
                    else:
                        for i in range(num // 2):
                            pair_list.append([ind[i], ind[num - i - 1]])
                        pair_list.append([ind[num // 2]])
                    client_grads_new = []
                    for pair in pair_list:
                        if len(pair) > 1:
                            grad_0 = client_grads_layer[pair[0]]
                            grad_1 = client_grads_layer[pair[1]]
                            inner_prod = torch.dot(grad_0, grad_1)
                            if inner_prod < 0:
                                # Sustract the conflicting component
                                grad_pc_0 = grad_0 - inner_prod / ((grad_1 ** 2).sum() + eplison) * grad_1
                                grad_pc_1 = grad_1 - inner_prod / ((grad_0 ** 2).sum() + eplison) * grad_0
                            else:
                                grad_pc_0 = grad_0
                                grad_pc_1 = grad_1

                            grad_pc_0_1 = grad_pc_0 + grad_pc_1
                            client_grads_new.append(grad_pc_0_1)
                        else:
                            grad_single = client_grads_layer[pair[0]]
                            client_grads_new.append(grad_single)
                    client_grads_layer = torch.stack(client_grads_new)
                elif num == 2:
                    grad_pc_0 = client_grads_layer[0]
                    grad_pc_1 = client_grads_layer[1]
                    inner_prod = torch.dot(grad_pc_0, grad_pc_1)
                    if inner_prod < 0:
                        # Sustract the conflicting component
                        grad_pc_0 = grad_pc_0 - inner_prod / ((grad_pc_1 ** 2).sum() + eplison) * grad_pc_1
                        grad_pc_1 = grad_pc_1 - inner_prod / ((grad_pc_0 ** 2).sum() + eplison) * grad_pc_0

                    grad_pc_0_1 = grad_pc_0 + grad_pc_1
                    grad_new = grad_pc_0_1 / client_num
                    break
                else:
                    assert False
            gamma = 0.99
            grad_history[key] = gamma * grad_history[key] + (1 - gamma) * grad_new
            grads.append(grad_new)
        else:
            grad_new = client_grads_[:, start:end].mean(0)
            grad_history[key] = grad_new
            grads.append(grad_new)
        start = end
    grad_new = torch.cat(grads)

    return grad_new, grad_history

def pcgrad_svd(client_num, client_grads, grad_history):
    """ Projecting conflicting gradients using SVD"""
    client_grads_ = torch.stack(client_grads)
    grads = []
    grad_len = grad_history['grad_len']
    start = 0
    for key in grad_len.keys():
        g_len = grad_len[key]
        end = start + g_len
        layer_grad_history = grad_history[key]
        client_grads_layer = client_grads_[:, start:end]
        naive_avg_grad = torch.mean(client_grads_layer, dim=0, keepdim=True)
        if layer_grad_history is not None:
            if not torch.all(layer_grad_history == 0):
                hessian = 1 / client_grads_layer.size(0) * client_grads_layer @ client_grads_layer.T
                u, s, e = torch.svd(hessian)
                v = client_grads_layer.T @ e # (d*m)*(m*m) = (d*m)
                v = v.T #(m*d)
                k = int(len(s) * args.k_proportion)
                w = v[:k]
                for j in range(k):
                    num_pos = 0
                    num_neg = 0
                    for i in range(client_grads_layer.size(0)):
                        if torch.dot(client_grads_layer[i], w[j]) >= 0:
                            num_pos += 1
                        else:
                            num_neg += 1
                    if num_pos < num_neg:
                        w[j] = -w[j]
                grad_agg = []
                for i in range(client_num):
                    grad_pc = client_grads_layer[i]
                    grad_revise = torch.zeros_like(grad_pc)
                    for j in range(k):
                        grad_revise_j= torch.dot(grad_pc, w[j])/torch.dot(w[j], w[j]) * w[j]
                        grad_revise_j = grad_revise_j * s[j]/s.sum() * torch.norm(grad_pc) / torch.norm(grad_revise_j)
                        grad_revise += grad_revise_j
                    grad_agg.append(grad_revise)
                grad_new = torch.mean(torch.stack(grad_agg), dim=0, keepdim=True)
                grad_new = torch.squeeze(grad_new)
                if g_len == 1:
                    grad_new = torch.squeeze(naive_avg_grad)
                    grad_new = torch.unsqueeze(grad_new, 0)
            else:
                client_grads_layer = client_grads_[:, start:end]
                naive_avg_grad = torch.mean(client_grads_layer, dim=0, keepdim=True)
                grad_new = naive_avg_grad
                grad_new = torch.squeeze(grad_new)
                if g_len == 1:
                    grad_new = torch.squeeze(naive_avg_grad)
                    grad_new = torch.unsqueeze(grad_new, 0)
            gamma = 0.99
            grad_history[key] = gamma * grad_history[key] + (1 - gamma) * grad_new
            grads.append(grad_new)
        else:
            grad_new = client_grads_[:, start:end].mean(0)
            grad_history[key] = grad_new
            grads.append(grad_new)
        start = end
    grad_new = torch.cat(grads)

    return grad_new, grad_history
    