import torch


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

def pcgrad_svd(client_num, client_grads, grad_history):
    """ Projecting conflicting gradients using SVD"""
    client_grads_ = torch.stack(client_grads)
    grads = []
    grad_len = grad_history['grad_len']
    start = 0
    for key in grad_len.keys():
        #print(key)
        g_len = grad_len[key]
        end = start + g_len
        layer_grad_history = grad_history[key]
        if layer_grad_history is not None:
            client_grads_layer = client_grads_[:, start:end]
            ## svd
            naive_avg_grad = torch.mean(client_grads_layer, dim=0, keepdim=True)
            hessian = 1 / client_grads_layer.size(0) * client_grads_layer @ client_grads_layer.T
            u, s, e = torch.svd(hessian)
            v = client_grads_layer.T @ e # (d*m)*(m*m) = (d*m)
            v = v.T #(m*d)
            k = int(len(s) * 0.9)
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
                grad_new = torch.unsqueeze(grad_new,0)
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
    