from methods.fedavg import LocalUpdate_FedAvg
from methods.fedpac import LocalUpdate_FedPAC
from methods.fedprox import LocalUpdate_FedProx
from methods.fedbn import LocalUpdate_FedBN
from methods.fedgh import LocalUpdate_FedGH
from methods.fedld import LocalUpdate_FedLD

def local_update(rule):
    LocalUpdate = {'FedAvg':LocalUpdate_FedAvg,
                   'FedPAC':LocalUpdate_FedPAC,
                   'FedProx':LocalUpdate_FedProx,
                   'FedBN':LocalUpdate_FedBN,
                   'FedGH':LocalUpdate_FedGH,
                   'FedLD':LocalUpdate_FedLD
    }

    return LocalUpdate[rule]