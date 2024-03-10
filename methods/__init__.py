from methods.fedavg import LocalUpdate_FedAvg
from methods.fedld import LocalUpdate_FedLD

def local_update(rule):
    LocalUpdate = {'FedAvg':LocalUpdate_FedAvg,
                   'FedLD':LocalUpdate_FedLD
    }

    return LocalUpdate[rule]