import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help="number of training epochs")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: n")
    parser.add_argument('--frac', type=float, default=1.0,
                        help='the fraction of clients: C')
    parser.add_argument('--local_epoch', type=int, default=5,
                        help="the number of local epochs")
    parser.add_argument('--local_iter', type=int, default=1,
                        help="the number of local iterations")
    parser.add_argument('--local_bs', type=int, default=50,
                        help="local batch size: b")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_warm_up', type=bool, default=False,
                        help='learning rate warm up')
    parser.add_argument('--warm_up_epochs', type=int, default=5,
                        help='learning rate warm up epochs')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum')
    parser.add_argument('--train_rule', type=str, default='FedAvg',
                        help='the training rule for personalized FL')
    parser.add_argument('--agg_g', type=int, default=1,
                        help='weighted average for personalized FL')
    parser.add_argument('--lam', type=float, default=1.0,
                        help='coefficient for reg term')
    parser.add_argument('--local_size', type=int, default=600,
                        help='number of samples for each client')
    parser.add_argument('--margin_loss_penalty', type=float, default=0.03,
                        help='coefficient for margin loss penalty')
    parser.add_argument('--k_proportion', type=float, default=0.8,
                        help='proportions of principal directions')
    parser.add_argument('--fedprox_proximal', type=float, default=0.01,
                        help='coefficient for fedprox proximal term')
    parser.add_argument('--marg_control_loss',type=bool,default=True,
                        help='whether use the margin control loss')
    parser.add_argument('--svd',type=bool,default=True,
                        help='whether use svd')

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar',
                        help="name of dataset")
    parser.add_argument('--retina_split', type=str, default='1',
                        help="retina dataset split setting")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--device', default='cuda:0', help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--noniid_s', type=int, default=80,
                        help='Default set to 0.2 Set to 1.0 for IID.')
    args = parser.parse_args()
    return args
