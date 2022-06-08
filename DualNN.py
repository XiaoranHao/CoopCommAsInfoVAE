# import time
# import argparse
# import torch
# from torch import nn
# import models
# import dataset
# from utils import train_VAE, train_DualOT


# parser = argparse.ArgumentParser(description='EOT Experiment')
# parser.add_argument('--batch_size', type=int, default=256, metavar='N',
#                     help='input batch size for training (default: 256)')
# parser.add_argument('--epochs', type=int, default=1000, metavar='N',
#                     help='number of epochs to train (default: 80)')
# parser.add_argument('--dataset', type=str, default='MNIST',
#                     help='dataset (default: MNIST)')
# parser.add_argument('--data', type=str, default='./',
#                     help='data root (default: ./)')
# parser.add_argument('--save', type=str, default='DualNN.pth',
#                     help='save file name (default: exp1)')
# parser.add_argument('--actif', type=str, default='lrelu', metavar='Activation',
#                     help='activation function (default: LeakyRelu)')
# parser.add_argument('--latent_dim', type=int, default=2, metavar='N',
#                     help='dimension of z space (default: 2)')
# parser.add_argument('--sample_size', type=int, default=10000, metavar='N',
#                     help='sample size for prior distribution (default: 1024)')
# parser.add_argument('--chunk_size', type=int, default=10, metavar='N',
#                     help='chunk size for batch (default: 10)')
# parser.add_argument('--epsilon', type=float, default=1.0, metavar='N',
#                     help='weight of regularization (default: 1.0)')
# parser.add_argument('--learning_rate', type=float, default=1e-4,
#                     help='init learning rate')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='disables CUDA training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')

# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()
# device = torch.device("cuda" if args.cuda else "cpu")

# torch.cuda.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)
# torch.manual_seed(args.seed)

# activations_list = {
#     'softplus': nn.Softplus(),
#     'lrelu': nn.LeakyReLU(),
#     'relu': nn.ReLU()
# }
# activFun = activations_list[args.actif]

# if __name__ == '__main__':

#     train_loader, ts, num_train, num_test = dataset.get_loaders(args)
#     if args.dataset == 'MNIST':
#         img_size = 28
#     else:
#         img_size = 64
#     # initialize model
#     in_channels = 1
#     model = models.CoopCommDualOT(num_train, args.sample_size, args.chunk_size,
#                                   args.epsilon, in_channels, args.latent_dim, activFun, img_size, device)

#     model = model.to(device)
#     start_time = time.time()
#     model.c_function.load_state_dict(torch.load('./savedmodels/vae_decoder1.pth'))
#     train_DualOT(model, train_loader, args, device, log_interval=1)

#     print("after training")
#     print('training time elapsed {}s'.format(time.time() - start_time))


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

import models
import torch
import dataset
import solver
from torch import nn
from utils import train_DualOT2

import time
import gc



torch.cuda.manual_seed(2)
torch.cuda.manual_seed_all(2)
torch.manual_seed(2)

actif='lrelu'
device = 'cuda'
activations_list = {
    'softplus': nn.Softplus(),
    'lrelu': nn.LeakyReLU(),
    'relu': nn.ReLU()
}
activFun = activations_list[actif]


class config(object):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.data = './'
        self.epochs = 200
        self.learning_rate = 1e-4
        self.save = 'first_try.pth'
        self.seed = 2
        self.n_samples = 50000
        self.n_iter = 5000
        self.z_bs = 2000
        
args = config('MNIST',512)
train_loader, ts, num_train, num_test, train_loader_binary, ts_binary = dataset.test_loader(args)


img_size = 28
epsilon = 1
in_channels = 1
latent_dim = 2
n_chunk = 80

model = models.CoopCommDualOT_v1(num_train, n_chunk, 
                               epsilon, in_channels, latent_dim, activFun, img_size)
model.c_function.load_state_dict(torch.load('./savedmodels/vae_decoder1.pth'))
for batch_idx, (data_binary, target, idx) in enumerate(ts_binary):
    pass


model = model.to(device)
start_time = time.time()

train_DualOT2(model, train_loader_binary, data_binary, args, device, log_interval=1)

print("after training")
print('training time elapsed {}s'.format(time.time() - start_time))



