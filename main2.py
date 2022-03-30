import time
import argparse
import torch
from torch import nn
import models
import dataset
from utils import train

parser = argparse.ArgumentParser(description='EOT Experiment')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=80, metavar='N',
                    help='number of epochs to train (default: 80)')
parser.add_argument('--dataset', type=str, default='MNIST',
                    help='dataset (default: MNIST)')
parser.add_argument('--data', type=str, default='./',
                    help='data root (default: ./)')
parser.add_argument('--save', type=str, default='exp1',
                    help='save file name (default: exp1)')
parser.add_argument('--actif', type=str, default='lrelu', metavar='Activation',
                    help='activation function (default: LeakyRelu)')
parser.add_argument('--latent_dim', type=int, default=2, metavar='N',
                    help='dimension of z space (default: 2)')
parser.add_argument('--sample_size', type=int, default=1024, metavar='N',
                    help='sample size for prior distribution (default: 1024)')
parser.add_argument('--chunk_size', type=int, default=10, metavar='N',
                    help='chunk size for batch (default: 10)')
parser.add_argument('--epsilon', type=float, default=1.0, metavar='N',
                    help='weight of regularization (default: 1.0)')
parser.add_argument('--learning_rate', type=float, default=1e-2,
                    help='init learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)

activations_list = {
    'softplus': nn.Softplus(),
    'lrelu': nn.LeakyReLU(),
    'relu': nn.ReLU()
}
activFun = activations_list[args.actif]

if __name__ == '__main__':

    train_loader, test_loader = dataset.get_loaders(args)
    if args.dataset == 'MNIST':
        img_size = 28
    else:
        img_size = 64
    # initialize model
    in_channel = 1
    model = models.Dirichletp(args.sample_size, args.chunk_size, args.epsilon, in_channel,
                                    args.latent_dim, activFun, img_size, device)
    model = model.to(device)

    start_time = time.time()
    train(model, train_loader, args, device, log_interval=10)
    print("after training")
    print('training time elapsed {}s'.format(time.time() - start_time))
