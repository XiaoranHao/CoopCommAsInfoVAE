import models
import torch
import dataset
import solver
from torch import nn
from utils import train_VAE, train_DualOT, train_SemiDualOT
import time
import gc

class config(object):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.data = './'
        self.epochs = 100
        self.learning_rate = 1e-4
        self.save = 'Semidual.pth'
        self.seed = 1
args = config('MNIST',512)

torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)

actif='lrelu'
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
activations_list = {
    'softplus': nn.Softplus(),
    'lrelu': nn.LeakyReLU(),
    'relu': nn.ReLU()
}
activFun = activations_list[actif]
train_loader, ts, num_train, num_test = dataset.get_loaders(args)
for batch_idx, (data, target, _) in enumerate(ts):
    pass

data = data.to(device)

img_size = 28
epsilon = 1
in_channels = 1
latent_dim = 2
n_samples = 3000
n_chunk = 600


model = models.CoopCommSemiDual2(data, num_train, n_samples, n_chunk, epsilon, in_channels, latent_dim, activFun, img_size, device)
model = model.to(device)

start_time = time.time()
model.c_function.load_state_dict(torch.load('./savedmodels/decoder.pth'))


train_SemiDualOT(model, train_loader, args, device, log_interval=1)
print("after training")
print('training time elapsed {}s'.format(time.time() - start_time))