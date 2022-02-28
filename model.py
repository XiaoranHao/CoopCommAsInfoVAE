import torch
import math
from base import *
from torch.distributions import Normal, Independent
from torch.distributions.bernoulli import Bernoulli
import solver


class Decoder(nn.Module):
    def __init__(self, in_channels, latent_dim,
                 activation, img_size=28, n_channels=None):
        super(Decoder, self).__init__()
        if n_channels is None:
            n_channels = [128, 64, 32]
        self.in_channels = in_channels
        self.activation = activation
        self.img_size = img_size
        self.n_channels = n_channels

        self.first_layer_width = math.ceil(img_size / (2 ** len(self.n_channels)))
        self.first_layer_size = self.first_layer_width ** 2

        self.decoder_input = fc_block(latent_dim, self.n_channels[0] * self.first_layer_size, self.activation, False)

        modules = []
        for i in range(len(self.n_channels) - 1):
            if img_size == 28 and i == 0:
                modules.append(convt_block(self.n_channels[i], self.n_channels[i + 1], self.activation, False,
                                           kernel_size=3, stride=2, padding=1, output_padding=0))
            else:
                modules.append(convt_block(self.n_channels[i], self.n_channels[i + 1], self.activation, False,
                                           kernel_size=3, stride=2, padding=1, output_padding=1))

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(self.n_channels[-1],
                               self.n_channels[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            self.activation,
            nn.Conv2d(self.n_channels[-1], out_channels=self.in_channels,
                      kernel_size=3, padding=1))

    def forward(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.n_channels[0], self.first_layer_width, self.first_layer_width)
        result = self.decoder(result)
        logit = self.final_layer(result)
        return Bernoulli(logits=logit)


class NegEntropy(object):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def Omega(self, x):
        x = x.masked_fill(x <= 0., 1.)
        return (x * torch.log(x)).sum()


class EOT_sample(nn.Module):
    def __init__(self, n_sample, epsilon, in_channels, latent_dim, activation, img_size, n_channels=None):
        super(EOT_sample, self).__init__()
        self.e = epsilon
        self.entropy = NegEntropy(epsilon)
        self.c_function = Decoder(in_channels, latent_dim, activation, img_size, n_channels)

        mu = torch.zeros(n_sample, latent_dim)
        self.pz_true = Independent(Normal(loc=mu, scale=torch.ones_like(mu)),
                                   reinterpreted_batch_ndims=1)

        self.pz = (torch.ones(n_sample) / n_sample).to(device)

    def sinkhorn_scaling(self, x):
        self.px = torch.ones(len(x)) / len(x)
        self.z = self.pz_true.sample()
        dist = self.c_function(self.z)
        x = x.repeat(1, len(self.z), 1, 1).unsqueeze(2)
        C = -dist.log_prob(x).sum([2, 3, 4])
        P = ot.sinkhorn(self.px, self.pz, C, self.e, method='sinkhorn_log')
        return P, C

    def semi_dual_sgd(self, x):
        self.px = (torch.ones(len(x)) / len(x)).to(device)
        self.z = self.pz_true.sample().to(device)
        dist = self.c_function(self.z)
        x = x.repeat(1, len(self.z), 1, 1).unsqueeze(2)
        C = -dist.log_prob(x).sum([2, 3, 4])
        P, u, v = solver.solve_semi_dual_entropic(self.px, self.pz, C, reg=1, numItermax=1000)
        return P, C

    def forward(self, x):
        P, C = self.semi_dual_sgd(x)
        kl = self.entropy.Omega(P) - self.entropy.Omega(self.px) - self.entropy.Omega(self.pz)
        return P, C, kl

    def loss(self, P, C, kl):
        return (P * C).sum() + kl