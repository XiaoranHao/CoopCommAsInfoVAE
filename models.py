import torch
import math
from base import *
from torch.distributions import Normal, Independent, Dirichlet, kl_divergence
from torch.distributions.bernoulli import Bernoulli
import solver
import ot
import LargeScaleOT
torch.set_default_dtype(torch.float32)


class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim,
                 activation, img_size=28, n_channels=None):
        super().__init__()
        if n_channels is None:
            n_channels = [32, 64, 128]
        else:
            n_channels.reverse()
        self.in_channels = in_channels
        self.activation = activation
        self.img_size = img_size
        self.n_channels = n_channels
        self.last_layer_size = math.ceil(img_size / (2 ** len(n_channels))) ** 2

        modules = []

        # Build Encoder
        for h_dim in n_channels:
            modules.append(conv_block(in_channels, h_dim, activation,
                                      kernel_size=3, stride=2, padding=1)
                           )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(n_channels[-1] * self.last_layer_size, latent_dim)
        self.fc_var = nn.Linear(n_channels[-1] * self.last_layer_size, latent_dim)

    def forward(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var


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

    def omega(self, x):
        x = x.masked_fill(x <= 0., 1.)
        return self.epsilon * (x * torch.log(x)).sum()


class CoopCommSemiDual(nn.Module):
    def __init__(self, n_sample, n_chunk, epsilon, in_channels, latent_dim, activation, img_size, device,
                 maxiter=1000, n_channels=None, method='sample'):
        super(CoopCommSemiDual, self).__init__()
        self.e = epsilon
        self.entropy = NegEntropy(epsilon)
        self.c_function = Decoder(in_channels, latent_dim, activation, img_size, n_channels)
        self.latent_dim = latent_dim
        self.n_sample = n_sample
        self.device = device
        self.maxiter = maxiter
        self.method = method
        self.n_chunk = n_chunk

    def z_sample(self, n_sample):
        mu = torch.zeros(n_sample, self.latent_dim)
        pz_true = Independent(Normal(loc=mu, scale=torch.ones_like(mu)), reinterpreted_batch_ndims=1)
        return pz_true.sample().to(self.device)

    def sinkhorn_scaling(self, x, z):
        px = (torch.ones(len(x)) / len(x)).to(self.device)
        pz = (torch.ones(len(z)) / len(z)).to(self.device)

        # px = torch.tensor(ot.utils.unif(len(x))).to(self.device)
        # pz = torch.tensor(ot.utils.unif(len(z))).to(self.device)

        dist = self.c_function(z)
        x = x.expand(-1, len(z), -1, -1).unsqueeze(2)
        C = -dist.log_prob(x).sum([2, 3, 4])
        with torch.no_grad():
            P = ot.sinkhorn(px, pz, C, self.e, method='sinkhorn_log')
        kl = self.entropy.omega(P) - self.entropy.omega(px) - self.entropy.omega(pz)

        return P, C, kl

    def semi_dual_sgd(self, x, z):
        px = (torch.ones(len(x)) / len(x)).to(self.device)
        pz = (torch.ones(len(z)) / len(z)).to(self.device)

        dist = self.c_function(z)
        x = x.expand(-1, len(z), -1, -1).unsqueeze(2)
        C = -dist.log_prob(x).sum([2, 3, 4])
        with torch.no_grad():
            P, u, v = solver.solve_semi_dual_entropic(px, pz, C, reg=self.e, device=self.device, numItermax=self.maxiter)
        P_con = P * len(x)
        kl = torch.nan_to_num(P_con * (torch.log(P_con) - torch.log(pz)))
        kl = kl.sum(dim=1).mean()

        return P, C, kl

    def semi_dual_sgd_sample(self, x, z):
        px = (torch.ones(len(x)) / len(x)).to(self.device)
        pz = (torch.ones(len(z)) / len(z)).to(self.device)
        with torch.no_grad():
            dist = self.c_function(z)
            x_exp = x.expand(-1, len(z), -1, -1).unsqueeze(2)
            x_chunk = torch.chunk(x_exp, self.n_chunk)
            C = []
            for i in range(self.n_chunk):

                C_ = -dist.log_prob(x_chunk[i]).sum([2, 3, 4])
                C.append(C_)
            C = torch.cat(C)
            P, u, v = solver.solve_semi_dual_entropic(px, pz, C, reg=self.e, device=self.device, numItermax=self.maxiter)

        P_con = P * len(x)
        kl = torch.nan_to_num(P_con * (torch.log(P_con) - torch.log(pz)))
        kl = kl.sum(dim=1).mean()
        categ = torch.distributions.categorical.Categorical(P_con)
        s = categ.sample()
        z_sample = z[s]
        dist = self.c_function(z_sample)
        C = -dist.log_prob(x).sum([1, 2, 3])

        return P, C, kl, z_sample

    def forward(self, x, n_sample=None):
        if n_sample is None:
            n_sample = self.n_sample
        z = self.z_sample(n_sample)
        if self.method == 'sample':
            P, C, kl, z_ = self.semi_dual_sgd_sample(x, z)
        else:
            P, C, kl = self.semi_dual_sgd(x, z)
        return P, C, kl, z

    def EotLoss(self, P, C, kl):
        if self.method == 'sample':
            return C.mean() + self.e * kl
        else:
            return (P * C).sum() + self.e * kl


class Dirichletp(CoopCommSemiDual):
    def __init__(self, *args, **kwargs):
        super(Dirichletp, self).__init__(*args, **kwargs)

    def z_sample(self, n_sample):
        alpha = 0.1 * torch.ones(self.latent_dim)
        pz_true = Dirichlet(alpha)
        return pz_true.sample([n_sample]).to(self.device)


class CoopCommDualOT(nn.Module):

    def __init__(self, num_data, n_sample, n_chunk, reg, in_channels, latent_dim, activation, img_size, device,
                 n_channels=None):
        super().__init__()
        self.c_function = Decoder(in_channels, latent_dim, activation, img_size, n_channels)
        self.latent_dim = latent_dim

        self.n_sample = n_sample
        self.num_data = num_data
        self.n_chunk = n_chunk
        self.reg = reg
        self.DualOT = LargeScaleOT.DualOT(latent_dim, num_data, reg, maxiter=1, lr1=3e-5, lr2=5e-2)
        self.device = device

    def z_sample(self, n_sample):
        mu = torch.zeros(self.latent_dim)
        pz = Normal(loc=mu, scale=torch.ones_like(mu))
        return pz.sample([n_sample]).to(self.device)

    def set_reg(self, reg):
        self.DualOT.sto.reg = reg

    def forward(self, x, idx, n_sample=None):
        if n_sample is None:
            n_sample = self.n_sample
        z = self.z_sample(n_sample)
        with torch.no_grad():
            dist = self.c_function(z)
            x_total_batch = x.expand(-1, len(z), -1, -1).unsqueeze(2)
            x_chunk = torch.tensor_split(x_total_batch, self.n_chunk)
            C = []
            for i in range(len(x_chunk)):
                C_ = -dist.log_prob(x_chunk[i]).sum([2, 3, 4])
                C.append(C_)
            C = torch.cat(C)
        W_xz = self.DualOT(idx, z, C)
        # print(W_xz.max())
        # print(W_xz.min())

        categ = torch.distributions.categorical.Categorical(W_xz)
        s = categ.sample()
        z_sample = z[s]
        dist = self.c_function(z_sample)
        C_ = -dist.log_prob(x).sum([1, 2, 3])

        return C, C_, z, z_sample

    def EotLoss(self, C):
        return C.mean()


class VAE(nn.Module):

    def __init__(self, in_channels, latent_dim,
                 activation, img_size=28, device='cuda', n_channels=None):
        super().__init__()
        self.decoder = Decoder(in_channels, latent_dim, activation, img_size, n_channels)
        self.encoder = Encoder(in_channels, latent_dim, activation, img_size, n_channels)
        self.device = device

    def forward(self, input, **kwargs):
        z_mu, log_z_var = self.encoder(input)
        z = self.reparameterize(z_mu, log_z_var, self.device)
        recon = self.decoder(z)
        return recon, z, z_mu, log_z_var

    @staticmethod
    def reparameterize(mu, logvar, device):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param device: current device CPU/GPU
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=device)
        return eps * std + mu

    @staticmethod
    def loss(x, *args, **kwargs):
        """
        Computes the VAE loss function.
        """
        recons, z, z_mu, log_z_var = args

        recons_loss = recons.log_prob(x).sum([1, 2, 3]).mean()

        q_z = Independent(Normal(loc=z_mu, scale=torch.exp(0.5 * log_z_var)),
                          reinterpreted_batch_ndims=1)

        p_z = Independent(Normal(loc=torch.zeros_like(z_mu), scale=torch.ones_like(log_z_var)),
                          reinterpreted_batch_ndims=1)
        kl_z = kl_divergence(q_z, p_z).mean()

        loss = -recons_loss + kl_z
        return loss, recons_loss, kl_z

