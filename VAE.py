import torch
from torch import nn
from torch.distributions import Normal, Independent, kl_divergence

from base import *
from utils import SaveModel

class VAE(nn.Module):

    def __init__(self, in_channels, latent_dim,
                 activation, img_size=28, device='cuda', n_channels=None):
        super().__init__()
        if in_channels == 1 or in_channels == 3:
            self.decoder = ConDecoder(in_channels, latent_dim, activation, img_size, n_channels)
            self.encoder = ConEncoder(in_channels, latent_dim, activation, img_size, n_channels)
        else:
            self.decoder = MLPDecoder(in_channels, latent_dim, activation, n_channels)
            self.encoder = MLPEncoder(in_channels, latent_dim, activation, n_channels)
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

    def loss(self, x, *args, **kwargs):
        """
        Computes the VAE loss function.
        """
        recons, z, z_mu, log_z_var = args
        if self.encoder.in_channels == 1 or self.encoder.in_channels == 3:
            recons_loss = recons.log_prob(x).sum([1, 2, 3]).mean()
        else:
            recons_loss = recons.log_prob(x).mean()


        q_z = Independent(Normal(loc=z_mu, scale=torch.exp(0.5 * log_z_var)),
                          reinterpreted_batch_ndims=1)

        p_z = Independent(Normal(loc=torch.zeros_like(z_mu), scale=torch.ones_like(log_z_var)),
                          reinterpreted_batch_ndims=1)
        kl_z = kl_divergence(q_z, p_z).mean()

        loss = -recons_loss + kl_z
        return loss, recons_loss, kl_z


    def train_model(self, train_loader, lr, epochs, save, seed, log_interval=20):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        for epoch in range(1, epochs + 1):
            loss_sum = 0
            recon_sum = 0
            kl_sum = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(self.device)
                optimizer.zero_grad()
                output = self.forward(data)
                loss, recon, kl = self.loss(data, *output)
                loss.backward()
                optimizer.step()
                scheduler.step()
                with torch.no_grad():
                    loss_sum += loss.item()
                    recon_sum += recon.item()
                    kl_sum += kl.item()

                if epoch % log_interval == 0 and batch_idx == len(train_loader)-1:
                    print(f"Epoch: {epoch}, Loss: {loss_sum / len(train_loader)}, Reconstruction:"
                        f" {recon_sum / len(train_loader)}, "
                        f"kl_z: {kl_sum / len(train_loader)}")
        SaveModel(self.decoder, "savedmodels", save)
        SaveModel(self, "savedmodels", "whole" + save)

