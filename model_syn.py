
import torch
from base import *
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, Independent, Dirichlet, kl_divergence
import LargeScaleOT


class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim,
                 activation, n_channels=None):
        super().__init__()
        if n_channels is None:
            n_channels = [256, 256, 256, 256]
        else:
            n_channels.reverse()
        self.in_channels = in_channels
        self.activation = activation
        self.n_channels = n_channels

        modules = []

        # Build Encoder
        for h_dim in n_channels:
            modules.append(fc_block(in_dim=in_channels, out_dim=h_dim, activation=activation))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(n_channels[-1], latent_dim)
        self.fc_var = nn.Linear(n_channels[-1], latent_dim)


    def forward(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, in_channels, latent_dim,
                 activation, n_channels=None):
        super(Decoder, self).__init__()
        if n_channels is None:
            n_channels = [128, 256, 512]
        self.in_channels = in_channels
        self.activation = activation
        self.n_channels = n_channels

        self.decoder_input = fc_block(latent_dim, n_channels[0],activation=activation,BN=False)
        modules = []
        for i in range(len(self.n_channels) - 1):
            modules.append(fc_block(in_dim=self.n_channels[i], out_dim=self.n_channels[i+1],activation=activation))

        self.decoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(n_channels[-1], in_channels)
        self.fc_var = nn.Linear(n_channels[-1], in_channels)

    def forward(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return Independent(Normal(loc=mu, scale=torch.exp(0.5 * log_var)),
                          reinterpreted_batch_ndims=1)

class VAE(nn.Module):

    def __init__(self, in_channels, latent_dim,
                 activation, device='cuda', n_channels=None):
        super().__init__()
        self.decoder = Decoder(in_channels, latent_dim, activation, n_channels)
        self.encoder = Encoder(in_channels, latent_dim, activation, n_channels)
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

        recons_loss = recons.log_prob(x).mean()

        q_z = Independent(Normal(loc=z_mu, scale=torch.exp(0.5 * log_z_var)),
                          reinterpreted_batch_ndims=1)

        p_z = Independent(Normal(loc=torch.zeros_like(z_mu), scale=torch.ones_like(log_z_var)),
                          reinterpreted_batch_ndims=1)
        kl_z = kl_divergence(q_z, p_z).mean()

        loss = -recons_loss + kl_z
        return loss, recons_loss, kl_z



## subroutine with q training
class CoopCommDualOT_v1(nn.Module):

    def __init__(self, num_data, n_chunk, reg, in_channels, latent_dim, activation,
                 n_channels=None):
        super().__init__()
        self.c_function = Decoder(in_channels, latent_dim, activation, n_channels)
        self.latent_dim = latent_dim

        self.num_data = num_data
        self.n_chunk = n_chunk
        self.reg = reg
        self.DualOT = LargeScaleOT.DualOT(latent_dim, num_data, reg, maxiter=1, lr1=1e-5, lr2=1e-2)

    def make_cost(self, x, z):
        with torch.no_grad():
            z_chunk = torch.tensor_split(z,5)
            C_all = []
            for j in range(len(z_chunk)):
                z = z_chunk[j]
                dist = self.c_function(z)
                x_total_batch = x.unsqueeze(1).expand(-1, len(z), -1)
                x_chunk = torch.tensor_split(x_total_batch, self.n_chunk)
                C = []
                for i in range(len(x_chunk)):
                    C_ = -dist.log_prob(x_chunk[i])
                    C.append(C_)
                C = torch.cat(C)  
                C_all.append(C)
            C_all = torch.cat(C_all,dim=1)
            return C_all    

    def z_sample(self, n_sample):
        mu = torch.zeros(self.latent_dim)
        pz = Normal(loc=mu, scale=torch.ones_like(mu))
        return pz.sample([n_sample])

    def set_reg(self, reg):
        self.DualOT.sto.reg = reg

    def forward(self, x, idx, z, C):
        W_xz = self.DualOT(idx, z, C, training=False)
        categ = torch.distributions.categorical.Categorical(W_xz)
        s = categ.sample()
        z_sample = z[s]
        dist = self.c_function(z_sample)
        C_ = -dist.log_prob(x)

        return C_, z_sample

    def DecLoss(self, C_):
        return C_.mean()


def train(model, train_dataloader, data_all, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    for epoch in range(1, args.epochs + 1):
        loss_sum = 0
        loss_best = 10
        z_all = model.z_sample(args.n_samples).to(device)
        C_all = model.make_cost(data_all.to(device), z_all).to('cuda')
        for i in range(1, 1 + args.n_iter):
            x_index = torch.arange(0,7500)
            z_index = torch.randint(0, args.n_samples, (args.z_bs,))
            z = z_all[z_index]
            C = C_all[x_index][:,z_index].to(device)
            model.DualOT.learn_OT(x_index, z, C)
            if i % 100 == 0:
                print(f"Iters {i}/{args.n_iter}")
                model.DualOT(x_index, z, C)
        print("OT done")
        for batch_idx, (data, idx) in enumerate(train_dataloader):
            data = data.to(device)
            C_batch = C_all[idx].to(device)
            optimizer.zero_grad()
            C_, z_ = model(data, idx, z_all, C_batch)
            loss = model.DecLoss(C_)
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                loss_sum += loss.item()
            
            if batch_idx == len(train_dataloader)-1:
                print(f"Epoch: {epoch}, Loss: {loss_sum / len(train_dataloader)}")
                if loss_sum / len(train_dataloader) < loss_best:
                    lost_best = loss_sum / len(train_dataloader)
                    torch.save(model.state_dict(), "syn_ot.pth")
    print("best_loss: ",loss_best)