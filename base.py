import torch
from torch import nn
from torch.distributions import Normal, Independent, Bernoulli
import math

def conv_block(in_channels, out_channels, activation, BN=True, *args, **kwargs):
    if BN:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            activation
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, *args, **kwargs),
            activation
        )


def convt_block(in_channels, out_channels, activation, BN=True, *args, **kwargs):
    if BN:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            activation
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, *args, **kwargs),
            activation
        )


def fc_block(in_dim, out_dim, activation, BN=True, *args, **kwargs):
    if BN:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim, *args, **kwargs),
            nn.BatchNorm1d(out_dim),
            activation
        )
    else:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim, *args, **kwargs),
            activation
        )

activations_list = {
    'softplus': nn.Softplus(),
    'lrelu': nn.LeakyReLU(),
    'relu': nn.ReLU()
}

class MLPEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim,
                 activation, n_channels=None):
        super().__init__()
        if n_channels is None:
            n_channels = [256, 256, 256, 256]
        else:
            n_channels.reverse()
        self.in_channels = in_channels
        self.activation = activations_list[activation]
        self.n_channels = n_channels

        modules = []

        # Build Encoder
        for h_dim in n_channels:
            modules.append(fc_block(in_dim=in_channels, out_dim=h_dim, activation=self.activation))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(n_channels[-1], latent_dim)
        self.fc_var = nn.Linear(n_channels[-1], latent_dim)


    def forward(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var


class MLPDecoder(nn.Module):
    def __init__(self, in_channels, latent_dim,
                 activation, n_channels=None):
        super().__init__()
        if n_channels is None:
            n_channels = [256, 256, 256, 256]
        self.in_channels = in_channels
        self.activation = activations_list[activation]
        self.n_channels = n_channels

        self.decoder_input = fc_block(latent_dim, n_channels[0],activation=self.activation,BN=False)
        modules = []
        for i in range(len(self.n_channels) - 1):
            modules.append(fc_block(in_dim=self.n_channels[i], out_dim=self.n_channels[i+1],activation=self.activation))

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



class ConEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim,
                 activation, img_size=28, n_channels=None):
        super().__init__()
        if n_channels is None:
            n_channels = [32, 64, 128]
        else:
            n_channels = n_channels[:]
            n_channels.reverse()
        self.in_channels = in_channels
        self.activation = activations_list[activation]
        self.img_size = img_size
        self.n_channels = n_channels
        self.last_layer_size = math.ceil(img_size / (2 ** len(n_channels))) ** 2

        modules = []
        # Build Encoder
        for h_dim in n_channels:
            modules.append(conv_block(in_channels, h_dim, self.activation,
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


class ConDecoder(nn.Module):
    def __init__(self, in_channels, latent_dim,
                 activation, img_size=28, n_channels=None):
        super().__init__()
        if n_channels is None:
            n_channels = [128, 64, 32]
        self.in_channels = in_channels
        self.activation = activations_list[activation]
        self.img_size = img_size
        self.n_channels = n_channels

        self.first_layer_width = math.ceil(img_size / (2 ** len(self.n_channels)))
        self.first_layer_size = self.first_layer_width ** 2

        self.decoder_input = fc_block(latent_dim, self.n_channels[0] * self.first_layer_size, self.activation, False)

        modules = []
        for i in range(len(self.n_channels) - 1):
            if img_size == 28 and i == 0:
                modules.append(convt_block(self.n_channels[i], self.n_channels[i + 1], self.activation,
                                           kernel_size=3, stride=2, padding=1, output_padding=0))
            else:
                modules.append(convt_block(self.n_channels[i], self.n_channels[i + 1], self.activation,
                                           kernel_size=3, stride=2, padding=1, output_padding=1))

        self.decoder = nn.Sequential(*modules)
        final_module = [ 
            nn.ConvTranspose2d(self.n_channels[-1],
                               self.n_channels[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            self.activation,
            nn.Conv2d(self.n_channels[-1], out_channels=self.in_channels,
                      kernel_size=3, padding=1)]
        if self.in_channels == 3:
            final_module.append(nn.Sigmoid())
        self.final_layer = nn.Sequential(*final_module)

    def forward(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.n_channels[0], self.first_layer_width, self.first_layer_width)
        result = self.decoder(result)
        logit = self.final_layer(result)
        if self.in_channels == 1:
            return Bernoulli(logits=logit)
        else:
            return Normal(loc=logit, scale=0.1 * torch.ones_like(logit))