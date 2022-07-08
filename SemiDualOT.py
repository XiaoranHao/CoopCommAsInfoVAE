import torch
from torch import nn

from base import *
from utils import SaveModel
import LargeScaleOT

import time
class CoopCommSemiDual(nn.Module):
    def __init__(self, num_data, epsilon, in_channels, latent_dim, activation, img_size,
                 maxiter=1, device='cuda', n_channels=None):
        super().__init__()
        if in_channels == 1 or in_channels == 3:
            self.c_function = ConDecoder(in_channels, latent_dim, activation, img_size, n_channels)
        else:
            self.c_function = MLPDecoder(in_channels, latent_dim, activation, n_channels)
        self.in_channels = in_channels
        self.epsilon = epsilon
        self.num_data = num_data
        # register v as buffer
        self.v = None
        self.latent_dim = latent_dim
        self.maxiter = maxiter
        self.device = device
        self.px = (torch.ones(num_data) / num_data).to(self.device)
        self.otsolver = LargeScaleOT.SemiDualOTsolver()
    def z_sample(self, n_sample):
        mu = torch.zeros(self.latent_dim)
        pz = Normal(loc=mu, scale=torch.ones_like(mu))
        return pz.sample([n_sample])

    def make_cost(self, x, z, n_xchunk, n_zchunk):
        # cost matrix shape x_n * z_m
        with torch.no_grad():
            z_chunk = torch.tensor_split(z,n_zchunk)
            C_all = []
            for j in range(len(z_chunk)):
                z = z_chunk[j]
                dist = self.c_function(z)
                C = []
                if self.in_channels == 1 or self.in_channels == 3:
                    x_total_batch = x.unsqueeze(1).expand(-1, len(z), -1, -1, -1)
                    x_chunk = torch.tensor_split(x_total_batch, n_xchunk)
                    for i in range(len(x_chunk)):
                        C_ = -dist.log_prob(x_chunk[i]).sum([2, 3, 4])
                        C.append(C_)
                else:
                    x_total_batch = x.unsqueeze(1).expand(-1, len(z), -1)
                    x_chunk = torch.tensor_split(x_total_batch, n_xchunk)
                    for i in range(len(x_chunk)):
                        C_ = -dist.log_prob(x_chunk[i])
                        C.append(C_)
                C = torch.cat(C)  
                C_all.append(C)
            C_all = torch.cat(C_all,dim=1)
            return C_all

    def SemiDual_Train(self, C):
        # SGD training of SemiDual EOT
        pz = None
        v = self.otsolver.averaged_sgd_entropic_transport(pz, self.px, C, reg=self.epsilon, numItermax=self.maxiter, cur_v=self.v, lr=1000., device=self.device)
        self.v = v

    def forward(self, x, idx, z, C):
        W_xz = self.otsolver(self.px, C, self.epsilon, self.v).t()[idx]
        categ = torch.distributions.categorical.Categorical(W_xz)
        s = categ.sample()
        z_sample = z[s]
        dist = self.c_function(z_sample)
        C_ = -dist.log_prob(x)

        return C_, z_sample

    def DecLoss(self, C_):
        return C_.mean()

    def train_model(self, train_dataloader, data_all, lr, epochs, n_iter, n_samples, n_xchunk, n_zchunk, device, save, seed, log_interval=10):
        print(len(train_dataloader.dataset))
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        loss_best = 10000
        for epoch in range(1, epochs + 1):
            loss_sum = 0
            for i in range(1, 1 + n_iter):
                if i % 20 == 1:
                    z_all = self.z_sample(n_samples).to(device)
                    C_all = self.make_cost(data_all.to(device), z_all, n_xchunk, n_zchunk).to(device)  
                    z_chunk = torch.tensor_split(torch.arange(0,len(z_all)), 20)
                x_index = torch.arange(0,self.num_data)
                z_index = z_chunk[(i-1)%20]
                z = z_all[z_index]
                C = C_all[x_index][:,z_index].to(device)
                self.SemiDual_Train(C.t())
                if i % 500 == 0:
                    print(f"Iters {i}/{n_iter}")
            self.otsolver.reset_iter()
            if self.in_channels == 1 or self.in_channels ==3:
                for batch_idx, (data, target, idx) in enumerate(train_dataloader):
                    data = data.to(device)
                    # C_batch = C_all[idx].to(device)
                    optimizer.zero_grad()
                    C_, z_ = self(data, idx, z_all, C_all.t())
                    loss = self.DecLoss(C_)
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()
                    with torch.no_grad():
                        loss_sum += loss.item()
                    
                    if epoch % log_interval == 0 and batch_idx == len(train_dataloader)-1:
                        loss_sum = loss_sum / len(train_dataloader)
                        print(f"Epoch: {epoch}, Loss: {loss_sum}")
                        if loss_sum  < loss_best:
                            loss_best = loss_sum 
                            SaveModel(self, "savedmodels", save)
            else:
                for batch_idx, (data, idx) in enumerate(train_dataloader):
                    data = data.to(device)
                    optimizer.zero_grad()
                    C_, z_ = self(data, idx, z_all, C_all.t())
                    loss = self.DecLoss(C_)
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()
                    with torch.no_grad():
                        loss_sum += loss.item()
                    
                    if epoch % log_interval == 0 and batch_idx == len(train_dataloader)-1:
                        loss_sum = loss_sum / len(train_dataloader)
                        print(f"Epoch: {epoch}, Loss: {loss_sum}")
                        if loss_sum  < loss_best:
                            loss_best = loss_sum 
                            SaveModel(self, "savedmodels", save)
        SaveModel(self, "savedmodels", save)


    def train_approx(self, train_dataloader, data_all, lr, epochs, n_iter, n_samples, n_xchunk, n_zchunk, device, save, seed, log_interval=10):
        print(len(train_dataloader.dataset))
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_best = 10000
        for epoch in range(1, epochs + 1):
            loss_sum = 0
            z_all = self.z_sample(n_samples).to(device)
            C_all = self.make_cost(data_all.to(device), z_all, n_xchunk, n_zchunk).to(device) 
            x_index = torch.arange(0,self.num_data)
            z_index = torch.randperm(n_samples, device=device)[:2000]
            for i in range(1, 1 + n_iter):
                z_index = torch.randperm(n_samples, device=device)[:2000]
                z = z_all[z_index]
                C = C_all[x_index][:,z_index].to(device)
                self.SemiDual_Train(C.t())
                if i % 500 == 0:
                    print(f"Iters {i}/{n_iter}")
            self.otsolver.reset_iter()
            if self.in_channels == 1 or self.in_channels ==3:
                for batch_idx, (data, target, idx) in enumerate(train_dataloader):
                    data = data.to(device)
                    # C_batch = C_all[idx].to(device)
                    optimizer.zero_grad()
                    C_, z_ = self(data, idx, z_all, C_all.t())
                    loss = self.DecLoss(C_)
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()
                    with torch.no_grad():
                        loss_sum += loss.item()
                    
                    if epoch % log_interval == 0 and batch_idx == len(train_dataloader)-1:
                        loss_sum = loss_sum / len(train_dataloader)
                        print(f"Epoch: {epoch}, Loss: {loss_sum}")
                        if loss_sum  < loss_best:
                            loss_best = loss_sum 
                            SaveModel(self, "savedmodels", save)
            else:
                for batch_idx, (data, idx) in enumerate(train_dataloader):
                    data = data.to(device)
                    optimizer.zero_grad()
                    C_, z_ = self(data, idx, z_all, C_all.t())
                    loss = self.DecLoss(C_)
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()
                    with torch.no_grad():
                        loss_sum += loss.item()
                    
                    if epoch % log_interval == 0 and batch_idx == len(train_dataloader)-1:
                        loss_sum = loss_sum / len(train_dataloader)
                        print(f"Epoch: {epoch}, Loss: {loss_sum}")
                        if loss_sum  < loss_best:
                            loss_best = loss_sum 
                            SaveModel(self, "savedmodels", save)
        SaveModel(self, "savedmodels", save)