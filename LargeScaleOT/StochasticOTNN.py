from tkinter import E
import torch
import torch.nn as nn
import torch.nn.functional as func


# class DualNet(nn.Module):
#     def __init__(self, input_d=2, output_d=2):
#         super().__init__()
#         self.fc1 = nn.Linear(input_d, 128)
#         self.fc2 = nn.Linear(128, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.fc4 = nn.Linear(256, 128)
#         self.fc5 = nn.Linear(128, output_d)

#     def forward(self, x):
#         x = func.relu(self.fc1(x))
#         x = func.relu(self.fc2(x))
#         x = func.relu(self.fc3(x))
#         x = func.relu(self.fc4(x))
#         x = self.fc5(x)
#         return x

class DualNet(nn.Module):
    def __init__(self, input_d=2, output_d=2):
        super().__init__()
        self.fc1 = nn.Linear(input_d, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_d)

    def forward(self, x):
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DiscretePotential(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.register_parameter(name="u", param=torch.nn.Parameter(torch.zeros(length) + 5))

    def forward(self, idx):
        return self.u[idx]


class ContinuousPotential(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.v = DualNet(input_d=latent_dim, output_d=1)

    def forward(self, x):
        return self.v(x).squeeze(-1)


class PyTorchStochasticOT(nn.Module):

    def __init__(self, u, v, reg=1):
        super().__init__()
        self.reg = reg
        self.u = u
        self.v = v

    def dual_OT_batch_loss(self, u_batch, v_batch,  M_batch):
        uv_cross = u_batch[:, None] + v_batch[None, :]
        exponent_ = (uv_cross - M_batch)/self.reg-1.
        if torch.any(exponent_>55) and self.training:
            exponent = torch.where(exponent_ > 55, exponent_- 15, exponent_)
            print("overflow prevented")
        else:
            exponent = exponent_
        # exponent = (uv_cross - M_batch)/self.reg-1.
        # if torch.any(exponent>55) and self.training:
        #     exponent = exponent - 30
        max_exponent = torch.max(exponent, dim=1, keepdim=True)[0]
        H_epsilon = torch.exp(exponent)
        H_epsilon_weight = torch.exp(exponent-max_exponent)
        f_epsilon = - self.reg * H_epsilon
        # print(f_epsilon.min())
        neg_loss = - torch.mean(uv_cross + f_epsilon)
        # return neg_loss, H_epsilon

        return neg_loss, H_epsilon_weight


    def forward(self, idx, z_batch, M):
        u_batch = self.u(idx)
        v_batch = self.v(z_batch)
        return self.dual_OT_batch_loss(u_batch, v_batch, M)


class DualOT(nn.Module):
    def __init__(self, latent_dim, num_data, reg, maxiter, lr1, lr2):
        super().__init__()
        u = DiscretePotential(num_data)
        v = ContinuousPotential(latent_dim)
        self.sto = PyTorchStochasticOT(u, v, reg)
        self.maxiter = maxiter
        self.lr1 = lr1
        self.lr2 = lr2

        trainable_params = [{"params": self.sto.u.parameters(), "lr": self.lr2},
                    {"params": self.sto.v.parameters()}]
        self.optimizer = torch.optim.Adam(trainable_params, lr=self.lr1)

    def learn_OT(self, d_idx, z_batch, M_batch):

        # trainable_params = [{"params": self.sto.u.parameters(), "lr": self.lr2},
        #                     {"params": self.sto.v.parameters()}]

        # optimizer = torch.optim.Adam(trainable_params, lr=self.lr1)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.maxiter, eta_min=1e-8)

        for i in range(1, self.maxiter+1):
            self.optimizer.zero_grad()
            loss_batch, H_epsilon = self.sto(d_idx, z_batch, M_batch)
            loss_batch.backward()
            self.optimizer.step()
            # scheduler.step()
            # if i % 10 == 1:
            #     print(i, "OT loss: ", loss_batch.item())

    def forward(self, d_idx, z_batch, M_batch,training=True):
        if training:
            self.learn_OT(d_idx, z_batch, M_batch)
        with torch.no_grad():
            _, H_epsilon = self.sto(d_idx, z_batch, M_batch)
            print(_.item())
        # importance weight
        return H_epsilon


