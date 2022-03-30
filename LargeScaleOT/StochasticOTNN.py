import torch
import torch.nn as nn
import torch.nn.functional as func


class DualNet(nn.Module):
    def __init__(self, input_d=2, output_d=2):
        super().__init__()
        self.fc1 = nn.Linear(input_d, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, output_d)

    def forward(self, x):
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = func.relu(self.fc3(x))
        x = func.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class DiscretePotential(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.register_parameter(name="v", param=torch.nn.Parameter(torch.zeros(length)))

    def forward(self, idx):
        return self.v[idx]


class ContinuousPotential(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.u = DualNet(input_d=latent_dim, output_d=1)

    def forward(self, x):
        return self.u(x)


class PyTorchStochasticOT(nn.Module):

    def __init__(self, u, v, reg=1):
        super().__init__()
        self.reg = reg
        self.u = u
        self.v = v

    def dual_OT_batch_loss(self, M_batch, u_batch, v_batch):
        uv_cross = u_batch[:, None] + v_batch[None, :]
        H_epsilon = torch.exp((uv_cross - M_batch) / self.reg-1.)
        f_epsilon = - self.reg * H_epsilon
        return - torch.mean(uv_cross + f_epsilon)

    def forward(self, z_batch, idx, M):
        u_batch = self.u(z_batch)
        v_batch = self.v(idx)
        return self.dual_OT_batch_loss(M, u_batch, v_batch)
