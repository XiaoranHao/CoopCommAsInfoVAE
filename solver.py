import torch
import warnings


def grad_semi_dual(beta, M, reg, v):
    """
    to compute the gradient w.r.t. dual variable v in regularized semi-dual OT problem.
    :param beta: Target measure. [Nt,]
    :param M: Cost matrix. [Ns,Nt]
    :param reg: regularization term
    :param v: dual variable of beta. [Nt,]

    :return: gradient
    """
    r = v - M
    max_r = torch.max(r, dim=1, keepdim=True)[0]
    exp_beta = torch.exp((r - max_r) / reg) * beta
    khi = exp_beta / exp_beta.sum(dim=1, keepdim=True)

    return beta - khi.mean(dim=0)


def averaged_sgd_entropic_transport(alpha, beta, M, reg, numItermax=10000, cur_v=None, lr=None, device='cuda', stopThr=1e-9):
    """
    solve regularized semi-dual OT problem by ASGD.

    :param alpha: Source measure [Nt,]
    :param beta: Target measure [Ns,]
    :param M: Cost matrix. [Ns,Nt]
    :param reg: regularization term
    :param numItermax: Number of iteration.
    :param lr: Learning rate.
    :param device: tensor device. cpu or cuda
    :param stopThr: stop threshold

    :return: optimal dual variable.
    """

    if lr is None:
        lr = 1. / max(alpha / reg)
    n_target = M.shape[1]
    if cur_v is None:
        cur_v = torch.zeros(n_target).to(device)
    ave_v = torch.zeros(n_target).to(device)

    for cur_iter in range(numItermax):
        k = cur_iter + 1
        cur_coord_grad = grad_semi_dual(beta, M, reg, cur_v)
        cur_v += (lr / torch.sqrt(torch.tensor(k))) * cur_coord_grad
        ave_v = (1. / k) * cur_v + (1 - 1. / k) * ave_v

    #     if cur_iter % 100 == 0:
    #         # we can speed up the process by checking for the error only all
    #         # the 10th iterations
    #
    #         opt_u = c_transform_entropic(beta, M, reg, ave_v)
    #         tmp2 = (torch.exp((opt_u[:, None] + ave_v[None, :] - M) / reg) *
    #                 alpha[:, None] * beta[None, :]).sum(dim=0)
    #         err = torch.norm(tmp2 - beta)  # violation of marginal
    #         if err < stopThr:
    #             break
    # warnings.warn("Sinkhorn did not converge.")
    return ave_v


def c_transform_entropic(beta, M, reg, v):
    """

    :param beta: Target measure [Ns,]
    :param M: Cost matrix. [Ns,Nt]
    :param reg: regularization term
    :param v: dual variable of beta. [Nt,]

    :return: u dual variable of alpha [Ns,]
    """

    r = v - M
    max_r = torch.max(r, dim=1, keepdim=True)[0]
    exp_beta = torch.exp((r - max_r) / reg) * beta
    u = -max_r - reg * torch.log(torch.sum(exp_beta, dim=1, keepdim=True))
    return u.squeeze(-1)


def solve_semi_dual_entropic(alpha, beta, M, reg, numItermax=10000, cur_v=None, lr=None, device='cuda'):
    """
    :param alpha: Source measure [Nt,]
    :param beta: Target measure [Ns,]
    :param M: Cost matrix. [Ns,Nt]
    :param reg: regularization term
    :param device: deivce
    :param numItermax: Maximum number of iterations
    :param lr: learning rate

    :return: optimal coupling pi, optimal dual variables u,v
    """

    opt_v = averaged_sgd_entropic_transport(alpha, beta, M, reg, numItermax, cur_v, lr, device)

    opt_u = c_transform_entropic(beta, M, reg, opt_v)

    # pi = (torch.exp((opt_u[:, None] + opt_v[None, :] - M) / reg) *
    #       alpha[:, None] * beta[None, :])

    uv_cross = opt_u[:, None] + opt_v[None, :]
    exponent = (uv_cross - M)/reg
    max_exponent = torch.max(exponent, dim=0, keepdim=True)[0]
    pi  = torch.exp(exponent-max_exponent)

    return pi, opt_u, opt_v, opt_u.mean()+opt_v.mean()
