import torch
from torch import nn
from torch.nn.functional import mse_loss


class SeparationLoss(nn.Module):
    def __init__(self, lam=0.05, consistency=1.0):
        super().__init__()
        if not 0.0 <= lam < 1.0:
            raise ValueError("lam must be in [0, 1); for lam >= 1 the loss is unbounded below")
        self.lam = lam
        self.consistency = consistency

    def forward(self, r_hat, g_hat, r, g, m_low):
        sim = mse_loss(r_hat, r) + mse_loss(g_hat, g)
        cons = mse_loss(r_hat + g_hat, m_low)
        disc = mse_loss(r_hat, g) + mse_loss(g_hat, r)
        total = sim + self.consistency * cons - self.lam * disc
        return total, {"sim": sim.detach(), "cons": cons.detach(), "disc": disc.detach()}
