# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class CtrlNet(nn.Module):
    """ πϕ: [s, s̄] -> a_i （动作用 tanh 实现盒约束） """
    def __init__(self, in_dim, out_dim=2, hid1=256, hid2=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid1)
        self.fc2 = nn.Linear(hid1, hid2)
        self.fc3 = nn.Linear(hid2, out_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

@torch.no_grad()
def soft_update(src: nn.Module, dst: nn.Module, tau: float):
    for p, tp in zip(src.parameters(), dst.parameters()):
        tp.data.mul_(1-tau).add_(tau * p.data)
