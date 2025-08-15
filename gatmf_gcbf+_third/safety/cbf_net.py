# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class CBFNet(nn.Module):
    """ hθ: [s, s̄] -> h_i ；与论文 IV-A 对齐的逐体 MLP 结构 """
    def __init__(self, in_dim, hid1=256, hid2=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid1)
        self.fc2 = nn.Linear(hid1, hid2)
        self.fc3 = nn.Linear(hid2, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

@torch.no_grad()
def soft_update(src: nn.Module, dst: nn.Module, tau: float):
    for p, tp in zip(src.parameters(), dst.parameters()):
        tp.data.mul_(1-tau).add_(tau * p.data)
