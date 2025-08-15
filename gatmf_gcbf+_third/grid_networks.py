# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

STATE_DIM = 10
ACT_DIM = 2

class Attention(nn.Module):
    def __init__(self, in_dim=STATE_DIM, qk_dim=32):
        super().__init__()
        self.Wq = nn.Linear(in_dim, qk_dim, bias=False)
        self.Wk = nn.Linear(in_dim, qk_dim, bias=False)
    def forward(self, s, G):
        q = self.Wq(s); k = self.Wk(s)
        logits = torch.matmul(q, k.transpose(-1,-2))
        w = torch.square(logits) * G
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)
        return w

class Actor(nn.Module):
    def __init__(self, in_dim=STATE_DIM*2, out_dim=ACT_DIM, h=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h); self.fc2 = nn.Linear(h, h); self.fc3 = nn.Linear(h, out_dim)
    def forward(self, s_all):
        x = F.relu(self.fc1(s_all)); x = F.relu(self.fc2(x)); return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, sdim=STATE_DIM*2, adim=ACT_DIM*2, h=128):
        super().__init__()
        self.fc1 = nn.Linear(sdim+adim, h); self.fc2 = nn.Linear(h,h); self.fc3 = nn.Linear(h,1)
    def forward(self, s_all, a_all):
        x = torch.cat([s_all, a_all], dim=-1); x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x)); return self.fc3(x)

class CostCritic(nn.Module):
    def __init__(self, sdim=STATE_DIM*2, adim=ACT_DIM*2, h=128):
        super().__init__()
        self.fc1 = nn.Linear(sdim+adim, h); self.fc2 = nn.Linear(h,h); self.fc3 = nn.Linear(h,1)
    def forward(self, s_all, a_all):
        x = torch.cat([s_all, a_all], dim=-1); x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x)); return self.fc3(x)
