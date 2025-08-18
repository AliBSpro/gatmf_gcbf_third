# -*- coding: utf-8 -*-
"""
严格掩码与均场（论文 定义1 / IV-A）：半径外邻居数值与梯度均为 0。
"""
import torch

@torch.no_grad()
def build_graph(pos, R):
    B,N,_ = pos.shape
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)
    dist = torch.linalg.norm(diff, dim=-1)
    G = (dist < R).float()
    eye = torch.eye(N, device=pos.device).unsqueeze(0)
    G = G * (1.0 - eye)
    D = G.sum(dim=-1, keepdim=True)
    return G, D, dist

def masked_mean_field(weight, X, G):
    w = weight * G
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)
    X_bar = torch.bmm(w, X)
    return X_bar, w
