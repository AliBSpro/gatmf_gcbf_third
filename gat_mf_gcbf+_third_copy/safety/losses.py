# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

def cbf_hinge_losses(h_t, h_t1, dt, alpha=1.0, gamma_margin=0.02,
                     mask_safe=None, mask_unsafe=None,
                     w_deriv=1.0, w_safe=1.0, w_unsafe=1.0):
    dot_h = (h_t1 - h_t) / (dt + 1e-12)
    loss_deriv = torch.relu(gamma_margin - dot_h - alpha * h_t).mean()
    loss_safe = torch.tensor(0.0, device=h_t.device)
    loss_unsafe = torch.tensor(0.0, device=h_t.device)
    if mask_safe is not None:
        loss_safe = (torch.relu(gamma_margin - h_t) * mask_safe).sum() / (mask_safe.sum()+1e-6)
    if mask_unsafe is not None:
        loss_unsafe = (torch.relu(gamma_margin + h_t) * mask_unsafe).sum() / (mask_unsafe.sum()+1e-6)
    L_cbf = w_deriv * loss_deriv + w_safe * loss_safe + w_unsafe * loss_unsafe
    return L_cbf, dict(loss_deriv=loss_deriv.item(), loss_safe=loss_safe.item(), loss_unsafe=loss_unsafe.item())

def ctrl_distill_loss(pi, pi_qp, eta_ctrl=1.0):
    return eta_ctrl * torch.nn.functional.mse_loss(pi, pi_qp)
