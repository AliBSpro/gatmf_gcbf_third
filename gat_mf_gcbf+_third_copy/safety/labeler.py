# -*- coding: utf-8 -*-
"""
有限时域 T 标注（IV-C）：整段安全→进 D_C；任一步不安全→进 D_A。
"""
import torch

@torch.no_grad()
def label_batch(env, states, Gs, actor_target, attn_actor_target, horizon_T:int):
    device = states.device
    B,N,S = states.shape
    mask_safe = torch.zeros((B,N,1), device=device)
    mask_unsafe = torch.zeros((B,N,1), device=device)
    for b in range(B):
        snap = env.snapshot()
        env.set_from_state(states[b])
        safe_all = torch.ones((N,), dtype=torch.bool, device=device)
        unsafe_any = torch.zeros((N,), dtype=torch.bool, device=device)
        for t in range(horizon_T):
            s, G = env.observe()
            s = s.unsqueeze(0); G = G.unsqueeze(0)
            att = attn_actor_target(s, G)
            s_bar = torch.bmm(att, s)
            s_all = torch.cat([s, s_bar], dim=-1)
            a = actor_target(s_all).squeeze(0)
            s, G, r, done, info = env.step(a)
            umask = env.unsafe_mask()
            unsafe_any |= umask
            safe_all &= ~umask
            if done: break
        mask_safe[b,:,0] = safe_all
        mask_unsafe[b,:,0] = unsafe_any
        env.restore(snap)
    return mask_safe, mask_unsafe
