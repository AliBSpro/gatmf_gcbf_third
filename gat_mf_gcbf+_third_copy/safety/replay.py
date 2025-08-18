# -*- coding: utf-8 -*-
import random, torch
class MaskedReplay:
    def __init__(self, capacity=200000):
        self.capacity = capacity; self.buf = []; self.idx = 0
    def push(self, s, a, r, s1, done, G, G1):
        item = (s.detach().cpu(), a.detach().cpu(), float(r), s1.detach().cpu(),
                float(done), G.detach().cpu(), G1.detach().cpu())
        if len(self.buf) < self.capacity: self.buf.append(None)
        self.buf[self.idx] = item; self.idx = (self.idx+1) % self.capacity
    def sample(self, batch):
        s,a,r,s1,d,G,G1 = zip(*random.sample(self.buf, batch))
        import torch
        return (torch.stack(s), torch.stack(a), torch.tensor(r).unsqueeze(-1),
                torch.stack(s1), torch.tensor(d).unsqueeze(-1),
                torch.stack(G), torch.stack(G1))
    def __len__(self): return len(self.buf)
