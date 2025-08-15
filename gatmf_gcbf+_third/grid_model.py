# -*- coding: utf-8 -*-
import torch, numpy as np

class GridEnv:
    def __init__(self, n_agents=8, world_size=10.0, obstacles=None, dt=0.1, device="cpu"):
        self.n=n_agents; self.world=world_size; self.dt=dt; self.device=device
        self.max_speed=2.0; self.obstacles=obstacles or [(0.0,0.0,1.0)]
        self.r_safe=0.7; self.R_graph=3.5; self.reset()
    def reset(self, seed=None):
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self.pos=torch.tensor(rng.uniform(-self.world,self.world,size=(self.n,2)),dtype=torch.float32,device=self.device)
        self.vel=torch.zeros((self.n,2),dtype=torch.float32,device=self.device)
        self.goal=torch.tensor(rng.uniform(-self.world,self.world,size=(self.n,2)),dtype=torch.float32,device=self.device)
        return self.observe()
    def snapshot(self): return (self.pos.clone(), self.vel.clone(), self.goal.clone())
    def restore(self, snap): self.pos, self.vel, self.goal = [x.clone() for x in snap]
    def set_from_state(self, state):
        self.pos = state[:,0:2].to(self.device); self.vel = state[:,2:4].to(self.device); self.goal= state[:,4:6].to(self.device)
    def observe(self):
        dist = torch.linalg.norm(self.goal - self.pos, dim=-1, keepdim=True)
        deg  = torch.zeros((self.n,1), device=self.device)
        ones = torch.ones((self.n,2), device=self.device)
        s = torch.cat([self.pos,self.vel,self.goal,dist,deg,ones],dim=-1)  # (N,10)
        G = self.build_graph(self.pos, self.R_graph)
        return s, G
    @staticmethod
    def build_graph(pos, R):
        diff = pos.unsqueeze(0)-pos.unsqueeze(1); dist=torch.linalg.norm(diff, dim=-1)
        G = (dist < R).float(); G.fill_diagonal_(0.0); return G
    def step(self, a):
        a = torch.clamp(a, -1.0, 1.0)
        self.vel = torch.clamp(self.vel + a*self.dt, -self.max_speed, self.max_speed)
        self.pos = self.pos + self.vel*self.dt
        for d in [0,1]:
            low = self.pos[:,d] < -self.world; high = self.pos[:,d] > self.world
            self.pos[low,d] = -self.world; self.pos[high,d] = self.world; self.vel[low|high,d]*=-0.5
        dist_to_goal = torch.linalg.norm(self.goal - self.pos, dim=-1)
        reward = -dist_to_goal.mean()
        unsafe = self.unsafe_mask(); reward = reward - unsafe.float().mean()*5.0
        done = (dist_to_goal.mean() < 0.5)
        s,G = self.observe(); info={"unsafe_frac": unsafe.float().mean().item()}
        return s, G, float(reward), bool(done), info
    def unsafe_mask(self):
        N=self.n; unsafe=torch.zeros(N,dtype=torch.bool,device=self.device)
        diff=self.pos.unsqueeze(0)-self.pos.unsqueeze(1); dmat=torch.linalg.norm(diff, dim=-1)
        mask=(dmat>0)&(dmat<2*self.r_safe); unsafe|=mask.any(dim=-1)
        for (cx,cy,r) in self.obstacles:
            ctr=torch.tensor([cx,cy],dtype=torch.float32,device=self.device).unsqueeze(0)
            d=torch.linalg.norm(self.pos-ctr, dim=-1); unsafe|=(d<(r+self.r_safe))
        return unsafe
