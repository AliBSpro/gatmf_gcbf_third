# -*- coding: utf-8 -*-
import torch, torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from grid_model import GridEnv
from grid_networks import Attention, Actor, Critic, CostCritic, STATE_DIM, ACT_DIM
from safety.cbf_net import CBFNet, soft_update as soft_update_cbf
from safety.ctrl_net import CtrlNet, soft_update as soft_update_pi
from safety.qp import qp_teacher
from safety.labeler import label_batch
from safety.losses import cbf_hinge_losses, ctrl_distill_loss
from safety.replay import MaskedReplay

class Trainer:
    def __init__(self, device="cpu", n_agents=8):
        self.device=device; self.env=GridEnv(n_agents=n_agents, device=device)
        self.gamma=0.98; self.gamma_c=0.98; self.tau=0.01
        self.batch=64; self.update_after=1000; self.update_every=50
        self.horizon_T=16; self.R=self.env.R_graph

        # 注意力
        self.attn_actor  = Attention().to(device)
        self.attn_actor_t= Attention().to(device); self.attn_actor_t.load_state_dict(self.attn_actor.state_dict())
        self.attn_critic = Attention().to(device)
        self.attn_critic_t=Attention().to(device); self.attn_critic_t.load_state_dict(self.attn_critic.state_dict())

        # RL 模块
        self.actor=Actor(in_dim=STATE_DIM*2,out_dim=ACT_DIM).to(device)
        self.actor_t=Actor(in_dim=STATE_DIM*2,out_dim=ACT_DIM).to(device); self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic=Critic().to(device); self.critic_t=Critic().to(device); self.critic_t.load_state_dict(self.critic.state_dict())
        self.cost_critic=CostCritic().to(device); self.cost_critic_t=CostCritic().to(device); self.cost_critic_t.load_state_dict(self.cost_critic.state_dict())

        # GCBF+ 模块
        self.cbf=CBFNet(in_dim=STATE_DIM*2).to(device)
        self.cbf_t=CBFNet(in_dim=STATE_DIM*2).to(device); self.cbf_t.load_state_dict(self.cbf.state_dict())
        self.pi=CtrlNet(in_dim=STATE_DIM*2, out_dim=ACT_DIM).to(device)
        self.pi_t=CtrlNet(in_dim=STATE_DIM*2, out_dim=ACT_DIM).to(device); self.pi_t.load_state_dict(self.pi.state_dict())

        # 优化器
        self.opt_actor=optim.Adam(self.actor.parameters(),lr=2e-4)
        self.opt_critic=optim.Adam(self.critic.parameters(),lr=2e-4)
        self.opt_cost=optim.Adam(self.cost_critic.parameters(),lr=2e-4)
        self.opt_attn_a=optim.Adam(self.attn_actor.parameters(),lr=1e-4)
        self.opt_attn_c=optim.Adam(self.attn_critic.parameters(),lr=1e-4)
        self.opt_cbf=optim.Adam(self.cbf.parameters(),lr=1e-4)
        self.opt_pi=optim.Adam(self.pi.parameters(),lr=1e-4)

        # CBF 超参（论文量级）
        self.alpha=1.0; self.gamma_margin=0.02; self.dt=self.env.dt; self.eta_ctrl=1.0

        # 拉格朗日对偶
        self.lam=torch.tensor(0.0, device=device); self.eta_lam=1e-3; self.lam_max=10.0; self.cost_budget=0.0

        self.buf=MaskedReplay(200000)

    @torch.no_grad()
    def build_state_all(self, s, G, attn):
        w = attn(s, G); s_bar = torch.bmm(w, s); s_all = torch.cat([s, s_bar], dim=-1); return w, s_bar, s_all

    def soft_update_all(self):
        for (src,dst) in [(self.actor,self.actor_t),(self.critic,self.critic_t),(self.cost_critic,self.cost_critic_t),
                          (self.attn_actor,self.attn_actor_t),(self.attn_critic,self.attn_critic_t)]:
            with torch.no_grad():
                for p,tp in zip(src.parameters(), dst.parameters()): tp.data.mul_(1-self.tau).add_(self.tau*p.data)
        soft_update_cbf(self.cbf, self.cbf_t, self.tau); soft_update_pi(self.pi, self.pi_t, self.tau)

    def step_env(self, noise_std=0.2, shield_teacher=False):
        s, G = self.env.observe(); s=s.to(self.device); G=G.to(self.device)
        w, s_bar, s_all = self.build_state_all(s.unsqueeze(0), G.unsqueeze(0), self.attn_actor)
        a = self.actor(s_all).squeeze(0)
        a = torch.clamp(a + noise_std*torch.randn_like(a), -1.0, 1.0)
        if shield_teacher:
            # 用 \bar h 线性化构造 A,b 并做 QP 投影（不反传；执行期安全）
            x_all = torch.cat([s.unsqueeze(0), torch.bmm(self.attn_critic(s.unsqueeze(0),G.unsqueeze(0)), s.unsqueeze(0))], dim=-1).requires_grad_(True)
            h_bar = self.cbf_t(x_all)
            grad = torch.autograd.grad(h_bar.sum(), x_all, retain_graph=False)[0]
            gx, gy = grad[...,0], grad[...,1]; gvx, gvy = grad[...,2], grad[...,3]
            vx, vy = s[...,2], s[...,3]; Lf = gx*vx + gy*vy
            A = torch.stack([gvx, gvy], dim=-1).unsqueeze(-2); b = (-Lf - self.alpha*h_bar.squeeze(-1)).unsqueeze(-1)
            a = qp_teacher(a.unsqueeze(0), A, b).squeeze(0)
        s1, G1, r, done, info = self.env.step(a)
        self.buf.push(s, a, r, s1, done, G, G1); return r, done, info

    def update(self, updates=1):
        log={}
        for _ in range(updates):
            if len(self.buf)<self.batch: break
            s,a,r,s1,done,G,G1=self.buf.sample(self.batch)
            d=self.device; s=s.to(d); a=a.to(d); r=r.to(d).float(); s1=s1.to(d); done=done.to(d).float(); G=G.to(d); G1=G1.to(d)

            # 注意力与拼接
            w_c_t, sbar_t, sall_t = self.build_state_all(s, G, self.attn_critic)
            w_c_1, sbar_1, sall_1 = self.build_state_all(s1, G1, self.attn_critic_t)
            w_a_1, sbar_a1, sall_a1= self.build_state_all(s1, G1, self.attn_actor_t)
            a1 = self.actor_t(sall_a1)

            a_bar_t=torch.bmm(w_c_t, a); a_all_t=torch.cat([a, a_bar_t], dim=-1)
            a_bar_1=torch.bmm(w_c_1, a1); a_all_1=torch.cat([a1,a_bar_1], dim=-1)

            # 奖励 Q^r
            with torch.no_grad():
                y_r = r + self.gamma * self.critic_t(sall_1, a_all_1).squeeze(-1) * (1-done)
            q_r = self.critic(sall_t, a_all_t).squeeze(-1)
            loss_qr = ((y_r - q_r)**2).mean()
            self.opt_critic.zero_grad(); loss_qr.backward(); self.opt_critic.step()

            # ===== GCBF+：有限时域标注 + CBF/控制损失 =====
            mask_safe, mask_unsafe = label_batch(self.env, s, G, self.pi_t, self.attn_actor_t, self.horizon_T)
            x_t  = torch.cat([s,  sbar_t], dim=-1); x_1 = torch.cat([s1, sbar_1], dim=-1)
            h_t  = self.cbf(x_t)
            with torch.no_grad(): h_1_tgt = self.cbf_t(x_1)  # 停梯度/目标网络
            L_cbf, parts = cbf_hinge_losses(h_t, h_1_tgt, self.env.dt, alpha=self.alpha, gamma_margin=self.gamma_margin,
                                            mask_safe=mask_safe, mask_unsafe=mask_unsafe)
            self.opt_cbf.zero_grad(); L_cbf.backward(); self.opt_cbf.step()

            # QP 老师（用 \bar h；不反传）
            x_all = x_t.detach().requires_grad_(True)
            h_bar = self.cbf_t(x_all)
            grad = torch.autograd.grad(h_bar.sum(), x_all, retain_graph=False)[0]
            gx, gy = grad[...,0], grad[...,1]; gvx, gvy = grad[...,2], grad[...,3]
            vx, vy = s[...,2], s[...,3]; Lf = gx*vx + gy*vy
            A = torch.stack([gvx, gvy], dim=-1).unsqueeze(-2); b = (-Lf - self.alpha*h_bar.squeeze(-1)).unsqueeze(-1)
            with torch.no_grad(): a_qp = qp_teacher(self.pi(x_t), A, b)

            # 控制蒸馏（式22）
            L_ctrl = ctrl_distill_loss(self.pi(x_t), a_qp, eta_ctrl=self.eta_ctrl)
            self.opt_pi.zero_grad(); L_ctrl.backward(); self.opt_pi.step()

            # 成本 Q^c（即时成本：导数铰链正项）
            with torch.no_grad():
                c_t = torch.relu(self.gamma_margin - (h_1_tgt - h_t)/self.env.dt - self.alpha*h_t).squeeze(-1)
                y_c = c_t + self.gamma_c * self.cost_critic_t(sall_1, a_all_1).squeeze(-1) * (1-done)
            q_c = self.cost_critic(sall_t, a_all_t).squeeze(-1)
            loss_qc = ((y_c - q_c)**2).mean()
            self.opt_cost.zero_grad(); loss_qc.backward(); self.opt_cost.step()

            # RL Actor：最大化 Q^λ 并蒸馏到 QP
            w_a_t, sbar_a_t, sall_a_t = self.build_state_all(s, G, self.attn_actor)
            a_pi = self.actor(sall_a_t)
            a_pi_all = torch.cat([a_pi, torch.bmm(w_c_t, a_pi)], dim=-1)
            q_r_pi = self.critic(sall_t, a_pi_all).mean()
            q_c_pi = self.cost_critic(sall_t, a_pi_all).mean()
            actor_loss = -(q_r_pi - self.lam*q_c_pi) + self.eta_ctrl * F.mse_loss(a_pi, a_qp)
            self.opt_actor.zero_grad(); actor_loss.backward(); self.opt_actor.step()

            # 对偶更新
            with torch.no_grad():
                self.lam = torch.clamp(self.lam + 1e-3*(q_c_pi.detach() - self.cost_budget), 0.0, 10.0)

            # 软更新
            self.soft_update_all()

            log = dict(loss_qr=float(loss_qr.item()), loss_qc=float(loss_qc.item()),
                       L_cbf=float(L_cbf.item()), L_ctrl=float(L_ctrl.item()), lam=float(self.lam.item()))
        return log

    def train(self, total_steps=6000, log_every=200, shield=False):
        s, G = self.env.reset(); ep_ret, ep_len=0.0, 0
        pbar = tqdm(range(total_steps), ncols=100, desc="Training (Strict GAT-MF × GCBF+)")
        for t in pbar:
            r, done, info = self.step_env(noise_std=0.3 if t<1000 else 0.1, shield_teacher=shield)
            ep_ret += r; ep_len += 1
            if t>=self.update_after and t%self.update_every==0:
                log = self.update(self.update_every)
                if log and (t%log_every==0): pbar.set_postfix(ret=ep_ret/ep_len, lam=log["lam"], Lcbf=log["L_cbf"], Lctrl=log["L_ctrl"])
            if done or ep_len>=1000: s,G = self.env.reset(); ep_ret,ep_len=0.0,0

if __name__=="__main__":
    device="cuda" if torch.cuda.is_available() else "cpu"
    Trainer(device=device, n_agents=8).train(total_steps=3000, shield=False)
