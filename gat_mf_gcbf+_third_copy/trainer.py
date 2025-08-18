# -*- coding: utf-8 -*-
"""
严格 GAT-MF × GCBF+ 训练器（GPU 优先 + 模型落盘 + 指标CSV）
要点：
- 输出目录与 CSV：基于脚本同级目录创建 model/；启动即写表头
- 保存条件：每到 save_every 步或总步数末步，必保存（用 t+1 计数）
- 修复：update() 中 r/d 与 q 的维度广播不匹配（将 r/d 展开为 (B,1,1)）
- 修复：需要 autograd.grad 的段落使用 torch.enable_grad()（不能在 no_grad 中）
- 新增：--update_after / --update_every / --batch_size，支持短测与自适应采样
"""

import argparse, time, csv, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from grid_model import GridEnv
from grid_networks import Attention, Actor, Critic, CostCritic, STATE_DIM, ACT_DIM

# 兼容两种包布局：safety.* 或本地同级文件
try:
    from safety.cbf_net import CBFNet, soft_update as soft_update_cbf
    from safety.ctrl_net import CtrlNet, soft_update as soft_update_pi
    from safety.qp import qp_teacher
    from safety.labeler import label_batch
    from safety.losses import cbf_hinge_losses, ctrl_distill_loss
    from safety.replay import MaskedReplay
except Exception:
    from cbf_net import CBFNet, soft_update as soft_update_cbf
    from ctrl_net import CtrlNet, soft_update as soft_update_pi
    from qp import qp_teacher
    from labeler import label_batch
    from losses import cbf_hinge_losses, ctrl_distill_loss
    from replay import MaskedReplay


class Trainer:
    def __init__(self,
                 device: str = "cpu",
                 n_agents: int = 8,
                 # GCBF+ 关键超参
                 alpha: float = 1.0,
                 gamma_margin: float = 0.02,
                 eta_ctrl: float = 1.0,
                 rho: float = 200.0,
                 horizon_T: int = 16,
                 # 训练与保存
                 seed: int = 0,
                 run_tag: str = "",
                 save_every: int = 3000,
                 update_after: int = 1000,
                 update_every: int = 50,
                 batch_size: int = 64):

        self.device = device
        self.env = GridEnv(n_agents=n_agents, device=device)
        self.n = n_agents
        self.dt = self.env.dt

        # 随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 注意力与拼接
        self.attn_actor = Attention().to(device)
        self.attn_critic = Attention().to(device)
        self.attn_actor_t = Attention().to(device).eval()
        self.attn_critic_t = Attention().to(device).eval()

        # 策略与评论家
        self.actor = Actor().to(device)
        self.actor_t = Actor().to(device).eval()
        self.critic = Critic().to(device)
        self.critic_t = Critic().to(device).eval()
        self.cost_critic = CostCritic().to(device)
        self.cost_critic_t = CostCritic().to(device).eval()

        # CBF 与 控制（蒸馏到 QP）
        self.cbf = CBFNet(in_dim=STATE_DIM*2).to(device)
        self.cbf_t = CBFNet(in_dim=STATE_DIM*2).to(device).eval()
        for p in self.cbf_t.parameters():  # 冻结目标 CBF 的参数，只对 x_all 求导
            p.requires_grad_(False)
        self.pi  = CtrlNet(in_dim=STATE_DIM*2, out_dim=ACT_DIM).to(device)
        self.pi_t= CtrlNet(in_dim=STATE_DIM*2, out_dim=ACT_DIM).to(device).eval()

        # 目标网络初始化
        self._hard_copy_targets()

        # 优化器
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=3e-4)
        self.opt_cost = optim.Adam(self.cost_critic.parameters(), lr=3e-4)
        self.opt_attn_a = optim.Adam(self.attn_actor.parameters(), lr=3e-4)
        self.opt_attn_c = optim.Adam(self.attn_critic.parameters(), lr=3e-4)
        self.opt_cbf = optim.Adam(self.cbf.parameters(), lr=3e-4)
        self.opt_pi = optim.Adam(self.pi.parameters(), lr=3e-4)

        # 折扣与更新节奏
        self.gamma = 0.99
        self.gamma_c = 0.99
        self.tau = 0.01
        self.update_after = update_after
        self.update_every = update_every
        self.batch_size = batch_size

        # GCBF+ 超参
        self.alpha = alpha
        self.gamma_margin = gamma_margin
        self.eta_ctrl = eta_ctrl
        self.rho = rho
        self.horizon_T = horizon_T

        # 拉格朗日乘子
        self.lam = torch.tensor(0.0, device=device)
        self.eta_lam = 1e-3
        self.cost_budget = 0.0

        # 回放
        self.replay = MaskedReplay(capacity=200000)

        # === 输出目录与 CSV：绑定到脚本同级目录 ===
        self.project_root = Path(__file__).resolve().parent
        self.model_root = (self.project_root / "model")
        self.model_root.mkdir(parents=True, exist_ok=True)

        self.run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_seed{seed}" + (f"_{run_tag}" if run_tag else "")
        self.ckpt_prefix = self.model_root / f"run_{self.run_id}"
        self.metrics_csv = self.model_root / f"run_{self.run_id}_metrics.csv"

        # 启动即写 CSV 表头
        with open(self.metrics_csv, 'w', newline='') as f:
            csv.writer(f).writerow([
                "step","Lcbf","Lctrl","lam",
                "eval_return","eval_unsafe","eval_success_rate",
                "alpha","gamma_margin","eta_ctrl","rho","horizon_T","seed","tag"
            ])

        # 记录本次超参
        self.hparams = dict(alpha=self.alpha, gamma_margin=self.gamma_margin,
                            eta_ctrl=self.eta_ctrl, rho=self.rho, horizon_T=self.horizon_T,
                            seed=seed, run_tag=run_tag, n_agents=n_agents, batch_size=batch_size)

    def _hard_copy_targets(self):
        self.attn_actor_t.load_state_dict(self.attn_actor.state_dict())
        self.attn_critic_t.load_state_dict(self.attn_critic.state_dict())
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.cost_critic_t.load_state_dict(self.cost_critic.state_dict())
        self.cbf_t.load_state_dict(self.cbf.state_dict())
        self.pi_t.load_state_dict(self.pi.state_dict())

    @torch.no_grad()
    def _soft_update_all(self):
        for net, tgt in [(self.attn_actor, self.attn_actor_t),
                         (self.attn_critic, self.attn_critic_t),
                         (self.actor, self.actor_t),
                         (self.critic, self.critic_t),
                         (self.cost_critic, self.cost_critic_t)]:
            for p, tp in zip(net.parameters(), tgt.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
        soft_update_cbf(self.cbf, self.cbf_t, self.tau)
        soft_update_pi(self.pi, self.pi_t, self.tau)

    def build_state_all(self, s, G, attn_module):
        """ s: (B,N,STATE_DIM), G: (B,N,N) -> s_all: (B,N,2*STATE_DIM) """
        w = attn_module(s, G)                # (B,N,N)
        s_bar = torch.bmm(w, s)             # (B,N,STATE_DIM)
        s_all = torch.cat([s, s_bar], dim=-1)
        return w, s_bar, s_all

    # 注意：不加 @torch.no_grad。函数内部按需局部 no_grad/enable_grad
    def evaluate(self, episodes: int = 5, max_steps: int = 1000, shield: bool = False):
        eval_env = GridEnv(n_agents=self.env.n, device=self.device)
        returns, unsafe_accum, succ = [], [], []
        for _ in range(episodes):
            s, G = eval_env.reset()
            ep_ret, ep_steps, done = 0.0, 0, False
            while (not done) and ep_steps < max_steps:
                s = s.to(self.device); G = G.to(self.device)

                # 先无梯度走策略
                with torch.no_grad():
                    _, _, s_all = self.build_state_all(s.unsqueeze(0), G.unsqueeze(0), self.attn_actor_t)
                    a = self.actor_t(s_all).squeeze(0)

                if shield:
                    # 对 x_all 求导（局部开启梯度）
                    with torch.enable_grad():
                        _, _, s_all_g = self.build_state_all(s.unsqueeze(0), G.unsqueeze(0), self.attn_actor_t)
                        x_all = s_all_g.detach().requires_grad_(True)  # (1,N,2*STATE_DIM)
                        h_bar = self.cbf_t(x_all)
                        grad = torch.autograd.grad(h_bar.sum(), x_all, retain_graph=False)[0]
                        gx, gy, gvx, gvy = grad[...,0], grad[...,1], grad[...,2], grad[...,3]  # (1,N)
                        s_b = s.unsqueeze(0)  # (1,N,STATE_DIM)
                        vx, vy = s_b[...,2], s_b[...,3]  # (1,N)
                        Lf = gx*vx + gy*vy                 # (1,N)
                        A = torch.stack([gvx, gvy], dim=-1).unsqueeze(-2)  # (1,N,1,2)
                        b = (-Lf - self.alpha*h_bar.squeeze(-1)).unsqueeze(-1)  # (1,N,1)
                        a = qp_teacher(a.unsqueeze(0), A, b).squeeze(0)

                s, G, r, done, info = eval_env.step(a)
                ep_ret += r; ep_steps += 1
                unsafe_accum.append(float(info.get("unsafe_frac", 0.0)))
            returns.append(ep_ret); succ.append(1.0 if done else 0.0)
        return dict(eval_return=float(np.mean(returns)),
                    eval_unsafe=float(np.mean(unsafe_accum)),
                    eval_success_rate=float(np.mean(succ)))

    def save_checkpoint(self, step: int, log: dict, eval_metrics: dict):
        self.model_root.mkdir(parents=True, exist_ok=True)

        state = {
            "step": step,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "cost_critic": self.cost_critic.state_dict(),
            "attn_actor": self.attn_actor.state_dict(),
            "attn_critic": self.attn_critic.state_dict(),
            "cbf": self.cbf.state_dict(),
            "pi": self.pi.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "opt_cost": self.opt_cost.state_dict(),
            "opt_attn_a": self.opt_attn_a.state_dict(),
            "opt_attn_c": self.opt_attn_c.state_dict(),
            "opt_cbf": self.opt_cbf.state_dict(),
            "opt_pi": self.opt_pi.state_dict(),
            "lam": float(self.lam.item()),
            "log": log, "eval": eval_metrics, "hparams": self.hparams
        }
        ckpt_path = f"{self.ckpt_prefix}_step{step}.pt"
        torch.save(state, ckpt_path)

        with open(self.metrics_csv, 'a', newline='') as f:
            csv.writer(f).writerow([
                step,
                f"{log.get('L_cbf', float('nan')):.6f}",
                f"{log.get('L_ctrl', float('nan')):.6f}",
                f"{log.get('lam', float('nan')):.6f}",
                f"{eval_metrics.get('eval_return', float('nan')):.6f}",
                f"{eval_metrics.get('eval_unsafe', float('nan')):.6f}",
                f"{eval_metrics.get('eval_success_rate', float('nan')):.6f}",
                self.hparams["alpha"], self.hparams["gamma_margin"], self.hparams["eta_ctrl"],
                self.hparams["rho"], self.hparams["horizon_T"],
                self.hparams["seed"], self.hparams["run_tag"],
            ])

        print(f"[SAVE] ckpt -> {ckpt_path}")
        print(f"[SAVE] metrics -> {self.metrics_csv}")

    def step_env(self, noise_std=0.3, shield_teacher=False):
        """单步交互 + 入回放；注意这里使用 env 的观测接口（与你项目一致即可）"""
        with torch.no_grad():
            s_cur, G_cur = self.env.observe()   # 若无 observe()，按你的接口调整
            s = torch.as_tensor(s_cur, device=self.device, dtype=torch.float32)   # (N,STATE_DIM)
            G = torch.as_tensor(G_cur, device=self.device, dtype=torch.float32)   # (N,N)

            _, _, s_all = self.build_state_all(s.unsqueeze(0), G.unsqueeze(0), self.attn_actor)
            a = self.actor(s_all).squeeze(0)  # (N,2)
            if noise_std > 0:
                a = torch.clamp(a + torch.randn_like(a) * noise_std, -1.0, 1.0)

            if shield_teacher:
                # 部署期盾牌：对 x_all 求导
                with torch.enable_grad():
                    _, _, s_all_g = self.build_state_all(s.unsqueeze(0), G.unsqueeze(0), self.attn_actor_t)
                    x_all = s_all_g.detach().requires_grad_(True)  # (1,N,2*STATE_DIM)
                    h_bar = self.cbf_t(x_all)
                    grad = torch.autograd.grad(h_bar.sum(), x_all, retain_graph=False)[0]
                    gx, gy, gvx, gvy = grad[...,0], grad[...,1], grad[...,2], grad[...,3]  # (1,N)
                    s_b = s.unsqueeze(0)
                    vx, vy = s_b[...,2], s_b[...,3]
                    Lf = gx*vx + gy*vy
                    A = torch.stack([gvx, gvy], dim=-1).unsqueeze(-2)  # (1,N,1,2)
                    b = (-Lf - self.alpha*h_bar.squeeze(-1)).unsqueeze(-1)  # (1,N,1)
                    a = qp_teacher(a.unsqueeze(0), A, b).squeeze(0)

            s1, G1, r, done, info = self.env.step(a)
            self.replay.push(
                s, a, torch.as_tensor(r, dtype=torch.float32),
                torch.as_tensor(s1, dtype=torch.float32),
                torch.as_tensor(done, dtype=torch.float32),
                G, torch.as_tensor(G1, dtype=torch.float32)
            )
            return float(r), bool(done), info

    def update(self, K: int):
        # 样本不足先返回
        if len(self.replay) < self.update_after:
            return None

        # 自适应 batch：不足 batch_size 就用当前长度
        B = min(self.batch_size, len(self.replay))

        # 采样
        s, a, r, s1, d, G, G1 = self.replay.sample(batch=B)
        s = s.to(self.device);
        a = a.to(self.device);
        r = r.to(self.device)
        s1 = s1.to(self.device);
        d = d.to(self.device);
        G = G.to(self.device);
        G1 = G1.to(self.device)

        # 维度
        B_eff, N, _ = s.shape
        r_exp = r.view(B_eff, 1, 1)
        d_exp = d.view(B_eff, 1, 1)

        # =========================
        # 1) 奖励评论家 Q^r
        # =========================
        with torch.no_grad():
            # 目标值部分（全 no_grad）
            _, _, s1_all_t = self.build_state_all(s1, G1, self.attn_critic_t)
            a1_t = self.actor_t(self.build_state_all(s1, G1, self.attn_actor_t)[2])
            att1t = self.attn_critic_t(s1, G1)
            q1_t = self.critic_t(s1_all_t, torch.cat([a1_t, torch.bmm(att1t, a1_t)], dim=-1))
            y_r = r_exp + (1.0 - d_exp) * self.gamma * q1_t

            # 当前步的特征（与下游 backward 解耦）
            _, _, s_all_qr = self.build_state_all(s, G, self.attn_critic)
            att_qr = self.attn_critic(s, G)

        q_r = self.critic(s_all_qr.detach(), torch.cat([a, torch.bmm(att_qr.detach(), a)], dim=-1))
        loss_qr = F.mse_loss(q_r, y_r)
        self.opt_critic.zero_grad();
        loss_qr.backward();
        self.opt_critic.step()

        # =========================
        # 2) 成本评论家 Q^c
        # =========================
        with torch.no_grad():
            mask_safe, mask_unsafe = label_batch(self.env, s, G, self.actor_t, self.attn_actor_t, self.horizon_T)

            _, _, s1_all_tc = self.build_state_all(s1, G1, self.attn_critic_t)
            a1_t = self.actor_t(self.build_state_all(s1, G1, self.attn_actor_t)[2])
            att1t = self.attn_critic_t(s1, G1)
            q1c_t = self.cost_critic_t(s1_all_tc, torch.cat([a1_t, torch.bmm(att1t, a1_t)], dim=-1))

            y_c = mask_unsafe.float() + (1.0 - d_exp) * self.gamma_c * q1c_t

            _, _, s_all_qc = self.build_state_all(s, G, self.attn_critic)
            att_qc = self.attn_critic(s, G)

        q_c = self.cost_critic(s_all_qc.detach(), torch.cat([a, torch.bmm(att_qc.detach(), a)], dim=-1))
        loss_qc = F.mse_loss(q_c, y_c)
        self.opt_cost.zero_grad();
        loss_qc.backward();
        self.opt_cost.step()

        # =========================
        # 3) CBF 损失
        # =========================
        with torch.no_grad():
            _, _, s_all_cbf = self.build_state_all(s, G, self.attn_critic)
            _, _, s1_all_cbf = self.build_state_all(s1, G1, self.attn_critic_t)

        h_t = self.cbf(s_all_cbf.detach())
        h_t1 = self.cbf_t(s1_all_cbf.detach())
        L_cbf, _ = cbf_hinge_losses(h_t, h_t1, dt=self.dt, alpha=self.alpha,
                                    gamma_margin=self.gamma_margin,
                                    mask_safe=mask_safe, mask_unsafe=mask_unsafe)
        self.opt_cbf.zero_grad();
        L_cbf.backward();
        self.opt_cbf.step()

        # =========================
        # 4) 控制蒸馏：π 贴近 QP 教师
        # =========================
        with torch.no_grad():
            _, _, s_all_pi = self.build_state_all(s, G, self.attn_critic)

        with torch.enable_grad():
            x_all = s_all_pi.detach().requires_grad_(True)  # 只对 x_all 求导
            h_bar = self.cbf_t(x_all)
            grad = torch.autograd.grad(h_bar.sum(), x_all, retain_graph=False)[0]
            gx, gy, gvx, gvy = grad[..., 0], grad[..., 1], grad[..., 2], grad[..., 3]
            vx, vy = s[..., 2], s[..., 3]
            Lf = gx * vx + gy * vy
            A = torch.stack([gvx, gvy], dim=-1).unsqueeze(-2)
            b = (-Lf - self.alpha * h_bar.squeeze(-1)).unsqueeze(-1)
            a_nom = self.pi_t(s_all_pi.detach())
            a_qp = qp_teacher(a_nom, A, b).detach()

        a_pi = self.pi(s_all_pi.detach())
        L_ctrl = ctrl_distill_loss(a_pi, a_qp, eta_ctrl=self.eta_ctrl)
        self.opt_pi.zero_grad();
        L_ctrl.backward();
        self.opt_pi.step()

        # =========================
        # 5) Actor（拉格朗日）：-Q^r + λ Q^c
        # =========================
        # 允许通过 attn_actor 回传梯度
        _, _, s_all_actor = self.build_state_all(s, G, self.attn_actor)
        with torch.no_grad():
            att_for_actor = self.attn_critic(s, G)

        a_actor = self.actor(s_all_actor)
        q_r_pi = self.critic(s_all_actor, torch.cat([a_actor, torch.bmm(att_for_actor.detach(), a_actor)], dim=-1))
        q_c_pi = self.cost_critic(s_all_actor, torch.cat([a_actor, torch.bmm(att_for_actor.detach(), a_actor)], dim=-1))
        loss_actor = (-q_r_pi + self.lam * q_c_pi).mean()
        self.opt_actor.zero_grad();
        self.opt_attn_a.zero_grad()
        loss_actor.backward()
        self.opt_actor.step();
        self.opt_attn_a.step()

        # 拉格朗日乘子
        with torch.no_grad():
            self.lam.clamp_(0.0, 10.0)
            self.lam += self.eta_lam * (q_c_pi.mean() - self.cost_budget)
            self.lam.clamp_(0.0, 10.0)

        # 目标网络软更新
        self._soft_update_all()

        return dict(
            L_cbf=float(L_cbf.item()),
            L_ctrl=float(L_ctrl.item()),
            loss_qr=float(loss_qr.item()),
            loss_qc=float(loss_qc.item()),
            lam=float(self.lam.item())
        )

    def train(self, total_steps=30000, save_every=3000, shield=False):
        # 训练主循环
        s, G = self.env.reset()
        ep_ret, ep_len = 0.0, 0
        log = None  # 确保首次保存时有兜底
        pbar = tqdm(range(total_steps), ncols=120, desc="Training (Strict GAT-MF × GCBF+)")
        for t in pbar:
            r, done, info = self.step_env(noise_std=0.3 if t < 1000 else 0.1, shield_teacher=shield)
            ep_ret += r; ep_len += 1

            if t >= self.update_after and t % self.update_every == 0:
                log = self.update(self.update_every)
                if log:
                    pbar.set_postfix(ret=f"{ep_ret/ep_len:.3f}",
                                     lam=f"{log['lam']:.3f}",
                                     Lcbf=f"{log['L_cbf']:.3f}",
                                     Lctrl=f"{log['L_ctrl']:.3f}")

            # 保存条件：每 save_every 步或最后一步必保存
            need_save = ((t + 1) % save_every == 0) or (t + 1 == total_steps)
            if need_save:
                if log is None:
                    log = dict(L_cbf=float('nan'), L_ctrl=float('nan'), lam=float(self.lam.item()))
                eval_metrics = self.evaluate(episodes=5, shield=False)
                self.model_root.mkdir(parents=True, exist_ok=True)
                self.save_checkpoint(step=t+1, log=log, eval_metrics=eval_metrics)

            if done or ep_len >= 1000:
                s, G = self.env.reset(); ep_ret, ep_len = 0.0, 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total_steps", type=int, default=30000)
    p.add_argument("--save_every", type=int, default=3000)
    p.add_argument("--shield", action="store_true")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--gamma_margin", type=float, default=0.02)
    p.add_argument("--eta_ctrl", type=float, default=1.0)
    p.add_argument("--rho", type=float, default=200.0)
    p.add_argument("--horizon_T", type=int, default=16)
    p.add_argument("--run_tag", type=str, default="")
    p.add_argument("--n_agents", type=int, default=8)
    # 便于短测
    p.add_argument("--update_after", type=int, default=1000)
    p.add_argument("--update_every", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = Trainer(device=device, n_agents=args.n_agents,
                 alpha=args.alpha, gamma_margin=args.gamma_margin,
                 eta_ctrl=args.eta_ctrl, rho=args.rho, horizon_T=args.horizon_T,
                 run_tag=args.run_tag, seed=args.seed,
                 save_every=args.save_every,
                 update_after=args.update_after, update_every=args.update_every,
                 batch_size=args.batch_size)
    tr.train(total_steps=args.total_steps, save_every=args.save_every, shield=args.shield)
