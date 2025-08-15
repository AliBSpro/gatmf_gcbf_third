# -*- coding: utf-8 -*-
"""
严格版 CBF-QP 教师（式(17)）：
min_{u,ζ}  0.5||u - u_nom||^2 + 0.5*ρ ζ^2
s.t.       A u >= b - ζ,   u_min <= u <= u_max,   ζ >= 0
实现：优先 ProxQP（proxsuite），否则 OSQP；均为“完整 QP 解算”，训练中不反传。
"""
import torch, numpy as np

_BACKEND = None
try:
    # ProxQP（更贴近原论文使用的 ProxQP/JaxProxQP 思路）
    from proxsuite import proxqp as _proxqp
    _BACKEND = "proxqp"
except Exception:
    try:
        import osqp as _osqp
        _BACKEND = "osqp"
    except Exception as e:
        raise ImportError("未检测到可用 QP 求解器。请安装 proxsuite 或 osqp 之一。") from e

@torch.no_grad()
def qp_teacher(u_nom, A, b, u_min=-1.0, u_max=1.0, rho=200.0):
    """
    u_nom: (B,N,m)
    A:     (B,N,K,m) 或 (B,N,m) -> 自动升维为 K=1
    b:     (B,N,K,1) 或 (B,N,1) -> 自动升维为 K=1
    返回:   u_qp: (B,N,m)
    """
    if A.dim()==3: A = A.unsqueeze(-2)
    if b.dim()==3: b = b.unsqueeze(-2)
    B,N,K,m = A.shape
    out = torch.empty_like(u_nom)

    for bi in range(B):
        for i in range(N):
            ui = u_nom[bi,i].cpu().numpy()
            Ai = A[bi,i].cpu().numpy().reshape(K,m)
            bi_vec = b[bi,i].cpu().numpy().reshape(K,1)

            if _BACKEND == "proxqp":
                # 变量 x = [u; ζ] ∈ R^{m+1}
                n = m+1; n_in = K + 2*m + 1; n_eq = 0
                H = np.eye(n); H[-1,-1] = rho
                g = np.zeros(n); g[:m] = -ui
                # 不等式矩阵 G x <= h 形式：
                # 1) -A u - ζ*1 <= -b
                G1 = np.hstack([-Ai, -np.ones((K,1))]); h1 = (-bi_vec).flatten()
                # 2) 盒约束 u_min <= u <= u_max  -> 上下界合并写成不等式
                G2 = np.vstack([ np.hstack([ np.eye(m), np.zeros((m,1)) ]),
                                 np.hstack([-np.eye(m), np.zeros((m,1)) ]) ])
                h2 = np.hstack([ np.full((m,), u_max), np.full((m,), -u_min) ])
                # 3) ζ >= 0  ->  -ζ <= 0
                G3 = np.hstack([ np.zeros((1,m)), -np.ones((1,1)) ]); h3 = np.array([0.0])
                G = np.vstack([G1, G2, G3]); h = np.hstack([h1, h2, h3])

                solver = _proxqp.dense.QP(n, n_eq, n_in)
                solver.settings.eps_abs = 1e-5
                solver.settings.eps_rel = 1e-5
                solver.settings.verbose = False
                solver.init(H, g, None, None, G, None, h)
                solver.solve()
                x = solver.results.x
                sol = x[:m]

            elif _BACKEND == "osqp":
                import osqp as osqp
                P = np.eye(m+1); P[-1,-1] = rho
                q = np.zeros((m+1,)); q[:m] = -ui
                # OSQP 采用 l <= A x <= u
                A1 = np.hstack([-Ai, -np.ones((K,1))]); l1 = -np.inf*np.ones((K,)); u1 = (-bi_vec).flatten()
                A2 = np.vstack([np.hstack([ np.eye(m), np.zeros((m,1)) ]),
                                np.hstack([-np.eye(m), np.zeros((m,1)) ]),
                                np.hstack([ np.zeros((1,m)), np.array([[1.0]]) ])])
                l2 = np.hstack([ np.full((m,), u_min), np.full((m,), -u_max), 0.0 ])
                u2 = np.hstack([ np.full((m,), u_max), np.full((m,), -u_min), np.inf ])
                A_osqp = np.vstack([A1, A2]); l_osqp = np.hstack([l1, l2]); u_osqp = np.hstack([u1, u2])
                prob = osqp.OSQP()
                prob.setup(P=P, q=q, A=A_osqp, l=l_osqp, u=u_osqp, verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=8000)
                res = prob.solve()
                x = res.x if res.info.status_val in (1,2) else np.hstack([ui,0.0])
                sol = x[:m]
            out[bi,i] = torch.from_numpy(sol).to(u_nom.device)
    return out
