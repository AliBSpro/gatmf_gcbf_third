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

                from scipy import sparse

                # ---- 1) P 必须是稀疏 CSC，且 float64 ----

                # P = diag([1,...,1, rho])  (大小 m+1)

                P = sparse.diags([1.0] * m + [float(rho)], format="csc", dtype=np.float64)

                # ---- 2) q 必须是一维 float64 向量 ----

                ui64 = ui.astype(np.float64, copy=False)

                q = np.zeros((m + 1,), dtype=np.float64)

                q[:m] = -ui64

                # ---- 3) A 也必须是 CSC 稀疏矩阵（float64），l/u 是一维 float64 ----

                A1 = np.hstack([-Ai, -np.ones((K, 1), dtype=np.float64)])

                l1 = -np.inf * np.ones((K,), dtype=np.float64)

                u1 = (-bi_vec).flatten().astype(np.float64, copy=False)

                A2 = np.vstack([

                    np.hstack([np.eye(m, dtype=np.float64), np.zeros((m, 1), dtype=np.float64)]),

                    np.hstack([-np.eye(m, dtype=np.float64), np.zeros((m, 1), dtype=np.float64)]),

                    np.hstack([np.zeros((1, m), dtype=np.float64), np.array([[1.0]], dtype=np.float64)])

                ])

                l2 = np.hstack([np.full((m,), -1.0 * u_min, dtype=np.float64)])  # 注意：下面用上下界统一到 l/u

                u2 = np.hstack([np.full((m,), 1.0 * u_max, dtype=np.float64)])

                # 实际上我们在 OSQP 里是按 l<=A x<=u 的形式构造：

                # 上面 l2/u2 是给盒约束的两组；我们会和 A1 一起拼起来

                # 为了和前面的写法一致，你也可以继续用你原来的 l2/u2 构造，只要 dtype 和稀疏转化正确即可

                A_osqp = np.vstack([A1, A2]).astype(np.float64, copy=False)

                A_osqp = sparse.csc_matrix(A_osqp)

                l_osqp = np.hstack([l1, np.hstack([np.full((m,), u_min, dtype=np.float64),

                                                   np.full((m,), -u_max, dtype=np.float64),

                                                   np.array([0.0], dtype=np.float64)])])

                u_osqp = np.hstack([u1, np.hstack([np.full((m,), u_max, dtype=np.float64),

                                                   np.full((m,), -u_min, dtype=np.float64),

                                                   np.array([np.inf], dtype=np.float64)])])

                prob = osqp.OSQP()

                prob.setup(P=P, q=q, A=A_osqp, l=l_osqp, u=u_osqp,

                           verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=8000, polish=True)

                res = prob.solve()

                x = res.x if res.info.status_val in (1, 2) else np.hstack([ui64, 0.0])

                sol = x[:m]

            out[bi,i] = torch.from_numpy(sol).to(u_nom.device)
    return out
