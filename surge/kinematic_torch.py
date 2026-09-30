"""PyTorch (GPU / batched / differentiable) kinematic-wave routing for CWatM.

Two schemes replace the serial upstream-to-downstream traversal of t5.cpp. The
paper calls them the partly parallel and the fully parallel scheme:

* ordered  : takes the topological levels in turn and computes every cell of a
  level at once, so it returns the serial solution (modulo fp sum order).
* parallel : takes k iterations over every cell at once, and iteration k
  carries upstream information k levels down. The iterations a network needs
  follow its Courant number, two on a 30 arc-minute network and thirty on a
  1 arc-minute one.

Both share the same vectorised Newton solver (IterateToQnew in t5.cpp).
All ops are differentiable -> autograd gives dQ/d(alpha, sideflow, ...).
"""
import numpy as np
import torch


class TorchRouter:
    def __init__(self, net, device='cuda', dtype=torch.float64):
        self.device = torch.device(device)
        self.dtype = dtype
        self.net = net
        t = lambda a, dt=dtype: torch.as_tensor(a, dtype=dt, device=self.device)
        self.alpha = t(net.channelAlpha)
        self.beta = float(net.beta)
        self.dt = float(net.dtRouting)
        self.dx = t(net.chanLength)
        self.n = net.n_cells
        self.edge_src = t(net.edge_src, torch.int64)
        self.edge_dst = t(net.edge_dst, torch.int64)
        # per-level schedule (edges sorted by level of dst cell)
        lv_dst = net.level[net.edge_dst]
        order = np.argsort(lv_dst, kind='stable')
        self.edge_src_lv = t(net.edge_src[order], torch.int64)
        self.edge_dst_lv = t(net.edge_dst[order], torch.int64)
        self.edge_level_ptr = np.searchsorted(lv_dst[order], np.arange(net.n_levels + 1))
        self.cells_by_level = t(net.dirDown_by_level, torch.int64)
        self.level_ptr = net.level_ptr

    # ---------------------------------------------------------------- newton
    # Discharge floor [m3/s]. t5.cpp uses 1e-30, but pow(x, beta-1) backward
    # blows up float32 gradients below ~1e-6; 1e-6 m3/s is physically zero.
    QFLOOR = 1e-6

    def newton(self, qin, qold, q, alpha=None, n_iter=10, dx=None):
        """Vectorised IterateToQnew (t5.cpp). Shapes broadcast over (B?, N)."""
        a = self.alpha if alpha is None else alpha
        beta, dt = self.beta, self.dt
        dtx = dt / (self.dx if dx is None else dx)
        qold_ = qold.clamp_min(0.0)
        ab_pQ = a * beta * ((qold_ + qin) / 2).clamp_min(self.QFLOOR) ** (beta - 1)
        C = dtx * qin + a * qold_.clamp_min(self.QFLOOR) ** beta + dt * q
        qk = (dtx * qin + qold_ * ab_pQ + dt * q) / (dtx + ab_pQ)
        qk = qk.clamp_min(self.QFLOOR)
        for _ in range(n_iter):
            f = dtx * qk + a * qk ** beta - C
            df = dtx + a * beta * qk ** (beta - 1)
            qk = (qk - f / df).clamp_min(self.QFLOOR)
        zero = (qin + qold_ + q) == 0
        return torch.where(zero, torch.zeros_like(qk), qk.clamp_min(0.0))

    # -------------------------------------------------------------- schemes
    def substep_ordered(self, qold, q, alpha=None, n_newton=10):
        """Exact level-scheduled substep. qold, q: (N,) or (B, N)."""
        qnew = torch.zeros_like(qold + q)
        for l in range(self.net.n_levels):
            c0, c1 = self.level_ptr[l], self.level_ptr[l + 1]
            cells = self.cells_by_level[c0:c1]
            e0, e1 = self.edge_level_ptr[l], self.edge_level_ptr[l + 1]
            qin = torch.zeros_like(qnew)
            if e1 > e0:
                src, dst = self.edge_src_lv[e0:e1], self.edge_dst_lv[e0:e1]
                qin.index_add_(-1, dst, qnew.index_select(-1, src))
            a_full = self.alpha if alpha is None else alpha
            qnew[..., cells] = self.newton(
                qin[..., cells], qold[..., cells], q[..., cells],
                alpha=a_full[..., cells], n_iter=n_newton, dx=self.dx[cells])
        return qnew

    def substep_parallel(self, qold, q, alpha=None, iterations=3, n_newton=6):
        """k iterations over every cell at once. qold, q: (N,) or (B, N)."""
        qk = qold.clamp_min(0.0)
        for _ in range(iterations):
            qin = torch.zeros_like(qk)
            qin.index_add_(-1, self.edge_dst, qk.index_select(-1, self.edge_src))
            qk = self.newton(qin, qold, q, alpha=alpha, n_iter=n_newton)
        return qk

    # ------------------------------------------------------------------ run
    def run(self, q0, sideflow, scheme='parallel', checkpoint_every=0, **kw):
        """Route a (T_substeps, N) or (T, B, N) sideflow series. Returns (T, ..., N) Q.

        checkpoint_every > 0 wraps blocks of substeps in torch.utils.checkpoint
        (recompute-on-backward) so long differentiable runs fit in VRAM.
        """
        step = {'parallel': self.substep_parallel, 'ordered': self.substep_ordered}[scheme]
        T = sideflow.shape[0]
        q = q0
        out = []
        if checkpoint_every > 0:
            from torch.utils.checkpoint import checkpoint

            def block(q, sf_blk, *params):
                res = []
                for t in range(sf_blk.shape[0]):
                    q = step(q, sf_blk[t], **kw)
                    res.append(q)
                return torch.stack(res)

            for t0 in range(0, T, checkpoint_every):
                blk = checkpoint(block, q, sideflow[t0:t0 + checkpoint_every],
                                 *(v for v in kw.values() if torch.is_tensor(v)),
                                 use_reentrant=False)
                out.append(blk)
                q = blk[-1]
            return torch.cat(out)
        for t in range(T):
            q = step(q, sideflow[t], **kw)
            out.append(q)
        return torch.stack(out)
