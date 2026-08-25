"""PyTorch (GPU / batched / differentiable) kinematic-wave routing for CWatM.

Two parallel schemes replacing the serial topological sweep in t5.cpp:

* wavefront : exact — processes topological levels in order, all cells in a
  level in parallel. Bit-comparable to the serial code (modulo fp sum order).
* jacobi    : approximate — k fully-parallel sweeps over ALL cells; sweep k
  propagates upstream information k levels. Since the kinematic wave moves
  only ~1 cell per routing substep (Courant ~ O(1)), a few sweeps suffice.

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
    def substep_wavefront(self, qold, q, alpha=None, n_newton=10):
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

    def substep_jacobi(self, qold, q, alpha=None, sweeps=3, n_newton=6):
        """k fully-parallel Jacobi sweeps. qold, q: (N,) or (B, N)."""
        qk = qold.clamp_min(0.0)
        for _ in range(sweeps):
            qin = torch.zeros_like(qk)
            qin.index_add_(-1, self.edge_dst, qk.index_select(-1, self.edge_src))
            qk = self.newton(qin, qold, q, alpha=alpha, n_iter=n_newton)
        return qk

    # ------------------------------------------------- blocked Gauss-Seidel
    def _build_blocks(self, n_blocks):
        """Partition topological levels into contiguous ranges (cached)."""
        key = int(n_blocks)
        if not hasattr(self, '_blocks'):
            self._blocks = {}
        if key in self._blocks:
            return self._blocks[key]
        L = self.net.n_levels
        bounds = np.linspace(0, L, n_blocks + 1).astype(int)
        blocks = []
        lv_dst_sorted = None
        for b in range(n_blocks):
            l0, l1 = bounds[b], bounds[b + 1]
            if l1 <= l0:
                continue
            c0, c1 = self.level_ptr[l0], self.level_ptr[l1]
            cells = self.cells_by_level[c0:c1]
            e0, e1 = self.edge_level_ptr[l0], self.edge_level_ptr[l1]
            # edges whose dst lies in this block (src may be anywhere upstream)
            src, dst = self.edge_src_lv[e0:e1], self.edge_dst_lv[e0:e1]
            blocks.append((cells, src, dst))
        self._blocks[key] = blocks
        return blocks

    def substep_blocked_gs(self, qold, q, alpha=None, n_blocks=8, sweeps=3,
                           n_newton=4):
        """Blocked Gauss-Seidel over level ranges, Jacobi inside each block.

        Blocks are processed upstream-to-downstream, so within-substep
        information propagates the full network depth at the FLOP cost of
        `sweeps` global Jacobi sweeps; truncation error is confined to
        within-block coupling deeper than `sweeps` levels.
        """
        blocks = self._build_blocks(n_blocks)
        a_full = self.alpha if alpha is None else alpha
        qk = (qold + q * 0).clamp_min(0.0)   # broadcast to common shape
        for cells, src, dst in blocks:
            a_b = a_full[..., cells] if a_full.shape[-1] == self.n else a_full
            for _ in range(sweeps):
                qin = torch.zeros_like(qk)
                qin.index_add_(-1, dst, qk.index_select(-1, src))
                qk = qk.clone()
                qk[..., cells] = self.newton(
                    qin[..., cells], qold[..., cells], q[..., cells],
                    alpha=a_b, n_iter=n_newton, dx=self.dx[cells])
        return qk

    # ------------------------------------------------------- focused sweeps
    def _build_deep(self, thresh):
        key = int(thresh)
        if not hasattr(self, '_deep'):
            self._deep = {}
        if key not in self._deep:
            lv = self.net.level
            deep = np.nonzero(lv > thresh)[0]
            sel = lv[self.net.edge_dst] > thresh
            t = lambda a: torch.as_tensor(a, dtype=torch.int64, device=self.device)
            self._deep[key] = (t(deep), t(self.net.edge_src[sel]),
                               t(self.net.edge_dst[sel]))
        return self._deep[key]

    def substep_focused(self, qold, q, alpha=None, sweeps=3, deep_sweeps=5,
                        n_newton=4, deep_thresh=None):
        """Jacobi sweeps everywhere, then extra sweeps on deep cells only.

        Cells with topological level <= `sweeps` are exact after the global
        sweeps; the extra sweeps push the truncation depth of the remaining
        (deep) cells to sweeps+deep_sweeps at ~quarter of the per-sweep cost
        (deep cells are ~27% of the domain for thresh=3).
        """
        thresh = sweeps if deep_thresh is None else deep_thresh
        deep, esrc_d, edst_d = self._build_deep(thresh)
        a_full = self.alpha if alpha is None else alpha
        a_d = a_full[..., deep] if a_full.shape[-1] == self.n else a_full
        qk = self.substep_jacobi(qold, q, alpha=alpha, sweeps=sweeps,
                                 n_newton=n_newton)
        for _ in range(deep_sweeps):
            qin = torch.zeros_like(qk)
            qin.index_add_(-1, edst_d, qk.index_select(-1, esrc_d))
            qk = qk.clone()
            qk[..., deep] = self.newton(qin[..., deep], qold[..., deep],
                                        q[..., deep], alpha=a_d,
                                        n_iter=n_newton, dx=self.dx[deep])
        return qk

    # ----------------------------------------------------------------- runs
    def run(self, q0, sideflow, scheme='jacobi', checkpoint_every=0, **kw):
        """Route a (T_substeps, N) or (T, B, N) sideflow series. Returns (T, ..., N) Q.

        checkpoint_every > 0 wraps blocks of substeps in torch.utils.checkpoint
        (recompute-on-backward) so long differentiable runs fit in VRAM.
        """
        step = {'jacobi': self.substep_jacobi, 'wavefront': self.substep_wavefront,
                'blocked_gs': self.substep_blocked_gs,
                'focused': self.substep_focused}[scheme]
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
