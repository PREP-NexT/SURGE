"""Reservoir-enabled routing: add managed reservoirs/lakes to any scheme.

A reservoir cell replaces the kinematic-wave update with a storage-based release
(a Hanasaki-style rule: outflow tracks the long-term mean inflow, modulated by the
fill fraction). The release depends only on the START-of-substep storage, so within
a substep a reservoir cell is a fixed source -- it decouples its upstream from its
downstream, which is why adding reservoirs does not slow the Jacobi fixed point or
change its convergence (see paper Section 3.6). Storage is mass-balanced once per
substep from the converged inflow.

Usage:
    from surge import RoutingNetwork
    from surge.reservoir import ResRouter
    r = ResRouter(net, device='cuda', dtype=torch.float32)
    r.set_reservoirs(res_cell, cap_m3, rmean)          # from map_grand_dams.py
    q = r.jacobi_res(qold, q_lat, sweeps=3)            # or r.wavefront_res(...)
"""
import torch
from .kinematic_torch import TorchRouter


class ResRouter(TorchRouter):
    def set_reservoirs(self, res_cell, cap_m3, rmean, target=0.85, start_frac=0.6):
        """res_cell: int cell indices; cap_m3: capacity [m3]; rmean: mean release [m3/s]."""
        dev, dt = self.device, self.dtype
        self.res_cell = torch.as_tensor(res_cell, dtype=torch.int64, device=dev)
        self.cap = torch.as_tensor(cap_m3, dtype=dt, device=dev).clamp_min(1.0)
        self.rmean = torch.as_tensor(rmean, dtype=dt, device=dev).clamp_min(0.0)
        self.target = float(target)
        self.is_res = torch.zeros(self.n, dtype=torch.bool, device=dev)
        self.is_res[self.res_cell] = True
        self.reset_storage(start_frac)

    def reset_storage(self, frac=0.6):
        self.S = frac * self.cap.clone()

    def _release(self):
        krls = (self.S / (self.target * self.cap)).clamp(0.0, 1.2)
        return (krls * self.rmean).clamp_min(0.0)          # start-of-substep storage only

    def _Rfull(self, R):
        Rf = torch.zeros(self.n, dtype=self.dtype, device=self.device)
        Rf[self.res_cell] = R
        return Rf

    def _update_storage(self, qfield, q, R):
        qin = torch.zeros(self.n, dtype=self.dtype, device=self.device)
        qin.index_add_(-1, self.edge_dst, qfield.index_select(-1, self.edge_src))
        I = qin.index_select(-1, self.res_cell) + (q * self.dx).index_select(-1, self.res_cell)
        self.S = (self.S + (I - R) * self.dt).clamp(torch.zeros_like(self.cap), self.cap)

    def jacobi_res(self, qold, q, sweeps=3, n_newton=4):
        R = self._release(); Rf = self._Rfull(R)
        qk = qold.clamp_min(0.0)
        for _ in range(sweeps):
            qin = torch.zeros_like(qk)
            qin.index_add_(-1, self.edge_dst, qk.index_select(-1, self.edge_src))
            qk = self.newton(qin, qold, q, n_iter=n_newton)
            qk = torch.where(self.is_res, Rf, qk)
        self._update_storage(qk, q, R)
        return qk

    def wavefront_res(self, qold, q, n_newton=4):
        R = self._release(); Rf = self._Rfull(R)
        qnew = torch.zeros_like(qold + q)
        for l in range(self.net.n_levels):
            c0, c1 = self.level_ptr[l], self.level_ptr[l + 1]
            cells = self.cells_by_level[c0:c1]
            e0, e1 = self.edge_level_ptr[l], self.edge_level_ptr[l + 1]
            qin = torch.zeros_like(qnew)
            if e1 > e0:
                src, dst = self.edge_src_lv[e0:e1], self.edge_dst_lv[e0:e1]
                qin.index_add_(-1, dst, qnew.index_select(-1, src))
            sol = self.newton(qin[..., cells], qold[..., cells], q[..., cells],
                              alpha=self.alpha[..., cells], n_iter=n_newton, dx=self.dx[cells])
            qnew[..., cells] = torch.where(self.is_res[cells], Rf[cells], sol)
        self._update_storage(qnew, q, R)
        return qnew
