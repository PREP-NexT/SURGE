"""Reservoir-enabled routing: add managed reservoirs/lakes to any scheme.

A reservoir cell replaces the kinematic-wave update with a storage-based release
(a Hanasaki-style rule: outflow tracks the long-term mean inflow, modulated by the
fill fraction). The release depends on the storage at the start of the time step and
on the inflow of the same step, so a reservoir cell is not a fixed source: at every
iteration of the parallel scheme its release is recomputed from the inflow the cells
above it deliver, and it then passes that release downstream like any other cell
(paper Section 2.4). Its response to a change in inflow is bounded, so the reservoirs
do not slow the convergence (paper Section 4.3). Storage is mass-balanced once per
time step from the converged inflow.

Usage:
    from surge import RoutingNetwork
    from surge.reservoir import ResRouter
    r = ResRouter(net, device='cuda', dtype=torch.float32)
    r.set_reservoirs(res_cell, cap_m3, rmean)          # from map_grand_dams.py
    q = r.parallel_res(qold, q_lat, iterations=3)            # or r.ordered_res(...)
"""
import numpy as np
import torch
from .kinematic_torch import TorchRouter


class ResRouter(TorchRouter):
    def set_reservoirs(self, res_cell, cap_m3, rmean, target=0.85, start_frac=0.6):
        """res_cell: int cell indices; cap_m3: capacity [m3]; rmean: mean release [m3/s]."""
        dev, dt = self.device, self.dtype
        self.res_cell = torch.as_tensor(res_cell, dtype=torch.int64, device=dev)
        # the storage of a reservoir is held in double precision: capacities span 1 to 1e11 m3
        # and a step moves up to 1e9 m3 through them, which single precision cannot resolve
        self.cap = torch.as_tensor(cap_m3, dtype=torch.float64, device=dev).clamp_min(1.0)
        self.rmean = torch.as_tensor(rmean, dtype=torch.float64, device=dev).clamp_min(0.0)
        self.target = float(target)
        self.is_res = torch.zeros(self.n, dtype=torch.bool, device=dev)
        self.is_res[self.res_cell] = True
        self.res_pos = torch.full((self.n,), -1, dtype=torch.int64, device=dev)
        self.res_pos[self.res_cell] = torch.arange(self.res_cell.numel(), device=dev)
        self._level_res = None
        if self.res_cell.numel():
            self._build_level_res()     # which reservoirs sit in each level, once, off the clock
        self.reset_storage(start_frac)

    def reset_storage(self, frac=0.6):
        self.S = frac * self.cap.clone()          # float64, like the capacity

    def _target_release(self):
        """the rule release, from the start-of-step storage alone"""
        krls = (self.S / (self.target * self.cap)).clamp_min(0.0)
        return (krls * self.rmean).clamp_min(0.0)

    def _inflow(self, qfield, q):
        """total inflow to each reservoir cell: upstream outflows plus its own lateral inflow"""
        qin = torch.zeros_like(qfield)
        qin.index_add_(-1, self.edge_dst, qfield.index_select(-1, self.edge_src))
        return qin.index_select(-1, self.res_cell) + (q * self.dx).index_select(-1, self.res_cell)

    def _release_at(self, idx, I):
        """the clipped release for the reservoirs named by idx"""
        I64 = I.to(torch.float64)
        S, cap = self.S[idx], self.cap[idx]
        krls = (S / (self.target * cap)).clamp_min(0.0)
        Rhat = (krls * self.rmean[idx]).clamp_min(0.0)
        Rmax = S / self.dt + I64
        Rmin = ((S + I64 * self.dt - cap) / self.dt).clamp_min(0.0)
        return torch.minimum(torch.maximum(Rhat, Rmin), Rmax).to(self.dtype)

    def _release(self, I):
        """the rule release clipped to what the storage bounds allow over the step:
        it cannot exceed the water available, and it must pass on whatever would overfill."""
        I64 = I.to(torch.float64)
        Rmax = self.S / self.dt + I64                                          # all there is
        Rmin = ((self.S + I64 * self.dt - self.cap) / self.dt).clamp_min(0.0)  # forced spill
        R = torch.minimum(torch.maximum(self._target_release(), Rmin), Rmax)
        return R.to(self.dtype)

    def _Rfull(self, R):
        Rf = torch.zeros(self.n, dtype=self.dtype, device=self.device)
        Rf[self.res_cell] = R
        return Rf

    def _update_storage(self, I, R):
        """mass balance, which stays inside [0, cap] because the release was clipped"""
        self.S = self.S + (I.to(torch.float64) - R.to(torch.float64)) * self.dt

    def parallel_res(self, qold, q, iterations=3, n_newton=4):
        qk = qold.clamp_min(0.0)
        if self.res_cell.numel() == 0:                 # nothing to do beyond the plain step
            return self.substep_parallel(qold, q, iterations=iterations, n_newton=n_newton)
        qlat = (q * self.dx).index_select(-1, self.res_cell)      # the same every iteration
        for _ in range(iterations):
            qin = torch.zeros_like(qk)
            qin.index_add_(-1, self.edge_dst, qk.index_select(-1, self.edge_src))
            R = self._release(qin.index_select(-1, self.res_cell) + qlat)
            qk = self.newton(qin, qold, q, n_iter=n_newton)
            qk.index_copy_(-1, self.res_cell, R)       # only the reservoir cells are overwritten
        qin = torch.zeros_like(qk)
        qin.index_add_(-1, self.edge_dst, qk.index_select(-1, self.edge_src))
        I = qin.index_select(-1, self.res_cell) + qlat  # the inflow the converged field delivers
        R = self._release(I)
        qk.index_copy_(-1, self.res_cell, R)
        self._update_storage(I, R)
        return qk

    def _build_level_res(self):
        """which reservoirs sit in each topological level, so the level loop never has to ask"""
        cells = self.cells_by_level.detach().cpu().numpy()
        ptr = self.level_ptr.detach().cpu().numpy() if torch.is_tensor(self.level_ptr) else self.level_ptr
        is_res = self.is_res.detach().cpu().numpy()
        pos = self.res_pos.detach().cpu().numpy()
        out = []
        for l in range(self.net.n_levels):
            c = cells[ptr[l]:ptr[l + 1]]
            m = is_res[c]
            if m.any():
                rc = c[m]
                out.append((torch.as_tensor(rc, dtype=torch.int64, device=self.device),
                            torch.as_tensor(pos[rc], dtype=torch.int64, device=self.device),
                            torch.as_tensor(np.flatnonzero(m), dtype=torch.int64, device=self.device)))
            else:
                out.append(None)
        self._level_res = out

    def ordered_res(self, qold, q, n_newton=4):
        if self.res_cell.numel() == 0:
            return self.substep_ordered(qold, q, n_newton=n_newton)
        qnew = torch.zeros_like(qold + q)
        R = torch.zeros_like(self.S)
        qlat = q * self.dx
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
            lr = self._level_res[l]
            if lr is not None:                          # the reservoirs of this level only
                rc, idx, within = lr
                I = qin.index_select(-1, rc) + qlat.index_select(-1, rc)
                Rl = self._release_at(idx, I)
                R.index_copy_(-1, idx, Rl.to(torch.float64))
                sol = sol.index_copy(-1, within, Rl)
            qnew[..., cells] = sol
        qin = torch.zeros_like(qnew)
        qin.index_add_(-1, self.edge_dst, qnew.index_select(-1, self.edge_src))
        I = qin.index_select(-1, self.res_cell) + qlat.index_select(-1, self.res_cell)
        self._update_storage(I, R.to(self.dtype))
        return qnew
