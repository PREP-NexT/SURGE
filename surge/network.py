"""Load a routing-network graph and build GPU-friendly structures.

A routing network is stored as a compressed .npz (see build/). The serial
kinematic wave walks cells in topological order (dirDown), pulling inflow from
already-updated upstream cells via a CSR structure (dirupLen/dirupID). Here we:

1. reconstruct the topological *levels* (all cells in one level have all their
   upstream neighbours in strictly lower levels) -> the wavefront schedule;
2. build a flat edge list (upstream_cell -> downstream_cell) for scatter-add
   inflow accumulation on the GPU.

Both are read directly from the .npz when the builder has precomputed them
(``level``, ``edge_src``, ``edge_dst`` present) -- essential for large networks
where the pure-Python fallback below would be prohibitively slow.
"""
import numpy as np


class RoutingNetwork:
    def __init__(self, npz_path, use_lr=False):
        d = np.load(npz_path)
        suf = '_LR' if use_lr else ''
        self.dirDown = d['dirDown' + suf].astype(np.int64)
        self.dirupLen = d['dirupLen' + suf].astype(np.int64)
        self.dirupID = d['dirupID' + suf].astype(np.int64)
        self.lendirDown = int(d['lendirDown' + suf])
        self.channelAlpha = d['channelAlpha'].astype(np.float64)
        self.beta = float(d['beta'])
        self.chanLength = d['chanLength'].astype(np.float64)
        self.dtRouting = float(d['dtRouting'])
        self.noRoutingSteps = int(d['noRoutingSteps'])
        self.cellArea = d['cellArea'].astype(np.float64)
        self.mask2d = d['mask2d']
        self.n_cells = int(d['mapC'])
        self.waterBodyID = d['waterBodyID'] if 'waterBodyID' in d else None

        if all(k in d for k in ('level', 'edge_src', 'edge_dst')):
            # fast path: builder precomputed the schedule (required for big nets)
            self.level = d['level'].astype(np.int64)
            self.edge_src = d['edge_src'].astype(np.int64)
            self.edge_dst = d['edge_dst'].astype(np.int64)
            order = np.argsort(self.level[self.dirDown], kind='stable')
            self.dirDown_by_level = self.dirDown[order]
            lv = self.level[self.dirDown_by_level]
            self.n_levels = int(lv.max()) + 1 if len(lv) else 0
            self.level_ptr = np.searchsorted(lv, np.arange(self.n_levels + 1))
        else:
            self._build_levels()
            self._build_edges()

    def _build_levels(self):
        """Pure-Python fallback: topological level of each cell along dirDown."""
        level = np.zeros(self.n_cells, dtype=np.int64)
        for c in self.dirDown:
            lo, hi = self.dirupLen[c], self.dirupLen[c + 1]
            ups = self.dirupID[lo:hi]
            level[c] = level[ups].max() + 1 if hi > lo else 0
        self.level = level
        order = np.argsort(level[self.dirDown], kind='stable')
        self.dirDown_by_level = self.dirDown[order]
        lv = level[self.dirDown_by_level]
        self.n_levels = int(lv.max()) + 1 if len(lv) else 0
        self.level_ptr = np.searchsorted(lv, np.arange(self.n_levels + 1))

    def _build_edges(self):
        """Pure-Python fallback: flat (src -> dst) edge arrays over dirDown."""
        dst, src = [], []
        for c in self.dirDown:
            lo, hi = self.dirupLen[c], self.dirupLen[c + 1]
            for j in range(lo, hi):
                src.append(self.dirupID[j])
                dst.append(c)
        self.edge_src = np.asarray(src, dtype=np.int64)
        self.edge_dst = np.asarray(dst, dtype=np.int64)

    def summary(self):
        return dict(n_cells=self.n_cells, n_routed=len(self.dirDown),
                    n_edges=len(self.edge_src), n_levels=self.n_levels,
                    noRoutingSteps=self.noRoutingSteps, dtRouting=self.dtRouting,
                    beta=self.beta)
