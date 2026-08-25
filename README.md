# SURGE: Scalable Upstream-decoupled Routing on GPU-Engine

**SURGE** is a GPU-parallel solver for the implicit kinematic-wave river routing
used by global hydrological models (CWatM, LISFLOOD, PCR-GLOBWB, wflow).
Routing normally advances cell by cell down the drainage network in strict
upstream-to-downstream order, a serial recursion that has kept it on a single
CPU core. SURGE decouples every cell from its upstream neighbours by recasting
the recursion as a convergent Jacobi fixed-point iteration: the whole network
is updated at once, and a fixed, small number of iterations reproduces the
serial solution.

- The per-iteration contraction is `g = Cr / (1 + Cr)`, set by the local
  Courant number `Cr`, so about three iterations suffice in the operational
  regime and the count is independent of network depth (116 to 4,696
  topological levels).
- One NVIDIA A100 routes the 56-million-cell global 1 arc-minute network up to
  **1,046x** faster than the serial C kernel on one CPU core; a global
  model-year routes in about 90 seconds instead of 26 hours.
- Managed reservoirs (the full GRanD set) are carried as storage-based fixed
  sources at about 3% extra cost per substep.
- Every operation is a tensor primitive, so the solver batches over ensembles;
  being built from differentiable operations, it also supports reverse-mode
  autograd.

Only the solution procedure changes. The governing equations and the drainage
network are untouched, so SURGE drops into existing models without
modification.

## Citation

This repository accompanies the manuscript:

> Zhanwei Liu and Xiaogang He.
> **GPU-accelerated kinematic-wave river routing via a convergent fixed-point
> reformulation** (in preparation).
> Department of Civil and Environmental Engineering, National University of
> Singapore.

---

## Table of contents

- [Installation](#installation)
- [Project layout](#project-layout)
- [Quick start](#quick-start)
- [Workflow](#workflow)
  - [1. Build a routing network](#1-build-a-routing-network)
  - [2. Route and benchmark](#2-route-and-benchmark)
  - [3. Reproduce the paper figures](#3-reproduce-the-paper-figures)
- [Key results](#key-results)
- [License](#license)

---

## Installation

SURGE targets Python 3.10+ with PyTorch 2.0+. Install via conda using the
environment file, then install the package in editable mode:

```bash
git clone https://github.com/PREP-NexT/SURGE.git
cd SURGE
conda env create -f environment.yml
conda activate surge
pip install -e .
```

A CUDA-capable GPU is required for the GPU benchmarks; the solver also runs
(more slowly) on CPU tensors, and the serial reference kernel is optional
(see `surge/serial_ref.py`).

## Project layout

```
surge/                       Solver package
├── network.py               Routing-network loader (levels, edge lists)
├── kinematic_torch.py       Fixed-point and wavefront substeps, Newton kernel
├── reservoir.py             Storage-based reservoir release (GRanD)
└── serial_ref.py            Bridge to CWatM's serial C kernel (reference)
build/                       Construct networks from public flow-direction maps
├── build_multiresolution.py 30/15/5 arc-minute networks (CWatM/Eilander maps)
└── build_from_cama.py       3/1 arc-minute networks (CaMa-Flood maps)
examples/                    Runners and benchmarks (accuracy, depth
                             independence, resolution scaling, wavefront,
                             reservoirs, CPU-parallel baseline)
paper/                       Figure scripts, figure data, and figures of the
                             manuscript
environment.yml              Conda environment specification
```

## Quick start

```python
import torch
from surge import RoutingNetwork, TorchRouter

net = RoutingNetwork("routing_static_global.npz")     # see build/
router = TorchRouter(net, device="cuda", dtype=torch.float32)

N  = net.n_cells
q  = torch.zeros(N, device="cuda")                    # discharge state
qd = torch.full((N,), 1e-4, device="cuda")            # lateral inflow (m^2/s)

# one routing substep: three network-wide fixed-point iterations
q = router.substep_jacobi(q, qd, sweeps=3, n_newton=4)

# exact level-scheduled traversal (machine-precision reference)
q_exact = router.substep_wavefront(q, qd)
```

State and parameters broadcast over a leading ensemble axis (`(B, N)`
tensors), so one kernel invocation advances all ensemble members.

## Workflow

### 1. Build a routing network

The builders construct self-contained `.npz` network files (topology, levels,
edge lists, synthetic hydraulic geometry) from public flow-direction maps:

```bash
python build/build_multiresolution.py   # 30', 15', 5' from CWatM/Eilander maps
python build/build_from_cama.py         # 3', 1'  from CaMa-Flood maps
```

Data sources are documented in the manuscript's Open Research section.

### 2. Route and benchmark

```bash
python examples/route_runoff.py  net.npz runoff.npy out.npy   # basic end use
python examples/bench_accuracy_vs_serial.py net.npz runoff.npy
python examples/bench_depth_independence.py static/*.npz
python examples/bench_resolution_scaling.py static/*.npz
python examples/bench_wavefront.py static/*.npz
python examples/bench_reservoir.py  net.npz res_grand.npz     # + map_grand_dams.py
python examples/bench_cpu_parallel.py net.npz 24              # CPU baseline
```

### 3. Reproduce the paper figures

```bash
python paper/figure_scripts/fig_method.py      # and the other fig_*.py
```

Measured benchmark data used by the figures are stored in
`paper/figure_data/`.

## Key results

Single-run speedup of one NVIDIA A100 over the serial C kernel on one CPU
core, across the global river network (three fixed-point iterations,
torch.compile-fused):

| Resolution | Cells | Levels | Fused / substep | Speedup |
|-----------:|------:|-------:|----------------:|--------:|
| 30 arc-min | 70,648 | 116 | 0.137 ms | 74x |
| 15 arc-min | 267,950 | 251 | 0.149 ms | 264x |
| 5 arc-min | 2,307,532 | 782 | 0.550 ms | 704x |
| 3 arc-min | 6,222,566 | 1,436 | 1.419 ms | 1,010x |
| **1 arc-min** | **55,812,946** | **4,696** | **12.367 ms** | **1,046x** |

At a matched Courant number the three-iteration error does not grow across
the forty-fold range of network depth.

## License

GPL-3.0 (see `LICENSE`). The serial reference kernel interfaces with CWatM,
which is itself GPL-3.0.
