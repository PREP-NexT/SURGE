<p align="center">
  <a href="https://github.com/PREP-NexT/SURGE">
    <img src="logo/surge_flow.gif" width="420" alt="SURGE logo">
  </a>
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Build" src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple"></a>
  <a href="https://github.com/PREP-NexT/SURGE"><img src="https://img.shields.io/github/contributors/PREP-NexT/SURGE.svg" alt="GitHub contributors"></a>
  <a href="https://github.com/PREP-NexT/SURGE"><img src="https://img.shields.io/github/issues/PREP-NexT/SURGE.svg" alt="GitHub issues"></a>
  <a href="https://twitter.com/PREPNexT_Lab"><img src="https://img.shields.io/twitter/follow/PREPNexT_Lab.svg?label=Follow&style=social" alt="Twitter Follow"></a>
  <a href="https://github.com/PREP-NexT/SURGE"><img src="https://img.shields.io/github/license/PREP-NexT/SURGE.svg" alt="License"></a>
  <a href="https://github.com/PREP-NexT/SURGE"><img src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103" alt="Open Source"></a>
</p>

# SURGE: Scalable Upstream-decoupled Routing on GPU-Engine

**SURGE** is a GPU-parallel solver for the implicit kinematic-wave river routing
used by global hydrological models (CWatM, LISFLOOD, PCR-GLOBWB, wflow).
Routing normally advances cell by cell down the drainage network in strict
upstream-to-downstream order, a serial recursion that has kept it on a single
CPU core. SURGE decouples every cell from its upstream neighbours by recasting
the recursion as a convergent fixed-point iteration: the whole network
is updated at once, and a fixed, small number of iterations reproduces the
serial solution.

- The per-iteration contraction is `g = Cr / (1 + Cr)`, set by the local
  Courant number `Cr`, so the iterations a network needs follow that number,
  two on the global 30 arc-minute network and thirty on the 1 arc-minute one.
- One NVIDIA A100 routes the global networks **137 to 531 times** faster than
  the serial C kernel on one CPU core. A simulated year of the 56-million-cell
  1 arc-minute network takes about 15 minutes instead of about 34 hours.
- Managed reservoirs (the full GRanD set) are carried as a storage-based
  release at each dam cell, which costs 1.3 to 7.9 per cent more time and no
  extra iterations.
- Every operation is a tensor primitive, so the solver batches over ensembles;
  being built from differentiable operations, it also supports reverse-mode
  autograd.

Only the solution procedure changes. The governing equations and the drainage
network are untouched, so SURGE drops into existing models without
modification.

## Citation

This repository accompanies the manuscript:

> Zhanwei Liu and Xiaogang He.
> **SURGE: a GPU-parallel kinematic-wave river routing model** (in preparation).
> Department of Civil and Environmental Engineering, National University of
> Singapore.

---

## Table of contents

- [SURGE: Scalable Upstream-decoupled Routing on GPU-Engine](#surge-scalable-upstream-decoupled-routing-on-gpu-engine)
  - [Citation](#citation)
  - [Table of contents](#table-of-contents)
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
├── kinematic_torch.py       Parallel and ordered substeps, Newton kernel
├── reservoir.py             Storage-based reservoir release (GRanD)
└── serial_ref.py            Bridge to CWatM's serial C kernel (reference)
build/                       Construct networks from public flow-direction maps
├── build_multiresolution.py 30/15/5 arc-minute networks (CWatM/Eilander maps)
├── build_from_cama.py       3/1 arc-minute networks (CaMa-Flood maps)
└── patch_alpha_area.py      the channel width of the paper, w = 3*sqrt(U)
examples/                    Using SURGE: routing a runoff series, putting a
                             0.5-degree runoff field into cell order, and
                             putting the GRanD dams on a network
experiments/                 Reproducing the paper: the year at each resolution
                             against the serial kernel, the per-cell metric
                             maps, the iterations each basin needs, the two
                             reservoir checks, and the one timing protocol
                             behind every reported number
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

# one routing time step, the iterations the network needs (seven at 5 arc-minutes)
q = router.substep_parallel(q, qd, iterations=7, n_newton=4)

# exact level-scheduled traversal (machine-precision reference)
q_exact = router.substep_ordered(q, qd)
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
python examples/route_runoff.py routing_05min.npz runoff.npy out.npy 7
python examples/regrid_runoff.py routing_05min.npz qtot_2010.npz runoff.npy
python examples/map_grand_dams.py routing_05min.npz GRanD_v13.csv res_grand.npz \
    --lat0 89.9583 --dlat 0.083333 --lon0 -179.9583 --dlon 0.083333
```

The year the paper reports, at one resolution, runs in four stages, each writing
what the next one reads. The serial stage is the long one, and it is the only
stage that needs the CWatM kernel.

```bash
export SURGE_DATA=/path/to/the/networks/and/runoff
python experiments/bench_year_resolution.py 05min spin      # a year to settle the field
python experiments/bench_year_resolution.py 05min serial    # the reference year
python experiments/bench_year_resolution.py 05min gpu 1,2,3,5,7,11
python experiments/bench_year_resolution.py 05min metrics   # KGE, NSE and bias per cell
python experiments/bench_metric_maps.py routing_05min.npz runoff.npy maps.npz
python experiments/check_reservoir_massbalance.py           # the storage stays in bounds
python experiments/check_dam_convergence.py                 # the dams add no iterations
```

The timings reported in the paper come from one protocol, in
`experiments/bench_unified_timing.py`: a uniform runoff of 2 mm/day, the steady
discharge field of that forcing shared by every configuration, ten warm-up time
steps and then 600 timed ones, with `--repeats` runs of each configuration. It
takes the network and that steady state, and writes one JSON file per
configuration.

```bash
python experiments/bench_unified_timing.py steady net.npz steady.npz
python experiments/bench_unified_timing.py time surge net.npz steady.npz out.json --repeats 3
python experiments/bench_unified_timing.py time gpu_ordered net.npz steady.npz out.json \
    --res res_grand.npz --repeats 3
```

The configurations are `surge`, `gpu_ordered`, `cpu_parallel` and `cpu_serial`,
and `SURGE_ITERATIONS` sets the iteration count each network runs at. The
`cpu_serial` configuration calls the CWatM kernel, so it needs `GPU_KWR_T5_SO`.

### 3. Reproduce the paper figures

```bash
cd paper/figure_scripts
python fig_timing_bars.py                      # and the other fig_*.py
```

Each figure of the manuscript has one script and the measured data it reads, in
`paper/figure_data/`. See [paper/README.md](paper/README.md) for the figure,
script and data of each one.

## Key results

Single-run wall-clock per simulated day of one NVIDIA A100 against the serial
C kernel on one core of an AMD EPYC 7713, over the global river network. Each
network runs at the number of iterations `k` it needs to reproduce the serial
solution, and a simulated day is 20 routing time steps of 4,320 s:

| Resolution | Cells | Levels | `k` | SURGE | CPU-serial | Speedup |
|-----------:|------:|-------:|----:|------:|-----------:|--------:|
| 30 arc-min | 70,648 | 116 | 2 | 1.88 ms | 327 ms | 174x |
| **15 arc-min** | **267,950** | **251** | **3** | **2.37 ms** | **1.26 s** | **531x** |
| 5 arc-min | 2,307,532 | 782 | 7 | 26.0 ms | 12.6 s | 484x |
| 3 arc-min | 6,222,566 | 1,436 | 11 | 105 ms | 35.9 s | 341x |
| 1 arc-min | 55,812,946 | 4,696 | 30 | 2.47 s | 339 s | 137x |

The speedup peaks at the middle resolutions. A coarse network gives the card
too little work to cover the cost of starting each update, and a fine one needs
more iterations, thirty against two, because a flood wave crosses more cells in
one time step.

## License

GPL-3.0 (see `LICENSE`). The serial reference kernel interfaces with CWatM,
which is itself GPL-3.0.
