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
  - [System requirements](#system-requirements)
  - [Installation](#installation)
  - [Demo](#demo)
  - [Project layout](#project-layout)
  - [Quick start](#quick-start)
  - [Running SURGE on your own data](#running-surge-on-your-own-data)
  - [Workflow](#workflow)
    - [1. Build a routing network](#1-build-a-routing-network)
    - [2. Route and benchmark](#2-route-and-benchmark)
    - [3. Reproduce the paper figures](#3-reproduce-the-paper-figures)
  - [Key results](#key-results)
  - [License](#license)

---

## System requirements

SURGE is a Python package and needs Python 3.10 or later, PyTorch 2.0 or later,
and NumPy 1.24 or later. Nothing else is required to route a network.

It has been run on Linux, under Python 3.11 and 3.12, with PyTorch 2.2.2 on CUDA
12.1 and PyTorch 2.5 and 2.6 on CUDA 11.8. macOS and Windows are untested.

No special hardware is needed: the solver runs on CPU tensors, and the demo below
does. A CUDA GPU is what makes it fast, and the results in the paper were measured
on one NVIDIA A100-SXM4-40GB. The card has to hold the network and the fields, so
the 30 arc-minute network fits in well under a gigabyte while the 55.8-million-cell
1 arc-minute network wants a 40 GB card. Reproducing the serial baseline also needs
the CWatM kernel, which `surge/serial_ref.py` finds through `GPU_KWR_T5_SO`;
without it the ordered scheme is an equivalent machine-precision reference.

## Installation

```bash
git clone https://github.com/PREP-NexT/SURGE.git
cd SURGE
conda env create -f environment.yml
conda activate surge
pip install -e .
```

About five minutes on a desktop with a warm conda cache, most of it spent
downloading PyTorch.

## Demo

```bash
python demo/run_demo.py
```

Thirty days on the global 30 arc-minute network, routed twice, once with the
parallel scheme and once with the ordered scheme that returns the serial solution,
and then compared. It needs no GPU and no download, and takes under a minute on
four CPU threads. [demo/README.md](demo/README.md) gives the output to expect and
what the numbers in it mean.

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
jobs/                        The PBS scripts that submitted the experiments on
                             the cluster they ran on, with the resources and the
                             order each one needed
demo/                        A thirty-day run that finishes on a laptop
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

## Running SURGE on your own data

SURGE replaces the channel-routing step and nothing else. It takes the runoff a
model already produces and returns discharge along the river network, so the same
solver serves CWatM, H08, JULES, VIC, MOSART, SUMMA and anything else that writes
runoff on a grid. Three things have to line up.

**The network.** `build/build_multiresolution.py` and `build/build_from_cama.py`
turn a public flow-direction map into the `.npz` that `RoutingNetwork` reads, and
`build/patch_alpha_area.py` gives it the channel width the paper uses. A network
built once is used by every run.

**The runoff.** The solver wants lateral inflow per unit channel length, in
m2 s-1, as `q = R * cellArea / (1000 * chanLength)` for a runoff `R` in
kg m-2 s-1, which is the same as mm s-1. `examples/regrid_runoff.py` does that
for a runoff field on a coarser grid, which is the usual case: a 0.5-degree
runoff drives a 5 arc-minute network by giving every routing cell the value of
the 0.5-degree cell that holds it.

**The iteration count.** It follows the Courant number of the network, so it
belongs to the network and the time step rather than to the run: two on a 30
arc-minute network and thirty at 1 arc-minute, with the time step of 4,320 s this
study uses. A count that is too low leaves the discharge short of the serial
solution everywhere the flow is fast. To find it for a network of your own, run
the year in `experiments/bench_year_resolution.py`, whose gpu stage walks the
count upwards until the solution matches and then reports it.

With those in hand, routing a series is the loop in `examples/route_runoff.py`.
Reservoirs need `ResRouter` in place of `TorchRouter` and a file of dam cells,
capacities and mean releases, which `examples/map_grand_dams.py` writes from the
GRanD catalogue.

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
stage that needs the CWatM kernel. The gpu stage needs no list of iteration
counts: it walks k upwards against the serial year, stops one count past the one
whose 5th percentile KGE reaches 0.99, and reports what the network needs, which
is two at 30 arc-minutes and thirty at 1 arc-minute.

```bash
export SURGE_DATA=/path/to/the/networks/and/runoff
python experiments/bench_year_resolution.py 05min spin      # a year to settle the field
python experiments/bench_year_resolution.py 05min serial    # the reference year
python experiments/bench_year_resolution.py 05min gpu       # walks k up to the crossing
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

Every experiment of the paper was submitted to a PBS cluster, and `jobs/` holds
the scripts that did it, with the walltime and the memory each stage needed and
the order they chain in. See [jobs/README.md](jobs/README.md).

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
