# The jobs behind the paper

The experiments ran on ASPIRE 2A, a PBS Pro cluster, on nodes holding four
NVIDIA A100-SXM4-40GB and 128 AMD EPYC 7713 cores. These are the scripts that
submitted them. They are here so the resources and the order each experiment
needed are on record, not only the solver it called: which stages chain, which
run beside each other, how long each took, and how much memory the largest
network asked for.

Three things are set from outside, so the scripts carry no path of their own:

    export SURGE_REPO=/path/to/this/repository   # defaults to the submit directory
    export SURGE_DATA=/path/to/the/data          # the networks, reservoirs and runoff
    export PBS_PROJECT=your_project              # the account to charge

A different scheduler needs the `#PBS` lines rewritten; everything below them is
ordinary shell.

## The year at each resolution, and how closely SURGE matches

`submit_year_resolution.sh` submits the whole chain for all five networks and is
the entry point. It reads a table at the bottom, one row per network, giving the
walltime of each stage and the memory the metrics stage needs, and it chains the
jobs so that each starts when the one it reads from has finished. It calls
`pbs_year_resolution.sh`, which runs one stage of
`experiments/bench_year_resolution.py`.

| Network | spin | serial | gpu | metrics |
|---|---|---|---|---|
| 30 arcmin | 1 h | 2 h | 2 h | 16 GB, 1 h |
| 15 arcmin | 1 h | 2 h | 2 h | 32 GB, 1 h |
| 5 arcmin | 2 h | 6 h | 6 h | 64 GB, 2 h |
| 3 arcmin | 3 h | 12 h | 10 h | 96 GB, 3 h |
| 1 arcmin | 6 h | 24 h x 2 | 24 h | 192 GB, 8 h |

The serial year is the long pole, and at 1 arc-minute it needs two chained jobs
because a day of it costs about twelve minutes. The gpu stage walks the
iteration count upwards against that year, so it waits for it; passing an
explicit list of counts instead lets the two run side by side.

Produces: Figure 3, Figure 4, Table 6.

`pbs_year_dams.sh` is the same year on the 5 arc-minute network with the GRanD
reservoirs operated, against the level-by-level solution rather than the serial
kernel, which carries no reservoirs. Produces Figure 7b.

## What each configuration costs

Every timing comes from `experiments/bench_unified_timing.py` under one
protocol: a uniform runoff of 2 mm/day, the steady field of that forcing shared
by every configuration, ten warm-up steps and 600 timed ones. The node is taken
whole, because GPU-ordered launches one kernel group per level and slows down
when something else is on the cores.

| Script | What it times |
|---|---|
| `pbs_unified_timing_gpu_excl.sh` | SURGE and GPU-ordered on every network, and with the reservoirs at 5, 3 and 1 arcmin |
| `pbs_unified_timing_cpu_excl.sh` | CPU-serial and CPU-parallel on every network |
| `pbs_surge_sweeps.sh` | SURGE across iteration counts, which gives the cost per iteration |
| `pbs_chosen_k_gpu.sh` | the counts each network needs that the sweeps do not already cover, and the 26 basins |
| `pbs_chosen_k_cpu.sh` | the CPU side of the same |
| `pbs_unified_timing_basins_excl.sh` | the 26 basins routed on their own |
| `pbs_reservoir_timing_v3.sh` | what the reservoirs cost, three repeats of each configuration |

Produces: Figure 5, Figure 6, Figure 7a.

The reservoir timing was measured twice. The first pass ran each configuration
once, and the spread between repeats on the node turned out to be wider than the
cost being measured, so `pbs_reservoir_timing_v3.sh` runs each three times and
reports the median. The published numbers are that second pass.
