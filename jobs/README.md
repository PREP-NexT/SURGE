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

Every stage but the serial one runs on one GPU with 16 cores and 160 GB. The
serial stage needs 2 cores and 64 GB, and the metrics stage needs the memory in
the last column, which is what it takes to hold the evaluated cells of every
iteration count at once.

A year is 7,300 routing time steps, so the walltimes below follow from the cost
of one step, which Figure 5 reports. The spin and gpu stages run a year each,
the serial stage runs one year on one core, and the metrics stage is bound by
reading the fields back, not by arithmetic.

| Network | spin | serial | gpu, per count | metrics |
|---|---|---|---|---|
| 30 arcmin | 1 h asked, 1 min | 2 h asked, 2 min | seconds | 16 GB, 1 h asked, minutes |
| 15 arcmin | 1 h asked, 2 min | 2 h asked, 8 min | seconds | 32 GB, 1 h asked, minutes |
| 5 arcmin | 2 h asked, 10 min | 6 h asked, 1.3 h | about 1 min | 64 GB, 2 h asked, under 1 h |
| 3 arcmin | 3 h asked, 25 min | 12 h asked, 3.6 h | about 4 min | 96 GB, 3 h asked, about 1 h |
| 1 arcmin | 6 h asked, 2 h | 24 h x 2 asked, 34 h | about 15 min | 192 GB, 8 h asked, several h |

The serial year is the long pole: at 1 arc-minute it takes about 34 hours, which
is why it is chained across two 24-hour jobs and checkpointed every ten days.
The gpu stage walks the iteration count upwards against that year, so it waits
for it; passing an explicit list of counts instead lets the two run side by
side. Thirty counts at 1 arc-minute is about seven hours of GPU and 2.3 TB of
daily fields, which is the other reason that network is the expensive one.

Produces: Figure 3, Figure 4, Table 6.

`pbs_year_dams.sh` is the same year on the 5 arc-minute network with the GRanD
reservoirs operated, against the level-by-level solution rather than the serial
kernel, which carries no reservoirs. One GPU, 16 cores, 160 GB, 8 h asked for,
about 2 h taken: the level-by-level reference year is most of it, and the twelve
iteration counts beside it are a minute each. Produces Figure 7b.

## Disk

The stages keep every daily field, which is what lets a later question cost a
metrics run instead of a model run, and at 1 arc-minute that is 75.9 GB a year.
Thirty counts, the spin year and the serial year come to about 2.3 TB for that
network alone, and 2.2 TB for the other four together. A scratch filesystem is
the place for them.

## What each configuration costs

Every timing comes from `experiments/bench_unified_timing.py` under one
protocol: a uniform runoff of 2 mm/day, the steady field of that forcing shared
by every configuration, ten warm-up steps and 600 timed ones. The node is taken
whole, because GPU-ordered launches one kernel group per level and slows down
when something else is on the cores.

| Script | What it times | Node | Asked for | Ran for |
|---|---|---|---|---|
| `pbs_unified_timing_gpu_excl.sh` | SURGE and GPU-ordered on every network, and with the reservoirs at 5, 3 and 1 arcmin | 1 A100, 64 cores, whole node | 8 h | about 2.5 h |
| `pbs_unified_timing_cpu_excl.sh` | CPU-serial and CPU-parallel on every network | 127 cores, whole node | 14 h | about 4 h |
| `pbs_surge_sweeps.sh` | SURGE at each of the counts Table 6 reports, on every network | 1 A100, whole node | 8 h | about 10 min |
| `pbs_chosen_k_gpu.sh` | the reservoirs at the count each network needs, the 26 basins, and 1 arc-minute at thirty | 1 A100, whole node | 6 h | about 1.5 h |
| `pbs_chosen_k_cpu.sh` | CPU-parallel at the count each network needs | 127 cores, whole node | 8 h | about 3.5 h, nearly all of it 1 arc-minute |
| `pbs_unified_timing_basins_excl.sh` | the 26 basins routed on their own | 1 A100, whole node | 3 h | about 1.5 h |
| `pbs_reservoir_timing_v3.sh` | what the reservoirs cost, three repeats of each configuration | 1 A100, whole node | 10 h | 4 h 40 min |

Produces: Figure 5, Figure 6, Figure 7a.

The nodes are taken whole and the memory is asked for in full because that is
what an exclusive placement gives; the timings themselves need far less. What
they do need is the card and the cores to themselves. A GPU job uses one of the
four cards.

Most of these ran well inside their walltime. The one that is nearly all of its
own cost is CPU-parallel at 1 arc-minute: a step there takes about 18 seconds,
so 600 of them is three hours on their own.

The reservoir timing was measured twice. The first pass ran each configuration
once, and the spread between repeats on the node turned out to be wider than the
cost being measured, so `pbs_reservoir_timing_v3.sh` runs each three times and
reports the median. The published numbers are that second pass.
