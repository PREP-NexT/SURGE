# The jobs behind the paper

The experiments ran on ASPIRE 2A, a PBS Pro cluster, on nodes holding four
NVIDIA A100-SXM4-40GB and 128 AMD EPYC 7713 cores. These are the scripts that
submitted them. They are here so the resources and the order each experiment
needed are on record, not only the solver it called: which stages chain, which
run beside each other, how long each took, and how much memory the largest
network asked for.

These are set from outside, so the scripts carry no path of their own:

    export SURGE_REPO=/path/to/this/repository   # defaults to the submit directory, or the repository for submit_year_resolution.sh
    export SURGE_DATA=/path/to/the/data          # the networks, reservoirs and runoff
    export PBS_PROJECT=your_project              # the account to charge

and two more for some of them:

    export GPU_KWR_T5_SO=/path/to/t5_linux.so    # the CWatM serial kernel, for CPU-serial
    export YEAR_OUT=/path/to/scratch             # where the year runs write their fields

PBS reads the `#PBS` lines before the shell starts, so it does not expand
`${PBS_PROJECT:-your_project}` in them. Pass the account at submission, as in
`qsub -P "$PBS_PROJECT" jobs/pbs_timing_gpu.sh`, or write it into the `#PBS -P`
line. `submit_year_resolution.sh` passes it to every job it submits when
`PBS_PROJECT` is set.

A different scheduler needs the `#PBS` lines rewritten. Everything below them is
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
of one step, which Figure 7 reports. The spin and gpu stages run a year each,
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

Produces: Figure 5, Figure 6, Table 6.

`pbs_year_dams.sh` is the same year on the 5 arc-minute network with the GRanD
reservoirs operated, against the level-by-level solution rather than the serial
kernel, which carries no reservoirs. One GPU, 16 cores, 160 GB, 8 h asked for,
about 2 h taken: the level-by-level reference year is most of it, and the twelve
iteration counts beside it are a minute each. Produces Figure 9b, whose dam
curve is these stages re-run on an RTX A6000 after the release was clipped to the
storage bounds.

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

The network timings of the paper come from one pair of jobs, submitted together
on 2026-11-23, so that every configuration shares one baseline. The 26-basin
timings of Figure 8 come from two earlier jobs under the same protocol.

| Script | What it times | Node | Asked for | Produces |
|---|---|---|---|---|
| `pbs_timing_gpu.sh` | SURGE at the iteration count of each network and GPU-ordered, on every network; both again with the GRanD dams at 5, 3 and 1 arcmin; SURGE at 8,640 s and 21,600 s | 1 A100, 64 cores, whole node | 10 h | Table S5, Figure 7, Figure 9a, Table S6 |
| `pbs_timing_cpu.sh` | CPU-serial, and CPU-parallel at the iteration count of each network, on every network | 127 cores, whole node | 14 h | Table S5, Figure 7 |
| `pbs_chosen_k_gpu.sh` | SURGE on the 26 basins at k = 30 (its other lines are superseded by `pbs_timing_gpu.sh`) | 1 A100, whole node | 6 h | Figure 8 |
| `pbs_unified_timing_cpu_excl.sh` | CPU-serial on the 26 basins (its other lines are superseded by `pbs_timing_cpu.sh`) | 127 cores, whole node | 14 h | Figure 8 |

The iteration counts are those of Table 6: 2, 3, 7, 11 and 30 from the 30 to the
1 arcmin network, and 12, 28, 58 and 140 for the longer time steps of Table S6.
Each GPU configuration runs three times and the median is kept, because the
spread between repeats on a node can be as wide as the cost of the reservoirs.
The CPU configurations run once each, and their JSON keeps the six block means
as the spread within the run. The two timing jobs skip any output that already
exists, so a rerun never overwrites a result.

The nodes are taken whole and the memory is asked for in full because that is
what an exclusive placement gives; the timings themselves need far less. What
they do need is the card and the cores to themselves. A GPU job uses one of the
four cards.

The GPU job took about 5 hours and the CPU job about 6.5. Nearly all of the CPU
time is 1 arc-minute, where a step of CPU-serial or of CPU-parallel at thirty
iterations takes about 17 seconds, so 600 of them is three hours each.
