#!/bin/bash
#PBS -N timing_cpu_excl
#PBS -q normal
#PBS -l select=1:ncpus=127:mem=440gb
#PBS -l place=excl
#PBS -l walltime=14:00:00
#PBS -P ${PBS_PROJECT:-your_project}
#PBS -j oe
#PBS -o logs/
# CPU half of the unified timing protocol on a whole CPU node, one job at a time:
# CPU-serial (the CWatM C++ kernel, the same t5_linux.so binary as the workstation) and
# CPU-parallel (the fixed point on 24 threads) at five resolutions, then CPU-serial on the
# 26 basins of the 1 arc-minute network.
# The paper now takes only the 26-basin line from this job (Figure 8, ASPIRE 2A, 2026-09-14).
# The CPU-serial and CPU-parallel timings of the networks come from pbs_timing_cpu.sh.
set -e
A=${SURGE_REPO:-$PBS_O_WORKDIR}
cd $A
module load pytorch/2.6.0-py3-cu11.8
# the CWatM serial kernel, compiled by the user; see the main README
: "${GPU_KWR_T5_SO:?set GPU_KWR_T5_SO to the compiled CWatM t5 library}"
export GPU_KWR_T5_SO
B=experiments/bench_unified_timing.py
D=${SURGE_DATA:-data}
S=$D/timing_states
O=timing/aspire_cpu_excl
mkdir -p $O
echo "=== exclusive CPU node $(hostname), $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) ($(date)) ==="
NETS="30min 15min 05min 03min 01min"
for n in $NETS; do python -u $B time cpu_serial $D/routing_$n.npz $S/steady_$n.npz $O/cpu_serial_$n.json; done
for n in $NETS; do python -u $B time cpu_parallel $D/routing_$n.npz $S/steady_$n.npz $O/cpu_parallel_$n.json --threads 24; done
python -u $B time-basins cpu_serial $D/routing_01min.npz $S/steady_01min.npz $O/basins_cpu_serial_01min.json
echo "=== done ($(date)) ==="
