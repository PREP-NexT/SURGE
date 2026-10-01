#!/bin/bash
#PBS -N chosenk_cpu
#PBS -q normal
#PBS -l select=1:ncpus=127:mem=440gb
#PBS -l place=excl
#PBS -l walltime=06:00:00
#PBS -P ${PBS_PROJECT:-your_project}
#PBS -j oe
#PBS -o logs/
# CPU-parallel at the iteration count each river network needs, on the same exclusive AMD node
# as the CPU-serial baseline of Table 5, so the two are comparable. The earlier sweep ran on a
# workstation Intel part and is not: at 1 arcmin it was 2.9 times slower than this node at k = 3.
# A 1 arc-minute step at k = 30 costs about 18 s, so that case is timed over fewer steps.
set -e
A=${SURGE_REPO:-$PBS_O_WORKDIR}
cd $A
module load pytorch/2.6.0-py3-cu11.8
export GPU_KWR_T5_SO=$A/cwatm_fork/cwatm/hydrological_modules/routing_reservoirs/t5_linux.so
B=experiments/bench_unified_timing.py
D=${SURGE_DATA:-data}
S=$D/timing_states
O=timing/aspire_cpu_sweeps
mkdir -p $O
echo "=== chosen-k CPU-parallel, exclusive node $(hostname), $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) ($(date)) ==="
for kn in 2:30min:600 3:15min:600 7:05min:600 11:03min:300 30:01min:120; do
  k=${kn%%:*}; rest=${kn#*:}; n=${rest%%:*}; t=${rest##*:}
  SURGE_ITERATIONS=$k TIMING_STEPS=$t python -u $B time cpu_parallel $D/routing_$n.npz $S/steady_$n.npz $O/cpu_parallel_k${k}_$n.json --threads 24
done
echo "=== done $(date) ==="
