#!/bin/bash
#PBS -N timing_cpu
#PBS -q normal
#PBS -l select=1:ncpus=127:mem=440gb
#PBS -l place=excl
#PBS -l walltime=14:00:00
#PBS -P ${PBS_PROJECT:-your_project}
#PBS -j oe
#PBS -o logs/
# CPU half of the timing set behind Table S5 and Figure 7 (run on ASPIRE 2A on 2026-11-23 as
# job 26349829, together with pbs_timing_gpu.sh), on one whole AMD EPYC 7713 node:
#   CPU-serial   = the CWatM C++ kernel on one core
#   CPU-parallel = the SURGE iteration on 24 threads, at the iteration count of each network
# 10 warm-up and 600 timed steps, one run each. No output that already exists is overwritten.
set -e
A=${SURGE_REPO:-$PBS_O_WORKDIR}
cd $A
module load pytorch/2.6.0-py3-cu11.8
export PYTHONDONTWRITEBYTECODE=1
# the CWatM serial kernel, compiled by the user; see the main README
: "${GPU_KWR_T5_SO:?set GPU_KWR_T5_SO to the compiled CWatM t5 library}"
export GPU_KWR_T5_SO
B=experiments/bench_unified_timing.py
D=${SURGE_DATA:-data}
S=$D/timing_states
O=timing/cpu
L=$O/cpu.log
mkdir -p $O
fresh() { if [ -e "$1" ]; then echo "SKIP: $1 exists, not overwritten" >> $L; return 1; fi; }
{
echo "=== CPU timing, job $PBS_JOBID, exclusive node $(hostname),$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2), $(nproc) logical CPUs, $(date) ==="
echo "md5 $(md5sum $B $GPU_KWR_T5_SO | tr '\n' ' ')"
} >> $L

# CPU-serial at five resolutions (does not iterate)
for n in 30min 15min 05min 03min 01min; do
  f=$O/cpu_serial_$n.json; fresh $f || continue
  python -u $B time cpu_serial $D/routing_$n.npz $S/steady_$n.npz $f >> $L 2>&1 || echo "FAILED: $f" >> $L
done
# CPU-parallel on 24 threads at the iteration count of each network
for kn in 2:30min 3:15min 7:05min 11:03min 30:01min; do
  k=${kn%%:*}; n=${kn#*:}
  f=$O/cpu_parallel_k${k}_$n.json; fresh $f || continue
  SURGE_ITERATIONS=$k python -u $B time cpu_parallel $D/routing_$n.npz $S/steady_$n.npz $f --threads 24 >> $L 2>&1 || echo "FAILED: $f" >> $L
done
echo "=== done $(date) ===" >> $L
