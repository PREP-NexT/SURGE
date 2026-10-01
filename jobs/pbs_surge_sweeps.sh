#!/bin/bash
#PBS -N surge_sweeps
#PBS -q normal
#PBS -l select=1:ncpus=64:ngpus=4:mem=440gb
#PBS -l place=excl
#PBS -l walltime=08:00:00
#PBS -P ${PBS_PROJECT:-your_project}
#PBS -j oe
#PBS -o logs/
# SURGE timed at several iteration counts, on a whole A100 node so nothing else shares its CPU
# cores, under the same protocol as the k = 3 timings: the steady state of 2 mm/day, 10 warm-up
# steps, then the mean over 600 steps. Each network needs its own k to match the serial solution
# to the same degree, so the paper needs the cost at those k as well as at k = 3. The counts are
# the ones Table 6 reports, 2, 3, 7, 11 and 30, and every network is timed at all of them so the
# cost of an iteration can be read across resolutions.
set -e
A=${SURGE_REPO:-$PBS_O_WORKDIR}
cd $A
module load pytorch/2.6.0-py3-cu11.8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
B=experiments/bench_unified_timing.py
D=${SURGE_DATA:-data}
S=$D/timing_states
O=timing/aspire_sweeps
mkdir -p $O
echo "=== SURGE at several k, exclusive node $(hostname), $(date) ==="
for n in 30min 15min 05min 03min 01min; do
  for k in 2 3 7 11 30; do
    SURGE_ITERATIONS=$k python -u $B time surge $D/routing_$n.npz $S/steady_$n.npz $O/surge_k${k}_$n.json
  done
done
echo "=== done $(date) ==="
