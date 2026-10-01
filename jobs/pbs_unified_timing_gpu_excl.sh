#!/bin/bash
#PBS -N timing_gpu_excl
#PBS -q normal
#PBS -l select=1:ncpus=64:ngpus=4:mem=440gb
#PBS -l place=excl
#PBS -l walltime=08:00:00
#PBS -P ${PBS_PROJECT:-your_project}
#PBS -j oe
#PBS -o logs/
# GPU half of the unified timing protocol on a whole A100 node, so no other job shares
# its CPU cores (GPU-ordered launches one kernel group per level and slowed when the
# shared node was loaded). One GPU is used. At 5, 3 and 1 arc-minutes both schemes run
# three ways in the same job: without the reservoir code, with it but no reservoir, and
# with the GRanD reservoirs, which separates the cost of the code path from that of the dams.
set -e
A=${SURGE_REPO:-$PBS_O_WORKDIR}
cd $A
module load pytorch/2.6.0-py3-cu11.8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
B=experiments/bench_unified_timing.py
D=${SURGE_DATA:-data}
S=$D/timing_states
O=timing/aspire_excl
mkdir -p $O
echo "=== exclusive GPU node $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ($(date)) ==="
for n in 30min 15min 05min 03min 01min; do
  for c in surge gpu_ordered; do
    python -u $B time $c $D/routing_$n.npz $S/steady_$n.npz $O/${c}_$n.json
  done
done
for n in 05min 03min 01min; do
  case $n in 05min) R=$D/res_grand_05min.npz;; 03min) R=$D/res_grand_03min.npz;; 01min) R=$D/res_grand_01min.npz;; esac
  for c in surge gpu_ordered; do
    python -u $B time $c $D/routing_$n.npz $S/steady_$n.npz $O/${c}_resempty_$n.json --res empty
    python -u $B time $c $D/routing_$n.npz $S/steady_$n.npz $O/${c}_dams_$n.json --res $R
  done
done
echo "=== done ($(date)) ==="
