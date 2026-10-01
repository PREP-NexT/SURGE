#!/bin/bash
#PBS -N res_timing_v3
#PBS -q normal
#PBS -l select=1:ncpus=64:ngpus=4:mem=440gb
#PBS -l place=excl
#PBS -l walltime=10:00:00
#PBS -P ${PBS_PROJECT:-your_project}
#PBS -j oe
#PBS -o logs/
# What the reservoirs cost, re-measured after the reservoir router was rewritten: the sweep no
# longer touches the whole grid to write a few hundred cells, and the level loop no longer asks
# the device, once per level, whether that level holds a dam. Each river network is timed at the
# iteration count it needs (7, 11, 30), three ways for each scheme: without the reservoir code,
# with the code but no dam, and with the GRanD dams. Protocol otherwise unchanged from
# bench_unified_timing.py: 2 mm/day steady state, 10 warm-up steps, 600 timed steps.
set -e
A=${SURGE_REPO:-$PBS_O_WORKDIR}
cd $A
module load pytorch/2.6.0-py3-cu11.8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
B=experiments/bench_unified_timing.py
D=${SURGE_DATA:-data}
S=$D/timing_states
O=timing/aspire_res_v3
mkdir -p $O
echo "=== reservoir timing v3, three repeats per configuration, exclusive node $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), $(date) ==="
for kn in 7:05min:res_grand_05min 11:03min:res_grand_03min 30:01min:res_grand_01min; do
  k=${kn%%:*}; rest=${kn#*:}; n=${rest%%:*}; r=${rest#*:}
  for c in surge gpu_ordered; do
    SURGE_ITERATIONS=$k python -u $B time $c $D/routing_$n.npz $S/steady_$n.npz $O/${c}_none_$n.json --repeats 3
    SURGE_ITERATIONS=$k python -u $B time $c $D/routing_$n.npz $S/steady_$n.npz $O/${c}_resempty_$n.json --res empty --repeats 3
    SURGE_ITERATIONS=$k python -u $B time $c $D/routing_$n.npz $S/steady_$n.npz $O/${c}_dams_$n.json --res $D/$r.npz --repeats 3
  done
done
echo "=== done $(date) ==="
