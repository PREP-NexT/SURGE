#!/bin/bash
#PBS -N timing_basins_excl
#PBS -q normal
#PBS -l select=1:ncpus=64:ngpus=4:mem=440gb
#PBS -l place=excl
#PBS -l walltime=03:00:00
#PBS -P ${PBS_PROJECT:-your_project}
#PBS -j oe
#PBS -o logs/
# SURGE on the 26 basins of the 1 arc-minute network, on a whole A100 node, so the basin
# timings share the conditions of the resolution series (the first basin run shared its node).
set -e
A=${SURGE_REPO:-$PBS_O_WORKDIR}
cd $A
module load pytorch/2.6.0-py3-cu11.8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
B=experiments/bench_unified_timing.py
D=${SURGE_DATA:-data}
O=timing/aspire_excl
echo "=== exclusive GPU node $(hostname) ($(date)) ==="
python -u $B time-basins surge $D/routing_01min.npz $D/timing_states/steady_01min.npz $O/basins_surge_01min.json
echo "=== done ($(date)) ==="
