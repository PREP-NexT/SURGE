#!/bin/bash
#PBS -N chosenk_gpu
#PBS -q normal
#PBS -l select=1:ncpus=64:ngpus=4:mem=440gb
#PBS -l place=excl
#PBS -l walltime=06:00:00
#PBS -P ${PBS_PROJECT:-your_project}
#PBS -j oe
#PBS -o logs/
# The GPU timings that the sweeps do not cover: the reservoirs at the count each network
# needs, and the 26 basins at the thirty the 1 arc-minute network needs. The 1 arc-minute
# line is here as well, because the sweep that was run first asked for twenty-nine.
# The paper now takes only the 26-basin line from this job (Figure 8, ASPIRE 2A, 2026-09-19).
# The network and reservoir timings it reports come from pbs_timing_gpu.sh.
# Protocol unchanged: 2 mm/day steady state, 10 warm-up steps, 600 timed.
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
echo "=== chosen-k GPU timings, exclusive node $(hostname), $(date) ==="
SURGE_ITERATIONS=30 python -u $B time surge $D/routing_01min.npz $S/steady_01min.npz $O/surge_k30_01min.json
for kn in 7:05min:res_grand_05min 11:03min:res_grand_03min 30:01min:res_grand_01min; do
  k=${kn%%:*}; rest=${kn#*:}; n=${rest%%:*}; r=${rest#*:}
  SURGE_ITERATIONS=$k python -u $B time surge $D/routing_$n.npz $S/steady_$n.npz $O/surge_dams_k${k}_$n.json --res $D/$r.npz
done
SURGE_ITERATIONS=30 python -u $B time-basins surge $D/routing_01min.npz $S/steady_01min.npz $O/basins_surge_k30_01min.json
echo "=== done $(date) ==="
