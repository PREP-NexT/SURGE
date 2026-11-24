#!/bin/bash
#PBS -N timing_gpu
#PBS -q normal
#PBS -l select=1:ncpus=64:ngpus=4:mem=440gb
#PBS -l place=excl
#PBS -l walltime=10:00:00
#PBS -P ${PBS_PROJECT:-your_project}
#PBS -j oe
#PBS -o logs/
# GPU half of the timing set behind Table S5, Figure 7, Figure 9a and Table S6 (run on
# ASPIRE 2A on 2026-11-23 as job 26349828, together with pbs_timing_cpu.sh). One whole A100
# node, one GPU used, so no other job shares its CPU cores. Every configuration runs three
# times and the median is kept. The runs without reservoirs are the baselines of the
# reservoir comparison and the 4,320 s rows of Table S6. No output that already exists is
# overwritten.
set -e
A=${SURGE_REPO:-$PBS_O_WORKDIR}
cd $A
module load pytorch/2.6.0-py3-cu11.8
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
B=experiments/bench_unified_timing.py
T=experiments/bench_timestep.py
D=${SURGE_DATA:-data}
S=$D/timing_states
O=timing/gpu
L=$O/gpu.log
mkdir -p $O
fresh() { if [ -e "$1" ]; then echo "SKIP: $1 exists, not overwritten" >> $L; return 1; fi; }
{
echo "=== GPU timing, job $PBS_JOBID, exclusive node $(hostname), $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1), CPU$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2), $(date) ==="
echo "md5 $(md5sum $B $T | tr '\n' ' ')"
} >> $L

# 1. Table S5 and Figure 7: SURGE at the iteration count of each network, and GPU-ordered
for kn in 2:30min 3:15min 7:05min 11:03min 30:01min; do
  k=${kn%%:*}; n=${kn#*:}
  for c in surge gpu_ordered; do
    f=$O/${c}_none_$n.json; fresh $f || continue
    SURGE_ITERATIONS=$k python -u $B time $c $D/routing_$n.npz $S/steady_$n.npz $f --repeats 3 >> $L 2>&1 || echo "FAILED: $f" >> $L
  done
done

# 2. Figure 9a: the same two configurations with the GRanD dams at 5, 3 and 1 arcmin
for kn in 7:05min 11:03min 30:01min; do
  k=${kn%%:*}; n=${kn#*:}
  for c in surge gpu_ordered; do
    f=$O/${c}_dams_$n.json; fresh $f || continue
    SURGE_ITERATIONS=$k python -u $B time $c $D/routing_$n.npz $S/steady_$n.npz $f --res $D/res_grand_$n.npz --repeats 3 >> $L 2>&1 || echo "FAILED: $f" >> $L
  done
done

# 3. Table S6: SURGE at 8,640 s and 21,600 s (bench_timestep.py runs three repeats itself)
for run in 05min:8640:12 05min:21600:28 01min:8640:58 01min:21600:140; do
  IFS=: read n dt k <<< "$run"
  f=$O/surge_${n}_dt${dt}_k${k}.json; fresh $f || continue
  SURGE_ITERATIONS=$k python -u $T $dt $D/routing_$n.npz $S/steady_$n.npz $f >> $L 2>&1 || echo "FAILED: $f" >> $L
done
echo "=== done $(date) ===" >> $L
