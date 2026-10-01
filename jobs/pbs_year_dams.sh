#!/bin/bash
#PBS -N y05dams
#PBS -q normal
#PBS -l select=1:ncpus=16:ngpus=1:mem=160gb
#PBS -l walltime=08:00:00
#PBS -P ${PBS_PROJECT:-your_project}
#PBS -j oe
#PBS -o logs/
# Convergence with the real reservoirs, on the 5 arc-minute network and the 2010 runoff, so the
# dam case can be held to the same threshold as section 4.1. The serial kernel carries no
# reservoirs, so the reference is the level-by-level GPU solution, which keeps the upstream to
# downstream order and uses the same arithmetic as SURGE; what is left is the truncated iteration.
set -e
A=${SURGE_REPO:-$PBS_O_WORKDIR}
cd $A
module load pytorch/2.6.0-py3-cu11.8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export YEAR_OUT=/scratch/users/nus/e0816573/paper1_year_dams
export YEAR_RES=$A/${SURGE_DATA:-data}/res_grand_05min.npz
B=experiments/bench_year_resolution.py
mkdir -p $YEAR_OUT
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
echo "=== spin with dams $(date)"
python -u $B 05min spin
echo "=== ordered reference year with dams $(date)"
YEAR_REF=ordered python -u $B 05min ordered
echo "=== SURGE years with dams $(date)"
python -u $B 05min gpu 1,2,3,4,5,6,7,8,9,10,11,12
echo "=== metrics against the ordered solution $(date)"
YEAR_REF=ordered YEAR_KS=1,2,3,4,5,6,7,8,9,10,11,12 python -u $B 05min metrics
echo "=== done $(date)"
