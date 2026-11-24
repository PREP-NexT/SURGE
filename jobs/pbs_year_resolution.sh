#!/bin/bash
#PBS -P ${PBS_PROJECT:-your_project}
#PBS -j oe
#PBS -o logs/
# One stage of bench_year_resolution.py on ASPIRE. TAG and STAGE come from qsub -v, and for
# the gpu stage KLIST lists the iteration counts separated by colons (qsub -v splits on commas).
# Every simulated field goes to YEAR_OUT, which must hold the ~760 GB the five networks need.
set -e
A=${SURGE_REPO:-$PBS_O_WORKDIR}
cd $A
module load pytorch/2.6.0-py3-cu11.8
# the serial stage needs GPU_KWR_T5_SO, the compiled CWatM kernel, set from outside
[ "$STAGE" = serial ] && : "${GPU_KWR_T5_SO:?set GPU_KWR_T5_SO to the compiled CWatM t5 library}"
# about 760 GB for the five networks, so point YEAR_OUT at a scratch file system
export YEAR_OUT=${YEAR_OUT:-$A/output/year_resolution}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $YEAR_OUT
# KS_LIST names the iteration counts the metrics stage summarises, colon separated for
# the same reason as KLIST
[ -n "${KS_LIST:-}" ] && export YEAR_KS=$(echo $KS_LIST | tr ':' ',')
echo "=== $TAG $STAGE ${KLIST:-}${KS_LIST:+ ks=$KS_LIST} on $(hostname), $(date) ==="
if [ "$STAGE" = spin ] || [ "$STAGE" = gpu ]; then nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1; fi
python -u experiments/bench_year_resolution.py $TAG $STAGE $(echo ${KLIST:-} | tr ':' ',')
echo "=== done $(date) ==="
