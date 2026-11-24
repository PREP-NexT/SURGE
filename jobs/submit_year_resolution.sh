#!/bin/bash
# Submit the one-year resolution experiment (bench_year_resolution.py) as PBS jobs; run on the
# ASPIRE login node, where it only calls qsub. For each network: spin on a GPU, then the
# serial year on CPU and the SURGE years on GPUs at the same time, then metrics once both
# have finished. The 1 arc-minute serial year (~34 h on one EPYC core) is split into two
# chained jobs that resume from its 30-day checkpoints, and its SURGE years into three jobs.
set -e
A=${SURGE_REPO:-$(cd "$(dirname "$0")/.." && pwd)}
J=$A/jobs/pbs_year_resolution.sh
GPU=ncpus=16:ngpus=1
# job names: PBS takes at most 15 characters and no colons, so y<res><stage>, e.g. y01ser2
q() { qsub ${PBS_PROJECT:+-P "$PBS_PROJECT"} -N "$1" -l "$2" -l walltime=$3 -v "$4" ${5:+-W depend=$5} $J; }

# tag | spin walltime | serial walltime x chained jobs | gpu iteration counts, a dash
# to let the stage find them | gpu walltime | metrics mem, walltime
while read -r TAG SPINW SERW SERN KGROUPS GPUW METM METW; do
  [ -z "$TAG" ] && continue
  RS=${TAG:0:2}
  sp=$(q y${RS}spin "select=1:$GPU:mem=160gb" $SPINW TAG=$TAG,STAGE=spin)
  prev=afterok:$sp; deps=""
  for i in $(seq 1 $SERN); do
    s=$(q y${RS}ser$i "select=1:ncpus=2:mem=64gb" $SERW TAG=$TAG,STAGE=serial $prev)
    prev=afterany:$s
  done
  deps=afterok:$s
  gi=0
  if [ "$KGROUPS" = - ]; then
    # one job, no list: the stage walks k upwards against the serial year, so it waits
    # for that year rather than running beside it
    g=$(q y${RS}gpu "select=1:$GPU:mem=160gb" $GPUW TAG=$TAG,STAGE=gpu afterok:$s)
    deps=$deps:$g
  else
    # an explicit group of counts needs no reference, so these run beside the serial year
    for kg in $(echo $KGROUPS | tr '|' ' '); do
      gi=$((gi + 1))
      g=$(q y${RS}gpu$gi "select=1:$GPU:mem=160gb" $GPUW TAG=$TAG,STAGE=gpu,KLIST=$kg afterok:$sp)
      deps=$deps:$g
    done
  fi
  m=$(q y${RS}met "select=1:ncpus=4:mem=$METM" $METW TAG=$TAG,STAGE=metrics $deps)
  echo "$TAG: spin $sp, last serial $s, metrics $m"
done <<'LIST'
30min   01:00:00 02:00:00 1 -  02:00:00 16gb  01:00:00
15min   01:00:00 02:00:00 1 -  02:00:00 32gb  01:00:00
05min   02:00:00 06:00:00 1 -  06:00:00 64gb  02:00:00
03min   03:00:00 12:00:00 1 -  10:00:00 96gb  03:00:00
01min   06:00:00 24:00:00 2 -  24:00:00 192gb 08:00:00
LIST
qstat -u $USER | tail -n +6
