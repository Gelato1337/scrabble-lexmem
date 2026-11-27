#!/bin/bash
#SBATCH --account=project_462001611
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 --mem=120G
#SBATCH --time=02:00:00
#SBATCH --job-name=net-check
#SBATCH --output=logs/net_check_%j.out
# Existing policy nets with the paired measures, CPU only: agreement with the teacher on NAGREE held-out positions of simsoft_31
# (eval/policy_agree.py, the reference first) and NG games each alone against the fixed HastyBot on seeds 777000
# (lumi/bignet_games.sh; a net whose result file exists is not played again).
# Env: REF NEW (checkpoint names without policy_ / .pt; NEW may hold several, space-separated) NAGREE NG.
source lumi/common.sh
export MACONDO=/scratch/project_462001611/macondo PYTHONPATH=.
set -o pipefail
REF=${REF:-simsoft_scratch_nolex}; NEW=${NEW:?set NEW}
OMP_NUM_THREADS=16 TEST=simsoft_31 N=${NAGREE:-5000} NWORK=60 OUT=data/agree_check_$SLURM_JOB_ID.pkl timeout 60m python eval/policy_agree.py \
  data/policy_$REF.pt $(for c in $NEW; do echo data/policy_$c.pt; done) 2>&1 | grep -v "Warning\|warnings.warn" || exit 1
REF=$REF NEW="$NEW" bash lumi/bignet_games.sh 2>&1 | grep -v "Warning\|warnings.warn\|Singularity\|python3, pip3\|installed by CSC\|docs.csc.fi\|^$" || exit 1
echo "=== NET CHECK DONE"
