#!/bin/bash
# Round-5 expert iteration, one arm, one round, as a chain of jobs (submission only: run on the login node).
#   bash lumi/ei_round5.sh <round> <nolex|lex> <init ckpt>
# 1 gen   : NG self-play games of the net against itself (argmax, no exchanges), every turn recorded (lexmem_search TRAJ=1);
#           then NPOS positions sampled (at most PER per game, bag >= 8, no openings)            -> data/searchpos_r5sp<R>_<V>.jsonl
# 2 label : NSPLIT jobs, each searching NPOS/NSPLIT positions with the net (K=8, S worlds, D=2, PM=64, mac leaf)
# 3 train : merge the labels into training games (soft targets = the search's win %) and fine-tune the net from <init> with
#           train/imitate.py EI=1 (TAU, MARGIN, BETA, LR, STEPS as the earlier EI rounds, lumi/round_ei.sh)  -> data/policy_r5ei<R>_<V>.pt
# 4 eval  : the new net alone and with the Macondo handoff vs HastyBot, 2000 games on seeds 777000 (paired with b0_<V>, b0h_<V>)
# Env (defaults): NG=6000 PER=5 NPOS=30000 NSPLIT=4 S=256 LAB_TIME=08:00:00 STEPS=4000 TAU=2 MARGIN=3 BETA=0.5 LR=1e-4
set -e
R=$1; V=$2; INIT=$3
cd /scratch/project_462001611/scrabble-lexmem
D=/scratch/project_462001611/scrabble-lexmem/data
NG=${NG:-6000}; PER=${PER:-5}; NPOS=${NPOS:-30000}; NSPLIT=${NSPLIT:-4}; S=${S:-256}
UL=0; RETR=""; [ "$V" = "lex" ] && UL=1 && RETR=data/lexmem_nwl.pt
SP=r5sp${R}_${V}; ET=r5ei${R}_${V}
sub() { sbatch --parsable --export=ALL "$@"; }
export CKPT=$INIT PRE= POST="python selfplay/ei_data.py positions $SP $NPOS $PER"
export RUNS="$SP|OURS=net OPP=net EXCH=0 NGAMES=$NG CONC=64 SEED=$((900000 + R * 20000)) TRAJ=1"
G=$(sub -t 03:00:00 -J r5ei$R-gen-$V -o logs/r5ei$R-gen-${V}_%j.out lumi/search_eval.sh)
LABS=""; DEPS=""; PER_JOB=$(( (NPOS + NSPLIT - 1) / NSPLIT ))
for q in $(seq 0 $((NSPLIT - 1))); do
  export POST= RUNS="r5lab${R}_${V}_$q|POSITIONS=$D/searchpos_$SP.jsonl POS0=$((q * PER_JOB)) NPOS=$PER_JOB OURS=search K=8 S=$S D=2 PM=64 LEAF=mac EXCH=0 CONC=64 PAR=4096 SEED=777000"
  L=$(sub --dependency=afterok:$G -t ${LAB_TIME:-08:00:00} -J r5ei$R-lab$q-$V -o logs/r5ei$R-lab$q-${V}_%j.out lumi/search_eval.sh)
  LABS="$LABS r5lab${R}_${V}_$q"; DEPS="$DEPS:$L"
done
export PRE="python selfplay/ei_data.py merge $SP $ET$LABS && python selfplay/ei_data.py check $ET $UL"
export USE_LEX=$UL RETR=$RETR INIT=$INIT TRAIN=$ET TEST=simsoft_31 TAG=$ET EI=1 KPLY=4 PLAY=0 STEPS=${STEPS:-4000} TAU=${TAU:-2} MARGIN=${MARGIN:-3} BETA=${BETA:-0.5} LR=${LR:-1e-4}
T=$(sub --dependency=afterok${DEPS} -t 06:00:00 -J r5ei$R-train-$V -o logs/r5ei$R-train-${V}_%j.out lumi/imitate.sh)
unset USE_LEX RETR INIT TRAIN TEST TAG EI KPLY PLAY STEPS TAU MARGIN BETA LR
B="EXCH=0 NGAMES=2000 CONC=64 SEED=777000"
export CKPT=data/policy_$ET.pt PRE= POST= RUNS="e${R}_${V}|OURS=net $B;e${R}h_${V}|OURS=net HANDOFF=8 EG_SECS=10 NBOT=14 $B"
E=$(sub --dependency=afterok:$T -t 03:00:00 -J r5ei$R-eval-$V -o logs/r5ei$R-eval-${V}_%j.out lumi/search_eval.sh)
echo "r5 EI round $R $V: gen $G, label${DEPS//:/ }, train $T, eval $E"
