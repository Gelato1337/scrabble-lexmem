#!/bin/bash
# Several lookups in a row on the three-turn planning benchmark (train_lexvalue.py HOPS), all training positions, seeds 0-1.
# Settings exactly as the round-7 three-turn all-data runs (logs/submit_planfix.sh: lexplanfix_h3ft_*_all_s*), plus HOPS=2:
#   lex2     memory, two lookups (the second with a query formed after the first lookups of all lanes), 3 more layers
#   lex1deep memory, one lookup, the same 3 more layers and query update without the second retrieval (control for the lookup)
# First a short GPU job writes the per-position rows of the four round-7 checkpoints (one lookup: memory and same-start control;
# its printed lines must equal those runs' stored last evaluations). Last, eval/plan_pairs.py compares the arms by position.
# Usage (LUMI login node): bash lumi/submit_hops.sh [ARMS]     ARMS default "lex2 lex1deep"; "nolex2" = the control, two passes
cd /scratch/project_462001611/scrabble-lexmem
A="-A project_462001611"; X="--exclude=nid005093,nid005114"
G="$A -p small-g --gpus-per-node=1 --cpus-per-task=32 --mem=160G $X"
P="HORIZON=3 DATA=data/plan_h3.pkl QINIT=fut_train NO_XCHK=1 NWORK=28 NEV_POS=3000 HOLDOUT_MOD=10"
OLD=lexplanfix_h3ft
R=$(sbatch --parsable $G -t 00:30:00 -J hop-rows -o logs/hop_rows_%j.out --wrap="source lumi/common.sh
for a in lex nolex; do
  grep -h '^\\[step 40000\\]' logs/${OLD}_\${a}_all_s0_*.out logs/${OLD}_\${a}_all_s1_*.out
  env $P ARM=\$a EVAL_CKPTS=data/${OLD}_\${a}_all_s0.pt,data/${OLD}_\${a}_all_s1.pt python train/train_lexplan.py | grep -E '^\\[' || exit 1
done")
echo "rows $R"; DEP=$R; TAGS=""
for arm in ${1:-lex2 lex1deep}; do
  case $arm in lex2) E="ARM=lex HOPS=2";; lex1deep) E="ARM=lex HOPS=2 HOP_LEX=0";; nolex2) E="ARM=nolex HOPS=2";; *) echo "unknown arm $arm"; exit 1;; esac
  for s in 0 1; do
    t=lexplanhop_h3_${arm}_all_s$s
    j=$(sbatch --parsable $G -t 03:00:00 -J hop_${arm}_s$s -o logs/${t}_%j.out --wrap="source lumi/common.sh; export $P $E EVAL_EVERY=4000 SEED=$s MAX_TRAIN=0 STEPS=40000 B=64 TAG=$t; python train/train_lexplan.py || exit 1")
    echo "$t $j"; DEP=$DEP:$j
  done
  TAGS="$TAGS ${t%_s1}_s0,${t%_s1}_s1"
done
J=$(sbatch --parsable $A -p small --cpus-per-task=1 --mem=4G -t 00:15:00 --dependency=afterok:$DEP -J hop-pairs -o logs/hop_pairs_%j.out --wrap="source lumi/common.sh
python eval/plan_pairs.py ${OLD}_lex_all_s0,${OLD}_lex_all_s1 ${OLD}_nolex_all_s0,${OLD}_nolex_all_s1 $TAGS || exit 1
for s in 0 1; do echo \"--- seed \$s\"; python eval/plan_pairs.py ${OLD}_lex_all_s\$s ${OLD}_nolex_all_s\$s $(for a in ${1:-lex2 lex1deep}; do printf 'lexplanhop_h3_%s_all_s$s ' $a; done) || exit 1; done")
echo "pairs $J"
