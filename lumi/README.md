# LUMI notes

- Project `project_462001611`; working copy `/scratch/project_462001611/scrabble-lexmem` (data in `data/`, logs in `logs/`),
  Macondo checkout `/scratch/project_462001611/macondo` (`bin/shell`, `bin/lexmem_bot`, `bin/lexmem_selfplay`), Go 1.23 in
  `/scratch/project_462001611/go`, ONNX Runtime 1.29 in `/scratch/project_462001611/ort`.
- Every job sources `lumi/common.sh`: `module load pytorch/2.7` (Singularity, ROCm), `PYTHONPATH` = repo root.
- Submit: `sbatch --export=ALL,KEY=VALUE lumi/<job>.sh`. Values containing commas (e.g. `TRAIN=a,b`) must be set in the
  environment instead (`TRAIN=a,b sbatch --export=ALL,... lumi/imitate.sh`): `--export` splits on commas.
- Job scripts print `|| exit 1` failures so dependency chains (`--dependency=afterok:`) stop instead of running on missing files.
- The pytorch container can take minutes to start on some nodes; stdlib-only jobs (`xt_crawl.sh`) use `/usr/bin/python3`.
- Requeued jobs can come back as "held" after a node failure: `scontrol release <jobid>`.
- A node with a stale ROCm cache file owned by another user (`/tmp/gfx90a*.ufdb.txt`) kills GPU jobs at start: resubmit with `--exclude=<node>`.

| Script | What it runs |
|---|---|
| `imitate.sh`, `play_eval.sh` | imitation / distillation training (GPU); games vs a Macondo bot (CPU) |
| `label_bot.sh` | teacher labels (array over game chunks; `CANDS`, `MAX_BAG`, `G_OFF`) |
| `gcg.sh`, `xt_crawl.sh`, `replay.sh`, `hasty_gen.sh` | human games, crawl, HastyBot game replay / generation |
| `td_rounds.sh` (+ `td_gen.sh`, `td_train.sh`) | self-play Track A v1 (Q / TD) |
| `v2_rounds.sh` (+ `v2_gen.sh`, `v2_train.sh`, `v2_eval.sh`) | self-play Track A v2 with gating |
| `sp_fast.sh`, `sp_branch_gen.sh` | Go engine + GPU server (throughput, parity, branch-pair generation) |
| `round*.sh`, `selfplay_gen.sh`, `label_sim.sh`, `train_*.sh` | earlier self-play / expert-iteration rounds and value / future-word heads |
| `feas2*.sh`, `feas4.sh`, `smoke.sh`, `coverage.sh`, `gen.sh` | feasibility experiments |

## History (job IDs)
- Project `project_462001611`; code + data at `/scratch/project_462001611/scrabble-lexmem` (logs in `logs/`, results/checkpoints in `data/`).
- Env: `module use /appl/local/csc/modulefiles; module load pytorch/2.7` (Singularity container, ROCm). Batch scripts in `lumi/`:
  `smoke.sh` (dev-g, 20 min), `gen.sh` (CPU, 4000+300 games -> data/games_big*.pkl), `feas2.sh` (SLOT/TAG via --export), `feas4.sh`.
- Submit: `sbatch --export=ALL,SLOT=0,TAG=noslot lumi/feas2.sh`. Check: `squeue -u $USER`, `tail logs/*.out`.
- Overnight 2026-09-26 -> 27: gen 22372459; feas2 no-slot 22372477; feas2 slot 22372478; feas4 latent 22372479 (all afterok gen).

## Macondo / HastyBot integration (2026-09-27)
- Macondo built from source (Go 1.23 in user space): LUMI `/scratch/project_462001611/macondo/bin/shell`, local `~/opt/macondo`.
  Lexica + leave values fetched with `go run ./cmd/fetch_lexica -data-path ./data` (NWL23, CSW24, ...).
- Lexicon switched to **NWL23** for anything involving Macondo: `data/NWL23.txt` extracted from the KWG by `lexmem/kwg.py` (196,601 playable words).
  Scripts take `LEXICON=data/NWL23.txt`.
- `lexmem/replay.py`, `replay_all.py`: Macondo autoplay logs -> our position format; every play re-scored and checked (0 mismatches).
- `harness.py`: our engine vs HastyBot (Macondo shell subprocess, CGP in / top-equity play out); HastyBot plays validated under our rules.
  Baseline: score-greedy bot vs HastyBot, 100 games: **35 % wins, −34 spread**.
- `lexmem/policy.py`, `imitate.py`: policy/value net over legal moves, `USE_LEX=1` adds the retrieval branch (rack-level 7/8-letter
  queries, 32 tokens each, keys frozen from the step-2 model). Imitation of HastyBot + final-spread value, then PLAY games vs HastyBot.
- LUMI chain (2026-09-27): hasty-gen 22376005 (100k games) -> replay 22376032 -> feas2 on NWL23 22376033 -> imitation ablation
  imit-nolex 22376045 / imit-lex 22376046 (each ends with 300 games vs HastyBot). Results: `data/result_imit_*.pkl`, `logs/imitate_*.out`.
  First result: no-lexicon imitation net = HastyBot: 50.3 % ± 1.1 over 2000 games (`play_eval.py`).

## Self-play loop (2026-09-27 evening)
- `lexmem/search.py`: 1-ply expectimax. Top-K policy candidates, each applied exactly, scored by the position-only value head
  from the opponent's view over S sampled opponent racks (K=6, S=6 default). `play_eval.py SEARCH=1` evaluates with it.
- `selfplay.py`: net vs net with search, 60 CPU procs (~15 s/game/proc), exploration = policy-temperature moves on turns 0-1 and 10 % of turns
  (masked out of the policy loss via `explore`). Output = our trajectory format -> `imitate.py` trains on it with `INIT=<prev ckpt>`.
- `lumi/round.sh <R> <nolex|lex> <init ckpt> [dep]`: submits gen (CPU) -> train (GPU, 8k steps) -> eval with/without search (CPU, 2000 games each).
- Pipeline submitted 2026-09-27: imit2 (imitation + pos-only value head) 22384208/22384209 -> evals 22384254-57 -> round 1: nolex 22384258-61, lex 22384266-69.
- Results so far: imitation net (no search) = HastyBot, 50.3 % ± 1.1 (2000 games).


### Macondo / HastyBot integration (2026-09-27)
- Macondo built from source (Go 1.23 in user space): LUMI `/scratch/project_462001611/macondo/bin/shell`, local `~/opt/macondo`.
  Lexica + leave values fetched with `go run ./cmd/fetch_lexica -data-path ./data` (NWL23, CSW24, ...).
- Lexicon switched to **NWL23** for anything involving Macondo: `data/NWL23.txt` extracted from the KWG by `lexmem/kwg.py` (196,601 playable words).
  Scripts take `LEXICON=data/NWL23.txt`.
- `lexmem/replay.py`, `replay_all.py`: Macondo autoplay logs -> our position format; every play re-scored and checked (0 mismatches).
- `harness.py`: our engine vs HastyBot (Macondo shell subprocess, CGP in / top-equity play out); HastyBot plays validated under our rules.
  Baseline: score-greedy bot vs HastyBot, 100 games: **35 % wins, −34 spread**.
- `lexmem/policy.py`, `imitate.py`: policy/value net over legal moves, `USE_LEX=1` adds the retrieval branch (rack-level 7/8-letter
  queries, 32 tokens each, keys frozen from the step-2 model). Imitation of HastyBot + final-spread value, then PLAY games vs HastyBot.
- LUMI chain (2026-09-27): hasty-gen 22376005 (100k games) -> replay 22376032 -> feas2 on NWL23 22376033 -> imitation ablation
  imit-nolex 22376045 / imit-lex 22376046 (each ends with 300 games vs HastyBot). Results: `data/result_imit_*.pkl`, `logs/imitate_*.out`.
  First result: no-lexicon imitation net = HastyBot: 50.3 % ± 1.1 over 2000 games (`play_eval.py`).

### Self-play loop (2026-09-27 evening)
- `lexmem/search.py`: 1-ply expectimax. Top-K policy candidates, each applied exactly, scored by the position-only value head
  from the opponent's view over S sampled opponent racks (K=6, S=6 default). `play_eval.py SEARCH=1` evaluates with it.
- `selfplay.py`: net vs net with search, 60 CPU procs (~15 s/game/proc), exploration = policy-temperature moves on turns 0-1 and 10 % of turns
  (masked out of the policy loss via `explore`). Output = our trajectory format -> `imitate.py` trains on it with `INIT=<prev ckpt>`.
- `lumi/round.sh <R> <nolex|lex> <init ckpt> [dep]`: submits gen (CPU) -> train (GPU, 8k steps) -> eval with/without search (CPU, 2000 games each).
- Pipeline submitted 2026-09-27: imit2 (imitation + pos-only value head) 22384208/22384209 -> evals 22384254-57 -> round 1: nolex 22384258-61, lex 22384266-69.
- Results so far: imitation net (no search) = HastyBot, 50.3 % ± 1.1 (2000 games).

