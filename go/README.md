# Go sources (build inside a Macondo checkout)

These live in our Macondo checkout on LUMI (`/scratch/project_462001611/macondo`), not in this module:

- `lexmem_bot/main.go` -> `cmd/lexmem_bot/`: plays any Macondo bot code; CGP per line in, `MOVE <play>` out (`CANDS=K` adds the sim's top-K candidates with win %). Used by `harness.py` (`MacondoBot`) and `label_bot.py`.
  Variants of the elite bots: `MIN_SIM_PLIES=n` (Macondo's MinSimPlies), `PEG_UNSEEN=n` with `PEG_SECS` / `PEG_THREADS` (Macondo's
  pre-endgame solver from n unseen tiles on instead of the simulation: 9 = two tiles in the bag; `ai/bot` `LexmemBestPlay`).
- `lexmem_selfplay/main.go` -> `cmd/lexmem_selfplay/`: fast self-play; Macondo plays the rules, our net (served by `server_infer.py` on a GPU) picks the moves; optional shared-draw branch pairs.
- `lexmem_search/main.go` -> `cmd/lexmem_search/`: whole games in Go vs a Macondo bot (HastyBot in-process on the real game state,
  as in the autoplay games; `OPPRACK=throw` or bot code `HASTY_BOT_THROW` = the pre-2026-11-24 variant with our rack in its bag;
  other bots in a pool of `lexmem_bot` processes); our side = the net, a search over the net (top-K candidates x S shared worlds x D plies of net play,
  win-% leaf), or a Macondo bot; optional Macondo handoff for the end of the game. Run by `lumi/search_eval.sh`; compare runs
  with `tools/search_games.py`. `POSITIONS=file.jsonl`: search single positions (CGPs) instead and write every candidate's
  playout value in every world (`eval/search_positions.py` prepares the positions and scores the choices against the teacher).
  `START=file.jsonl`: each position of the file (both racks, our side on turn) is played to the end of the game instead of whole
  games (`eval/late_game.py`, `lumi/late_game.sh`); `BOT_ERR=file` keeps the bot processes' stderr.
  `OPP=net TRAJ=1`: self-play of the net against itself with every turn recorded (`selfplay/ei_data.py` turns games + search
  labels into expert-iteration training data; `lumi/ei_round5.sh` runs one round). `LEAF=win`: our own leaf, the win
  probability of a separate value net (`train/value_leaf.py`, trained on the search's labels; served with `VCKPT=` and
  `VHEAD=td`) instead of Macondo's win-% table.
- `lexmem_plan/main.go` -> `cmd/lexmem_plan/`: exact planning labels by enumeration over the dictionary (one turn; `H2=1` two
  turns, `H3=1` three turns; `CHECK=1` verifies against full enumeration). Run by `lumi/plan_label.sh`, merged by `prep/plan_positions.py`.
  Optional input `okeep` (one turn): the opponent's kept tiles per sample (+ a refill) instead of a uniform rack; `SEED0` shifts the
  sample seeds (`eval/plan_infer.py`, `lumi/plan_infer.sh`).
- `lexmem_infer/main.go` -> `cmd/lexmem_infer/`: Macondo's rack inference (rangefinder) on positions from `eval/infer_macondo.py`,
  per-tile marginals of the opponent's kept tiles: the reference for the rack-inference benchmark (`train/train_lexinfer.py`).
  Diagnostics: `PROPOSAL=prior|floor`, `TAU`.
- `lexmem_invert/main.go` -> `cmd/lexmem_invert/`: exact rack inference against a static HastyBot: every possible kept set is
  checked by HastyBot's own move choice (likelihood 1/t when the recorded play is one of t moves tied at the top equity);
  posterior marginals, whether the true kept set reproduces the play, and with `SAMPLES=k` k posterior samples
  (`lumi/infer_exact.sh`). `OPPRACK=keep` (default) leaves our rack as HastyBot's opponent rack, as in the Macondo autoplay
  games; `throw` puts it in the bag (as `lexmem_search` and the harness did before 2026-11-24).
- `lexmem_track/main.go` -> `cmd/lexmem_track/`: exact tracking of the opponent's rack over the whole move history against a
  static HastyBot: after every move of the opponent a weight for each kept set it can hold; its draw, our own draws and its
  next move (HastyBot's own choice with every rack it can hold) update the weights. Plays use `lexmem_invert`'s likelihood
  (1/t among top-equity ties); an exchange shows only the number of tiles. Macondo's top-play generator rejects most racks
  before the full move list is needed (`FAST=0` switches that off). `HIST=k` uses the last k moves only (`HIST=1` =
  `lexmem_invert`, except after our own exchange on the first turn: see the file header). Output as `lexmem_invert` plus tracking diagnostics (`eval/track_infer.py`, `lumi/track_infer.sh`).
- `lexmem_simrack/main.go` -> `cmd/lexmem_simrack/`: SIMMING_BOT's own simulation of a position with the opponent's rack from a
  chosen source: uniform (the bot as it is), kept-tile samples of an inference method, or the true kept tiles (`okeep`, the
  input files of `eval/plan_infer.py`); `MODE=eval` scores given plays with a fixed number of iterations and no pruning. `PLIES=n` simulates n plies after the
  candidate play (4, 6; 99 = to the end of the game) where the bot would simulate fewer (`lumi/sim_depth.sh`).
  Position-level test of rack modelling inside the simulator (`eval/sim_infer.py`, `lumi/sim_infer.sh`).
- `macondo_patch/lexmem_accessor.go` -> `ai/bot/`: exposes the bot's Monte Carlo simmer (needed for `CANDS`), and `LexmemSim`:
  the simulation set-up of `nonEndgameBest` with optional inferred opponent leaves (for `lexmem_simrack`); and `LexmemBestPlay`: BestPlay
  with the pre-endgame solver starting at a chosen number of unseen tiles (for `lexmem_bot` `PEG_UNSEEN`).

Build: `go build -o bin/<name> ./cmd/<name>` for lexmem_bot, lexmem_selfplay, lexmem_plan, lexmem_search, lexmem_infer, lexmem_invert, lexmem_track, lexmem_simrack (Go 1.23).
