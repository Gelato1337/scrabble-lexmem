# Go sources (build inside a Macondo checkout)

These live in our Macondo checkout on LUMI (`/scratch/project_462001611/macondo`), not in this module:

- `lexmem_bot/main.go` -> `cmd/lexmem_bot/`: plays any Macondo bot code; CGP per line in, `MOVE <play>` out (`CANDS=K` adds the sim's top-K candidates with win %). Used by `harness.py` (`MacondoBot`) and `label_bot.py`.
- `lexmem_selfplay/main.go` -> `cmd/lexmem_selfplay/`: fast self-play; Macondo plays the rules, our net (served by `server_infer.py` on a GPU) picks the moves; optional shared-draw branch pairs.
- `lexmem_search/main.go` -> `cmd/lexmem_search/`: whole games in Go vs a Macondo bot (HastyBot in-process, other bots in a pool of
  `lexmem_bot` processes); our side = the net, a search over the net (top-K candidates x S shared worlds x D plies of net play,
  win-% leaf), or a Macondo bot; optional Macondo handoff for the end of the game. Run by `lumi/search_eval.sh`; compare runs
  with `tools/search_games.py`. `POSITIONS=file.jsonl`: search single positions (CGPs) instead and write every candidate's
  playout value in every world (`eval/search_positions.py` prepares the positions and scores the choices against the teacher).
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
  checked by HastyBot's own move choice; posterior marginals, whether the true kept set reproduces the play, and with `SAMPLES=k`
  k posterior samples (`lumi/infer_exact.sh`).
- `macondo_patch/lexmem_accessor.go` -> `ai/bot/`: exposes the bot's Monte Carlo simmer (needed for `CANDS`).

Build: `go build -o bin/<name> ./cmd/<name>` for lexmem_bot, lexmem_selfplay, lexmem_plan, lexmem_search, lexmem_infer, lexmem_invert (Go 1.23).
