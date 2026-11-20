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
- `macondo_patch/lexmem_accessor.go` -> `ai/bot/`: exposes the bot's Monte Carlo simmer (needed for `CANDS`).

Build: `go build -o bin/<name> ./cmd/<name>` for lexmem_bot, lexmem_selfplay, lexmem_plan, lexmem_search (Go 1.23).
