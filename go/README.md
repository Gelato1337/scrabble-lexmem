# Go sources (build inside a Macondo checkout)

These live in our Macondo checkout on LUMI (`/scratch/project_462001611/macondo`), not in this module:

- `lexmem_bot/main.go` -> `cmd/lexmem_bot/`: plays any Macondo bot code; CGP per line in, `MOVE <play>` out (`CANDS=K` adds the sim's top-K candidates with win %). Used by `harness.py` (`MacondoBot`) and `label_bot.py`.
- `lexmem_selfplay/main.go` -> `cmd/lexmem_selfplay/`: fast self-play; Macondo plays the rules, our net (served by `server_infer.py` on a GPU) picks the moves; optional shared-draw branch pairs.
- `macondo_patch/lexmem_accessor.go` -> `ai/bot/`: exposes the bot's Monte Carlo simmer (needed for `CANDS`).

Build: `go build -o bin/lexmem_bot ./cmd/lexmem_bot` and `go build -o bin/lexmem_selfplay ./cmd/lexmem_selfplay` (Go 1.23).
