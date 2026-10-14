# Benchmarks

Every number below is from complete games in our harness (`harness.py`, `play_eval.py`): NWL23, each game with its own
seeded bag, the first seat alternating. **Win rate** = share of games won (± standard error); **spread** = mean final score
difference per game. Seed sets: `777000` = fresh seeds (rows on these seeds are paired: identical bags); `10000` = the
original seeds. All runs on LUMI, project_462001611.

## Opponents

| Player | Method | vs HastyBot | Games, seeds |
|---|---|---|---|
| NO_LEAVE_BOT (Macondo) | highest-scoring move, no leave value | — | — |
| HastyBot (Macondo) | static equity (score + leave value) | 50 % by definition | — |
| SIMMING_BOT (Macondo), 10 s/move, 1 thread | Monte Carlo sim + endgame/pre-endgame solvers, no rack inference | **62.6 % ± 1.1**, spread +19.5 | 2000, 777000 |

## Nets alone (one forward pass per move, no search)

| Net (checkpoint `data/policy_*.pt`) | Training targets | vs HastyBot | vs SIMMING_BOT 10 s (1000 games, 10000) |
|---|---|---|---|
| `imit3_nolex` | HastyBot's moves (30k games) | 51.2 % ± 1.1 (2000, 10000) | 37.9 % |
| `human_nolex` | human annotators' moves (27k cross-tables games) | 48.5 % (2000, 10000) | 35.5 % |
| `humhasty_nolex` | human + HastyBot games | 49.5 % (2000, 10000) | 37.1 % |
| `simbot_ft_nolex` | SIMMING_BOT's chosen move, 160k positions, fine-tuned from imit3 | 49.8 % (2000, 10000) | 38.0 % |
| `simbot_scratch_nolex` | same, from scratch | 50.3 % (2000, 10000) | 38.7 % |
| `simsoft_ft_nolex` | SIMMING_BOT top-16 win % as soft targets, 715k positions, from imit3 | 52.3 % (2000, 10000) | 39.2 % |
| **`simsoft_scratch_nolex`** | same, from scratch | **52.8 %** (2000, 10000); **53.2 % ± 0.8** (4000, 777000) | 38.6 % |
| `simsoft_scratch_lex` | same + lexicon branch | 51.0 % (2000, 10000) | 37.8 % |
| `simsoft_big_nolex` | soft targets, 2.0M positions, 80k steps | 52.3 % (2000, 10000) | 39.0 % |
| `simsoft_exch_nolex` | simsoft_scratch + exchange tokens | 52.4 % (4000, 777000) | 39.6 % |
| `c1_costw` | + cost-weighted loss | 53.1 % (4000, 777000) | — |
| `c2_deep_costw` | + 30 s teacher labels for positions with ≤ 30 tiles in the bag | 52.8 % (4000, 777000) | — |

vs NO_LEAVE_BOT (1000 games, 10000): `imit3_nolex` 65.3 %, `human_nolex` 61.4 %, `humhasty_nolex` 62.6 %.

## Net + Macondo at the end of the game (`play_eval.py ENDGAME=k`)

The net plays until the unseen-tile count drops to 6 + k; SIMMING_BOT (10 s) plays the rest. 4000 games, seeds 777000.

| Net | Alone | Bag empty (k=1) | ≤ 1 tile in bag (k=2) | ≤ 7 tiles in bag (k=8) |
|---|---|---|---|---|
| `simsoft_scratch_nolex` | 53.2 % | 54.9 % | 55.2 % | **58.1 %** (58.1 % with a 30 s solver) |
| `c1_costw` | 53.1 % | — | — | 57.9 % |
| `c2_deep_costw` | 52.8 % | — | — | 57.2 % |

Our Python rollout search on top of the net (net's top 6 candidates, 24 + 96 worlds, endgame solver), 2000 games, 777000:
net rollouts 52.6 %, HastyBot rollouts 50.4 %.

## Self-play (only game results as signal), vs HastyBot

Track A v1 (Q-learning, act by argmax Q), 2000 games per round, 777000:

| Round | Warm start (Q among policy top 8) | From scratch |
|---|---|---|
| 1 | 3.4 % | 0 % (spread −444) |
| 2 | 23.6 % | 0 % (−426) |
| 3 | 19.6 % | 0 % (−382) |
| 4 | 3.2 % | 0 % (−398) |

Track A v2 (policy plays; TD value, advantage-weighted policy updates, shared-draw branch pairs, KL limit, gate), 4000 games,
777000, start `simsoft_exch_nolex` (52.4 %): round 1 52.5 % (accepted), rounds 2–4 51.0 / 50.9 / 50.9 % (rejected);
pooled rounds 2–4 51.8 %; branch-pairs-only training 52.0 %.

Held-out branch pairs (15,060): share where the net's preferred move won its playout — start 0.586 (pre-endgame 0.740);
after branch-only training 0.582 (pre-endgame 0.734): no learning beyond the start net.

## Diagnostics

| Quantity | Value |
|---|---|
| Teacher self-agreement (SIMMING_BOT 10 s, same position twice, 4433 positions) | 0.872 (bag empty 0.992, 1–7 in bag 0.735) |
| Net–teacher agreement, held out | 58.7 % (HastyBot imitation) → 61.2 % (soft, 730k) → 63.7 % (soft, 2.0M) |
| Cost of the net's moves in teacher win % (22,868 held-out positions) | 0.32 / 0.34 / 0.60 / 3.47 per move for opening / middle / late / pre-endgame; pre-endgame = 8 % of positions, 41 % of the total cost |
| Branch-pair noise, sd of dU (win + spread/400) | Python whole-game playouts 0.58; Python per-player draw priorities 0.53; Go, paired bag, 3-turn horizon + value 0.33 |

## Engines

| Engine | Throughput |
|---|---|
| Python self-play (`selfplay_td.py`), net on CPU | ~15 s per game per core (~240 games/h/core) |
| Go engine (`go/lexmem_selfplay`) + GPU server (`server_infer.py`), 1 GCD + 7 cores, 1024 concurrent games | 525 positions/s ≈ 76k games/h |
| same, 1 GCD + 28 cores | 593 positions/s (GPU server is the bottleneck) |
| Value net in Go via ONNX Runtime 1.29, 1 core | 650 µs/position (PyTorch 635 µs); outputs equal to 4.8e-7 |

Parity Go vs Python (300 positions): move scores identical; 99.6 % / 98.8 % move-set coverage; logit correlation ≥ 0.9993.
A Python scoring bug for blank plays (premiums and cross words ignored) was found this way and fixed (`lexmem/core.py`
`blank_penalty`, `designate`); Python self-play before 2026-10-13 used the inflated scores.
