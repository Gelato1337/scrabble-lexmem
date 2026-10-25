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

## Lexicon memory: controlled tests (round 1)

All on Go-engine self-play games of `v2_w_g1`; held out by game; the lex and no-lex arms have the same data, steps and
(near-)equal trainable parameters. See [round1_summary.md](round1_summary.md).

**Branch pairs** (which of two moves wins its shared-draw playout; 251,216 held-out pairs; 0.5 = no signal):

| Net | Zero-shot | After 1 pass over 2.27M pairs |
|---|---|---|
| `simsoft_scratch_nolex` (50k subsample) | 0.605 (late 0.575) | 0.601–0.604 during training |
| `simsoft_scratch_lex` | 0.608 all / 0.607 subsample (late 0.584) | 0.603 |
| `v2_w_g1` (no-lex) | 0.609 | 0.606 |

**Value of the afterstate** (`train/train_lexvalue.py`; 12 lexicon-free lanes x 2 sides; lex = retrieval over each side's exact
rules mask, top 256 words; no-lex = the same query without retrieval):

| Targets | Training positions | Seeds | R² lex / no-lex | Win log-loss lex / no-lex |
|---|---|---|---|---|
| one game's result | 5.85M (250k games) | 1 | — (MAE 52.1 / 52.5 pts) | 0.510 / 0.528 |
| mean of 16 rollouts | 177k | 3 | 0.230 / 0.236 | 0.4372 / 0.4358 |
| rollouts, frozen query encoder | 10k | 2 | 0.117 / 0.129 | 0.4485 / 0.4493 |
| rollouts, frozen query encoder | 50k | 2 | 0.247 / 0.232 | 0.4378 / 0.4391 |
| rollouts, frozen query encoder | 177k | 2 | 0.257 / 0.266 | 0.4336 / 0.4327 |

Seed-to-seed spread of R² ≈ 0.01–0.04. Opponent next word in the top 8 of its lane (lex arm): 0.42–0.44 frozen, falling to
~0.32 when the query encoder trains with the value loss.

Rollout targets: `go/lexmem_selfplay` with `P_RO=0.1 K_RO=16` — at 10 % of positions, 16 continuations of the chosen move with
a reshuffled bag and the opponent's rack redrawn from the unseen tiles, played to the end by the policy (80k games, 197k labelled
positions, 1.5 h on 8 GPUs).

## Planning benchmark (round 2a, 2026-10-22 – 25)

300k positions from Go self-play (`spf_0`), top 16 plays each, labelled by enumeration (`go/lexmem_plan`, 256 shared tile samples,
exact best-score move generation over NWL23): E_opp = opponent's expected best reply, E_my = my expected best next play (leave +
refill, same board). PV = score − E_opp + E_my. Planning-critical: the PV-best move beats the top-scoring move by ≥ 5 points (743
of 3000 held-out positions). Model: the stage-2 afterstate net, query encoder frozen in both arms; mean of 2 seeds.

| Training positions | Arm | R² E_opp (within position) | R² E_my (within position) | Decision acc / regret | Critical acc / regret |
|---|---|---|---|---|---|
| 10k | no-lex | 0.673 (0.003) | 0.740 (0.345) | 0.584 / 1.87 | 0.375 / 4.63 |
| 10k | **lex** | **0.735 (0.201)** | **0.774 (0.429)** | 0.582 / 1.87 | 0.387 / 4.48 |
| 40k | no-lex | 0.738 (0.099) | 0.786 (0.419) | 0.588 / 1.82 | 0.411 / 4.42 |
| 40k | **lex** | **0.764 (0.206)** | **0.807 (0.476)** | 0.593 / 1.73 | 0.415 / 4.10 |
| 270k | no-lex | 0.778 (0.265) | 0.818 (0.489) | 0.593 / 1.71 | 0.426 / 4.06 |
| 270k | **lex** | **0.798 (0.294)** | **0.830 (0.521)** | 0.601 / 1.66 | 0.448 / 3.89 |
| — | top-score baseline | | | 0.488 / 3.21 | — / 10.51 |

Lex > no-lex on every R² in every seed pair; the gap is largest with the least data (within-position E_opp 0.20 vs 0.00 at 10k)
and narrows to +0.03 at 270k. Decision accuracy / regret move in the same direction but by less than the seed spread.
