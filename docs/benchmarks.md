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

**Small data (round 2b, 2026-10-26)** — same benchmark, 3 seeds (10k: seeds 0–2), within-position R² and decision regret:

| Training positions | Arm | R² E_opp within | R² E_my within | Regret (all) | Regret (critical) |
|---|---|---|---|---|---|
| 1k | no-lex | −0.055 | 0.286 | 2.11 | 5.50 |
| 1k | **lex** | **0.135** | **0.367** | **2.02** | **5.21** |
| 3k | no-lex | −0.058 | 0.266 | 2.08 | 5.29 |
| 3k | **lex** | **0.159** | **0.414** | **1.87** | **4.58** |
| 10k | no-lex | 0.020 | 0.348 | 1.91 | 4.67 |
| 10k | **lex** | **0.208** | **0.439** | **1.84** | **4.40** |

Without the memory the net cannot rank candidates by the opponent's threat at all until ~40k positions (within-position R² ≤ 0.02);
with it, 1k positions already give 0.13. Decisions follow: critical-position regret is 0.3–0.7 points lower with the memory at
1k–10k positions.

**Listwise ranking loss** (`LOSS=rank`, softmax over each position's candidates on the planning value, τ = 1.5 points, 4 positions
per batch): worse for both arms at every size (regret 2.0–2.6 vs 1.7–2.1 with regression only; within-position R² near or below 0).
The regression targets are the better training signal here; the ranking term as configured destabilises them.

**Planning policies in games (round 2c, 2026-10-27)** — move = the top-16 play by score with the best predicted
score − E_opp + E_my (net) or the exact one (`go/lexmem_plan` live, 256 samples); 4000 games vs HastyBot each, seeds 777000;
nets = seed-0 checkpoints of the planning benchmark (regression loss, frozen query encoder).

| Player | Win rate | Spread |
|---|---|---|
| top score (no planning) | 36.6 % ± 0.8 | −33.8 |
| net planner, 3k positions, no-lex | 38.9 % | −23.7 |
| **net planner, 3k positions, lex** | **43.3 %** | **−15.7** |
| net planner, 10k, no-lex | 42.5 % | −16.6 |
| net planner, 10k, lex | 43.7 % | −14.8 |
| net planner, 270k, no-lex | 43.4 % | −12.7 |
| net planner, 270k, lex | 43.5 % | −12.7 |
| exact planner | 46.7 % | −4.6 |

The planning value is worth +10 points of win rate over top-score play. Trained on 3k positions, the memory planner recovers 66 %
of that gap (43.3 %), the control 23 % (38.9 %); the arms converge by 270k positions.

## Dictionary transfer NWL23 → CSW24 (round 2d, 2026-11-01)

The same 300k positions relabelled under CSW24 (280,887 words; 87,984 not in NWL23, incl. 20 two-letter words), same tool, K, S and
held-out games. Lex arm: word keys rebuilt from the frozen letter encoder over the CSW list; both arms see CSW cross-check sets.
Dictionary-sensitive = top quarter of held-out positions by label shift vs NWL (≥ 8.6 points; 750 of 3000). Top-score baseline:
regret 3.52 all / 11.19 critical (NWL: 3.21 / 10.51).

| Condition | Training positions | Arm | R² E_opp within | R² E_my within | Regret all | Regret critical |
|---|---|---|---|---|---|---|
| zero-shot (trained on NWL) | 1k | no-lex / lex | −0.044 / **0.142** | 0.286 / **0.360** | 2.22 / 2.19 | 5.71 / **5.55** |
| zero-shot | 3k | no-lex / lex | −0.045 / **0.162** | 0.270 / **0.414** | 2.21 / **2.02** | 5.47 / **4.96** |
| zero-shot | 10k | no-lex / lex | 0.021 / **0.197** | 0.352 / **0.439** | 2.08 / **1.98** | 5.08 / **4.70** |
| zero-shot | 40k | no-lex / lex | 0.095 / **0.212** | 0.425 / **0.476** | 1.96 / **1.85** | 4.80 / **4.36** |
| zero-shot | 270k | no-lex / lex | 0.239 / **0.281** | 0.489 / **0.523** | 1.84 / **1.69** | 4.43 / **4.00** |
| zero-shot, dictionary-sensitive | 270k | no-lex / lex | 0.235 / **0.288** | 0.504 / **0.540** | 1.93 / **1.75** | — |
| trained on CSW | 3k | no-lex / lex | −0.048 / **0.165** | 0.306 / **0.422** | 2.06 / 1.99 | 4.97 / **4.85** |
| trained on CSW | 10k | no-lex / lex | −0.103 / **0.197** | 0.315 / **0.446** | 2.08 / 1.96 | 4.95 / **4.61** |
| trained on CSW | 270k | no-lex / lex | 0.198 / **0.271** | 0.510 / **0.536** | 1.74 / 1.68 | 4.05 / **3.81** |
| few-shot (NWL 270k + CSW fine-tune) | 1k | no-lex / lex | 0.202 / 0.250 | 0.478 / 0.470 | 1.83 / 1.84 | 4.23 / 4.25 |
| few-shot | 3k | no-lex / lex | 0.196 / 0.264 | 0.473 / 0.481 | 1.82 / 1.82 | 4.27 / 4.19 |

Both arms transfer with little loss; the memory loses less. (Not because the control reads CSW cross-check sets: removing them
changes neither arm, see the isolation test below.) At 270k NWL positions the critical-regret gap is 0.17 in-domain (NWL) and
0.43 zero-shot on CSW; the NWL-trained memory planner (4.00) matches a no-memory planner trained on 270k CSW positions (4.05).
Fine-tuning on 1k–3k CSW positions (lr 2e-4, full model) erases most of the difference and lowers R² vs zero-shot for both arms.

**Planning policies in CSW games** — the round-2c players, unchanged (seed-0 nets trained on NWL, zero-shot), against HastyBot
playing CSW24; our engine and the lex arm's word keys use the CSW list, the exact planner enumerates CSW24. 4000 games each,
seeds 777000 (± 0.8).

| Player | Win rate CSW24 | Spread | (NWL23, round 2c) |
|---|---|---|---|
| top score (no planning) | 35.6 % | −36.7 | 36.6 % |
| net planner, 3k, no-lex / lex | 40.0 % / 41.4 % | −24.2 / −20.2 | 38.9 % / 43.3 % |
| net planner, 10k, no-lex / lex | 41.6 % / 43.2 % | −19.1 / −16.7 | 42.5 % / 43.7 % |
| net planner, 270k, no-lex / lex | **45.7 %** / 43.7 % | −11.8 / −15.8 | 43.4 % / 43.5 % |
| exact planner | 48.4 % | −4.1 | 46.7 % |

Planning is worth more under CSW (+12.8 points vs +10.1). The benchmark's transfer advantage of the memory does not show up in
games: its lead is +1.4 at 3k and +1.6 at 10k, and at 270k the control wins more (45.7 vs 43.7), although on the CSW benchmark the
same seed-0 nets rank the other way (critical regret 4.38 no-lex vs 4.12 lex). One checkpoint per arm; per-game results of these
runs were not kept (all planner runs wrote the same file, unordered), so the differences cannot be paired. `play_eval.py` now
stores the spread by game index under a file named after the planner and dictionary.

**Isolation: no cross-check sets (2026-11-01)** — `NO_XCHK=1`: the lane tokens' 26 allowed-letter features are set to "any letter
fits" in training and evaluation, both arms; the board CNN has no cross-check input, so the control's only remaining
dictionary-trained part is the frozen query encoder (pretrained on NWL future-word prediction, shared by both arms). Trained on
NWL as in round 2a/2b, seeds 0–1; compared with the cross-check runs of the same seeds (means of 2 seeds).

| Eval | Training positions | Arm | R² E_opp within: with / without | Regret all: with / without | Regret critical: with / without |
|---|---|---|---|---|---|
| NWL | 3k | no-lex | −0.033 / −0.042 | 2.08 / 1.89 | 5.40 / 4.57 |
| NWL | 3k | lex | 0.158 / 0.175 | 1.87 / 1.84 | 4.53 / 4.48 |
| NWL | 10k | no-lex | 0.003 / −0.024 | 1.87 / 1.91 | 4.63 / 4.76 |
| NWL | 10k | lex | 0.201 / 0.184 | 1.87 / 1.82 | 4.48 / 4.45 |
| NWL | 270k | no-lex | 0.265 / 0.252 | 1.71 / 1.68 | 4.06 / 4.06 |
| NWL | 270k | lex | 0.293 / 0.285 | 1.66 / 1.72 | 3.89 / 4.15 |
| CSW zero-shot | 3k | no-lex | −0.024 / −0.029 | 2.25 / 2.07 | 5.68 / 5.03 |
| CSW zero-shot | 3k | lex | 0.159 / 0.173 | 2.02 / 2.01 | 4.95 / 4.86 |
| CSW zero-shot | 10k | no-lex | 0.006 / −0.014 | 2.07 / 2.05 | 5.05 / 5.00 |
| CSW zero-shot | 10k | lex | 0.193 / 0.181 | 2.02 / 1.94 | 4.78 / 4.66 |
| CSW zero-shot | 270k | no-lex | 0.239 / 0.229 | 1.84 / 1.83 | 4.43 / 4.49 |
| CSW zero-shot | 270k | lex | 0.281 / 0.288 | 1.69 / 1.79 | 4.00 / 4.22 |
| CSW, dictionary-sensitive | 270k | no-lex | 0.234 / 0.236 | 1.93 / 1.87 | 5.06 / 4.68 |
| CSW, dictionary-sensitive | 270k | lex | 0.287 / 0.310 | 1.75 / 1.81 | 4.38 / 4.44 |

Prediction was that the control collapses without cross-checks, most on CSW. It does not: within-position R² moves by ≤ 0.03 in
every cell of both arms, and the memory's lead on ranking is unchanged. Regret moves by up to 0.8 points, in both directions, which
is the seed spread at 3k (e.g. no-lex 3k seed 0 vs 1 on CSW: 6.28 vs 5.08 critical with cross-checks). So the round-2d explanation
(the control transfers through CSW cross-check sets) is wrong: the control learns what it needs from the labelled positions
(and possibly the shared NWL-pretrained query encoder), given enough of them.

## Round 3 (2026-11-17)

**Labeller bugs (fixed).** Checking `go/lexmem_plan` against full move enumeration (`CHECK=1`) found three problems in the binary
used for `plan_v1`, `plan_csw` and the exact planner of rounds 2c–2e:
1. The best-score generators had the game attached. When the sampled bag was empty, Macondo's shadow search then pruned with
   endgame bounds (not-playing-out penalty) while ranking plays by raw score, and returned a lower-scoring play in 4.2 % of
   evaluations (3419 of 81,920 in a 40-position test). Fix: no game on those generators (score-only bounds).
2. The bag's tile order before the seeded shuffle was random (the candidate's own refill is drawn with the CGP game's unseeded RNG),
   so labels were not reproducible and candidates did not share their tile samples. Fix: canonical order before every shuffle.
3. Ties at the top-K cut were broken by map iteration order. Fix: full sort key.

The fixed binary is deterministic (same output across runs and worker counts) and agrees with full enumeration on every
evaluation. Old vs fixed labels on the 300k positions (candidates matched; new − old):

| Phase | E_opp shift (mean / mean abs) | E_my shift (mean / mean abs) | one-turn best move changed |
|---|---|---|---|
| all | +0.13 / 1.24 | +0.28 / 1.11 | 23 % |
| opening | −0.02 / 1.37 | +0.35 / 1.33 | 30 % |
| middle | −0.08 / 1.34 | +0.09 / 1.15 | 23 % |
| late | −0.05 / 1.13 | 0.00 / 0.99 | 21 % |
| pre-end / end | **+0.81** / 1.09 | **+0.74** / 0.94 | 19 % |

The bias is confined to the end of the game; the rest is sampling noise in the old labels (unshared samples), which is enough to
flip the best move where candidates are within a point of each other. The round-2 one-turn checkpoints re-scored on the fixed
labels (same positions and held-out set, seeds 0–1) keep their ordering: all regrets drop by ~0.2 (less label noise), the gaps do
not change.

| Training positions | Arm | R² E_opp within | R² E_my within | Regret all | Regret critical |
|---|---|---|---|---|---|
| 3k | no-lex / lex | −0.037 / **0.171** | 0.274 / **0.428** | 1.83 / **1.63** | 5.22 / **4.35** |
| 10k | no-lex / lex | 0.003 / **0.224** | 0.357 / **0.444** | 1.65 / 1.60 | 4.42 / **4.17** |
| 270k | no-lex / lex | 0.293 / **0.325** | 0.503 / **0.536** | 1.45 / 1.39 | 3.80 / 3.63 |
| — | top-score baseline | | | 2.95 | 10.42 |

**Paired games, seeds 0 and 1** (4000 games per run vs HastyBot, seeds 777000, spreads stored by game index; lex − no-lex = mean
per-game difference of wins, ± its standard error, i.e. game noise only):

| Dictionary | Training positions | no-lex s0 / s1 | lex s0 / s1 | lex − no-lex s0 / s1 | pooled |
|---|---|---|---|---|---|
| NWL | 3k | 38.9 / 43.1 | 43.4 / 43.0 | +4.5 / −0.2 | **+2.2 ± 0.6** |
| NWL | 10k | 42.5 / 41.0 | 43.7 / 43.3 | +1.1 / +2.3 | **+1.7 ± 0.6** |
| NWL | 270k | 43.4 / 44.0 | 43.5 / 43.2 | +0.1 / −0.9 | −0.4 ± 0.6 |
| CSW | 3k | 40.0 / 42.1 | 41.4 / 42.5 | +1.5 / +0.4 | +0.9 ± 0.6 |
| CSW | 10k | 41.6 / 40.9 | 43.2 / 43.8 | +1.6 / +2.9 | **+2.3 ± 0.6** |
| CSW | 270k | 45.7 / 43.5 | 43.7 / 44.0 | −2.0 / +0.5 | −0.7 ± 0.6 |

Top score 36.6 % (NWL) / 35.6 % (CSW); exact planner with the fixed labeller **48.2 % / 48.1 %** (old labeller: 46.7 / 48.4). Seed-0 net
runs reproduce rounds 2c/2e to ±0.1. The round-2c headline (+4.4 at 3k) was seed 0; seed 1's control is as strong as the memory
arm, so the seed-to-seed spread is larger than the game noise. Per-game outcomes of two planners agree in 68–71 % of games.
First- vs second-seat win rates are now meaningful (first seat ~9 points higher for every player); the seat splits printed before
2026-11-17 came from completion order and carry no information.

**Clean control** (`QINIT=random NO_XCHK=1`, no-lex): the query encoder starts fresh and is trained, so the control has no
dictionary-trained part and no cross-checks; it also trains 6.37M parameters vs 3.50M (frozen-encoder control) and 3.57M (memory
arm, encoder frozen). NWL benchmark (round-2 labels, as the other rows) and zero-shot CSW, means of seeds 0–1, all arms without
cross-checks:

| Eval | Training positions | R² E_opp within: frozen no-lex / lex / clean | Regret all | Regret critical |
|---|---|---|---|---|
| NWL | 3k | −0.042 / **0.175** / 0.034 | 1.89 / 1.84 / **1.80** | 4.57 / 4.48 / **4.35** |
| NWL | 10k | −0.024 / **0.184** / 0.095 | 1.91 / 1.82 / **1.74** | 4.76 / 4.45 / **4.30** |
| NWL | 270k | 0.252 / **0.285** / 0.283 | 1.68 / 1.72 / **1.60** | 4.06 / 4.15 / **3.84** |
| CSW zero-shot | 3k | −0.029 / **0.173** / 0.038 | 2.07 / 2.01 / **1.94** | 5.03 / 4.86 / **4.70** |
| CSW zero-shot | 10k | −0.014 / **0.181** / 0.097 | 2.05 / 1.94 / **1.86** | 5.00 / 4.66 / **4.62** |
| CSW zero-shot | 270k | 0.229 / **0.288** / 0.260 | 1.83 / 1.79 / **1.66** | 4.49 / 4.22 / **3.99** |

Without any dictionary-trained part the control is better than with the frozen word-prediction encoder on every metric. The
memory still ranks candidates by the opponent's threat better at 3k–10k (+0.09–0.14 within-position R²), but the clean control
makes the better decisions at every size, in-domain and zero-shot on CSW. Its extra trainable capacity is a confound: the
comparison that isolates the memory is a memory arm with a trainable encoder too (not yet run).

**Two-turn horizon** (`go/lexmem_plan H2=1`, fixed labeller, same 300k positions, record order of `plan_v1`): after the candidate the
opponent plays its best-scoring reply, I answer with my best-scoring play (leave + refill), the opponent plays again (its leave +
refill); E_my2, E_opp2 = means over 256 samples; PV2 = score − E_opp + E_my2 − E_opp2. The two-turn answer differs from the
one-turn answer in 29 % of positions (one-turn answer loses 1.07 PV2 points on average, 1.51 at the end); 24.7 % of positions are
critical under PV2 vs 22.0 % under PV. `train_lexplan.py HORIZON=2`: three outputs, round-2 setting (frozen encoder, cross-checks
on), seeds 0–1:

| Training positions | Arm | R² within: E_opp / E_my2 / E_opp2 | Regret all (PV2) | Regret critical (PV2) |
|---|---|---|---|---|
| 3k | no-lex | 0.009 / 0.447 / −0.009 | 2.13 | 5.15 |
| 3k | lex | **0.222 / 0.516** / −0.013 | **1.83** | **4.16** |
| 10k | no-lex | 0.049 / 0.504 / 0.022 | 2.00 | 4.62 |
| 10k | lex | **0.257 / 0.548** / 0.051 | **1.70** | **3.82** |
| 270k | no-lex | 0.293 / 0.633 / 0.167 | 1.48 | 3.41 |
| 270k | lex | **0.349** / 0.624 / 0.163 | 1.46 | 3.27 |
| — | top-score baseline | | 3.32 | 10.73 |

Same shape as one turn: a clear memory advantage at 3k–10k (critical regret −1.0 and −0.8, both seeds), converging by 270k
(−0.14). Neither arm learns the opponent's second reply below 270k positions. Against the frozen-encoder control only; the clean
control was not run on two turns.

## Round 4 (2026-11-18): capacity-matched memory test

`QINIT=fut_train`: the query encoder starts from the future-word model (as in round 2) and is **trained** with the planner, so
the memory arm has the same trainable capacity as the clean control (6.44M vs 6.37M parameters). Three arms, all without
cross-checks, seeds 0–1:
- **clean**: no memory, encoder trained from scratch (`QINIT=random`, round 3);
- **same start**: no memory, pretrained encoder trained (`QINIT=fut_train ARM=nolex`);
- **memory**: lexicon memory, pretrained encoder trained (`QINIT=fut_train ARM=lex`).

One turn on the round-2 labels (as the clean-control rows of round 3; 743 critical positions), zero-shot on CSW, and on the
dictionary-sensitive CSW subset (`SENS_REF`):

| Eval | Training positions | R² E_opp within: clean / same start / memory | Regret all | Regret critical |
|---|---|---|---|---|
| NWL | 3k | 0.034 / 0.035 / **0.197** | 1.80 / 1.79 / **1.72** | 4.35 / 4.35 / **4.05** |
| NWL | 10k | 0.095 / −0.091 / **0.183** | **1.74** / 1.90 / 1.83 | **4.30** / 4.65 / 4.40 |
| NWL | 270k | **0.283** / 0.188 / 0.254 | **1.60** / 1.75 / 1.66 | **3.84** / 4.12 / 3.95 |
| CSW zero-shot | 3k | 0.038 / 0.044 / **0.199** | 1.94 / 1.93 / **1.82** | 4.70 / 4.65 / **4.27** |
| CSW zero-shot | 10k | 0.097 / −0.067 / **0.180** | **1.86** / 2.05 / 1.94 | **4.62** / 4.99 / 4.66 |
| CSW zero-shot | 270k | **0.260** / 0.170 / 0.253 | **1.66** / 1.81 / 1.74 | **3.99** / 4.31 / 4.13 |
| CSW dictionary-sensitive | 3k | 0.051 / 0.059 / **0.217** | 2.00 / 1.99 / **1.75** | 5.20 / 4.95 / **4.42** |
| CSW dictionary-sensitive | 10k | 0.103 / −0.048 / **0.200** | 2.02 / 2.22 / **1.99** | 5.24 / 5.61 / **5.02** |
| CSW dictionary-sensitive | 270k | 0.260 / 0.180 / **0.285** | **1.66** / 1.71 / 1.86 | **4.29** / 4.36 / 4.72 |

Critical regret per seed (NWL): clean 4.19/4.51, 4.35/4.25, 3.74/3.94; same start 4.20/4.49, 4.65/4.64, 3.66/4.57; memory
4.06/4.03, 4.46/4.34, 4.12/3.78.

- With the same start and capacity, the memory arm decides better at every size (critical regret −0.30 / −0.25 / −0.17 NWL).
- Without memory the pretrained encoder is a handicap at 10k–270k: the same-start control is worse than the clean control by
  0.35 / 0.28. Against the best no-memory net (clean), the memory arm wins only at 3k (−0.30 NWL, both seeds; −0.43 CSW; −0.78 on
  dictionary-sensitive positions) and is behind at 10k (+0.10) and 270k (+0.11 NWL, +0.43 on dictionary-sensitive CSW, both seeds).
- Training the encoder helps the memory arm: frozen (round 3, no cross-checks) 4.48 / 4.45 / 4.15 → trained 4.05 / 4.40 / 3.95.
- 10k caveat: every arm is still improving at its last evaluation (step 8000 of the round-2 budget). Both pretrained-encoder arms reach
  lower regret at 3k (4000 steps) than at 10k; the clean arm does not. The 10k comparison depends on the step budget.

**Two turns, trainable encoders** (fixed labels, PV2, 757 critical positions; no cross-checks). The same-start control was not run on
two turns:

| Training positions | Arm | R² within: E_opp / E_my2 / E_opp2 | Regret all (PV2) | Regret critical (PV2), seeds 0 / 1 |
|---|---|---|---|---|
| 3k | clean | 0.052 / 0.512 / 0.049 | 1.88 | 4.30 (4.21 / 4.39) |
| 3k | memory | **0.268 / 0.553** / 0.036 | **1.67** | **3.74** (3.61 / 3.87) |
| 10k | clean | 0.107 / **0.553 / 0.054** | **1.75** | 4.10 (3.88 / 4.31) |
| 10k | memory | **0.176** / 0.494 / −0.055 | 1.81 | **4.04** (3.95 / 4.12) |
| 270k | clean | 0.220 / 0.601 / 0.118 | 1.58 | 3.56 (3.49 / 3.63) |
| 270k | memory | **0.325 / 0.629 / 0.166** | **1.41** | **3.14** (3.17 / 3.11) |
| — | top-score baseline | | 3.32 | 10.73 |

The memory arm is better at 3k (−0.56) and at 270k (−0.42), both seeds, and tied at 10k. At 270k it is the best two-turn net so far
(frozen encoder with cross-checks, round 3: memory 3.27, control 3.41) and the first full-data decision gain for the memory. It is
not isolated yet. On one turn the pretrained start alone did not help a no-memory net. On two turns, though, the frozen pretrained
control with cross-checks (3.41) beats the clean control (3.56), so the pretrained start may help without memory here. The
same-start two-turn control (`HORIZON=2 QINIT=fut_train ARM=nolex`) settles it.

## Round 5 (2026-11-18): search-based self-play

**Baseline against a BestBot proxy.** Macondo's `SIMMING_INFER_BOT` (simulation + rack inference + pre-endgame and endgame
solvers), at most 60 s per move, 2 sim threads, 1000 games per row, seeds 777000. Rack inference reads the opponent's last
move from the game history; the bot wrapper now rebuilds the position as (position before our last tile play) + that play,
so inference runs (it was silently skipped before: a bare CGP has no history).

| Our side | Win rate vs the proxy |
|---|---|
| Net alone (`simsoft_scratch_nolex`) | 42.5 % ± 1.6 |
| Net + SIMMING_BOT 10 s for the last 7 bag tiles | 47.4 % ± 1.6 |
| SIMMING_BOT 10 s (our teacher) | 51.8 % ± 1.6 |

The proxy is not BestBot strength: our teacher beats it, and Macondo's stopping rule ended most of its searches well before
60 s (runs took 1.2–1.9 h). Read as "against a teacher-level bot". The net-alone run's per-game spreads were overwritten by
the hybrid run (same output name), so only its summary is kept.

**Stage-1 gate: does search over the net improve on the net?** Whole games in Go (`go/lexmem_search`, `lumi/search_eval.sh`)
against HastyBot, 2000 games per row on seeds 777000 + g (we move first in even g), net `policy_simsoft_exch_nolex` served
on one GCD. "Handoff" = SIMMING_BOT 10 s once bag + opponent rack ≤ 14 tiles (the last 7 bag tiles). Search: the policy's
top K = 8 moves, each played out in the same S = 32 sampled worlds (bag shuffle + opponent rack), D = 2 further half-moves
by net argmax over the 64 highest-scoring moves; the leaf is Macondo's win-probability table at the leaf spread, with the
last mover's leave value (`mac`) or our value head's 4-ply spread prediction (`net`) added. Z = 1: a candidate replaces
the policy's top move only if its mean paired gain over the 32 worlds exceeds one standard error. Paired = mean per-game
win difference against the hybrid over the same seeds. Pass criterion: ≥ +3 points over the hybrid.

| Our side | Win rate | Mean spread | vs hybrid, paired |
|---|---|---|---|
| Net alone | 51.8 % ± 1.1 | +4.0 | −4.7 ± 0.7 |
| Net + handoff (hybrid) | 56.5 % ± 1.1 | +10.0 | — |
| Search, `mac` leaf + handoff | 57.1 % ± 1.1 | +10.0 | +0.6 ± 1.4 |
| Search, `mac` leaf, Z = 1 + handoff | 58.3 % ± 1.1 | +11.3 | +1.8 ± 1.2 |
| Search, `net` leaf + handoff | 57.6 % ± 1.1 | +5.8 | +1.1 ± 1.4 |
| SIMMING_BOT 10 s, whole game (teacher; 1000 games) | 60.7 % ± 1.5 | +17.7 | +6.6 ± 2.0 |

The gate fails: no search variant reaches +3. The net-alone row matches the Python harness (52.4 %), so the Go harness
reproduces our earlier evaluations; the teacher is 1.9 points below its Python-harness 62.6 %. The `net`-leaf search
trades spread for wins as a win-probability objective should (losses by < 30 points 9.4 % → 6.6 % of games, losses by
> 100 points 12.8 % → 15.0 %), but its net gain is within noise. Cost: 1.5–1.75 GCD-hours per 2000 search games at
1700–2300 positions/s.

**Position-level check: does the search approach the teacher with more worlds?** (`eval/search_positions.py`,
`lexmem_search POSITIONS=...`) 1000 held-out positions of `simsoft_31` (one per game; teacher = SIMMING_BOT 10 s with its
top-16 sim candidates and win %; positions where the teacher played tiles, bag ≥ 8, no openings), same net and search as the
gate (K = 8, D = 2, PM = 64). One run per leaf with S = 512 (`mac`) or 256 (`net`) worlds; smaller S = the first S worlds.
Agreement = the choice is the teacher's move; loss = the teacher's best win % minus the teacher's win % for the choice
(win-% points, per move; choices outside its 16 candidates counted at its lowest: a lower bound). Policy top move:
agreement 68.6 % ± 1.5, loss 0.32; the teacher's move is among the policy's top 8 in 95.3 %.

| Search (Z = 1) | `mac` agreement | `mac` loss | `net` agreement | `net` loss |
|---|---|---|---|---|
| S = 32 | 70.1 % (+1.5 ± 1.1) | 0.25 | 67.9 % (−0.6 ± 1.2) | 0.34 |
| S = 64 | 73.3 % (+4.7 ± 1.1) | 0.21 | 71.2 % (+2.7 ± 1.2) | 0.25 |
| S = 128 | 75.4 % (+6.8 ± 1.2) | 0.19 | 71.5 % (+3.0 ± 1.2) | 0.25 |
| S = 256 | 77.7 % (+9.1 ± 1.2) | 0.14 | 72.5 % (+4.0 ± 1.3) | 0.23 |
| S = 512 | 81.1 % (+12.5 ± 1.2) | 0.13 | — | — |

(± = paired against the policy's top move over the same positions.) With the `mac` leaf the search moves steadily towards
the teacher: at S = 512 it switches away from the policy's top move in 22 % of positions, 147 times to the teacher's move
and 22 times away from it, and removes 60 % of the policy's teacher-estimated loss. Z = 0 is worse at every S (at S = 32 it
loses agreement with both leaves: −6.7 `mac`, −10.8 `net`): with few worlds the plain argmax follows noise. The `net` leaf
gains less at every S. Caveat: the `mac` leaf uses the same leave values and win-% table as the teacher's own sim, so part
of the agreement is a shared evaluator; agreement cannot show play beyond the teacher. Cost: 2.9 GCD-seconds per position
at S = 512 (`mac`, 2820 positions/s).

**Stage-1 gate at S = 256** (the position check's recommendation): search `mac` leaf, Z = 1, S = 256, otherwise as in the
gate table (K = 8, D = 2, PM = 64, handoff), vs HastyBot, the same 2000 seeds (four runs of 500 games, `G0` = 0 … 1500).

| Our side | Win rate | Mean spread | vs hybrid, paired |
|---|---|---|---|
| Net + handoff (hybrid) | 56.5 % ± 1.1 | +10.0 | — |
| Search S = 32, Z = 1 + handoff | 58.3 % ± 1.1 | +11.3 | +1.8 ± 1.2 |
| **Search S = 256, Z = 1 + handoff** | **60.1 % ± 1.1** | **+16.5** | **+3.6 ± 1.3** |
| SIMMING_BOT 10 s (teacher; games 0–999) | 60.7 % ± 1.5 | +17.7 | +6.6 ± 2.0 |

The gate passes (+3.6 against the +3 criterion; the position check predicted about +3.7). On games 0–999 the search is
−1.7 ± 1.7 against the teacher, within noise; S = 256 is +1.8 ± 1.3 over S = 32. The gain is larger when we move first
(+5.2 ± 1.8) than second (+2.0 ± 1.9). Cost: about 2 GCD-hours per 500 games (1.4 s of search per move).
