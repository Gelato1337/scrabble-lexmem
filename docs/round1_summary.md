# Round 1 summary (2026-09-26 – 2026-10-22)

**Question.** Can a model with explicit memory over the whole lexicon learn Scrabble beyond what a model without it can —
and, eventually, beyond human level?

**Answer so far.** The memory retrieves words very well, and a policy trained against a strong teacher beats HastyBot, but in
every controlled comparison of *playing quality* or *position evaluation* the memory has not yet made a measurable difference.
All numbers: [benchmarks.md](benchmarks.md).

## What held

| Finding | Evidence |
|---|---|
| The memory retrieves the right words | NDCG@32 0.97; every playable NWL23 word in the top 32 of its rules mask; held-out words behave like trained ones |
| The memory predicts the words that will actually be played | opponent's next words in the top 8 of a lane: 0.51 vs 0.30 for score ranking (sim-based targets); 0.42–0.44 on real next moves |
| A net can learn a strong teacher's judgement | soft targets over SIMMING_BOT's candidates: 53.2 % ± 0.8 vs HastyBot, no search (single-pick imitation: 50.3 %) |
| Games are decided at the end of the bag | 8 % of positions (1–7 tiles in the bag) carry 41 % of the net's loss against the teacher; Macondo's solver there: 53.2 % → 58.1 % |
| Fast, exact self-play is possible | Go + GPU engine: ~76k games/h per GPU, identical scores to our Python engine (and found a Python bug in blank-play scoring) |
| Low-noise value targets from our own play | 16 rollouts per position: value R² 0.27 and win log-loss 0.43 (one game's result: no better than predicting zero) |

## What did not (controlled, held-out, several seeds where noted)

| Test | Lexicon memory | No memory | Verdict |
|---|---|---|---|
| Move choice, imitation of HastyBot (win rate) | 48.6 % | 51.2 % | no gain (memory slightly worse) |
| Move choice, soft teacher targets (win rate) | 51.0 % | 52.8 % | no gain |
| Which of two moves wins its playout, 251k held-out branch pairs (zero-shot) | 0.607 | 0.605 | within noise; late game 0.584 vs 0.575 |
| Same, after training on 2.3M pairs | 0.603 | ~0.602 | neither learns from the pairs |
| Value, game-outcome targets (1 seed) | LL 0.510 | LL 0.528 | gain only in the opening, none late |
| Value, rollout targets, 3 seeds | R² 0.230, LL 0.4372 | R² 0.236, LL 0.4358 | no gain |
| Value, frozen retrieval, 10k / 50k / 177k positions, 2 seeds | R² 0.117 / 0.247 / 0.257 | R² 0.129 / 0.232 / 0.266 | no sample-efficiency gain |

![Lexicon memory sample efficiency](figures/lexmem_sample_efficiency.png)

Self-play with game results as the only signal did not improve the policy either: Q-learning collapsed (53 % → 3 %); an
actor-critic with gating stayed level (52.5 %); training on 2.3M shared-draw branch pairs taught the net nothing new.

## How to read the negative results
- They are **strong for the designs tested**: per-lane, per-side retrieval tokens feeding a value network or appended to a policy,
  with equal parameters, data and training; three seeds where it mattered.
- They do **not** test the memory on decisions whose right answer depends on specific words several draws away (keeping tiles
  for a bingo, fishing, blocking a lane). Value and next-move imitation are dominated by luck and by the current score; a better
  word map shifts them only slightly. That is the main open gap.
- The memory's one consistent strength is *predicting words*. Nothing so far turns that into *better decisions*.

## Open directions for round 2
1. **Planning targets.** Build positions where the right move is defined by specific future words (constructed "keep for word X"
   positions with exact answers; expert positions where only deep search finds the move) and test lex vs no-lex on those.
2. **The memory as a planner, not an input.** Propose target words (for both sides), then choose the move that keeps mine
   reachable and theirs blocked; compare against the same planner without the lexicon.
3. **Opponent modelling.** The memory's opponent-word prediction (0.51 top 8) is its strongest signal; inference of the
   opponent's rack and threats is where human experts use word knowledge most.
4. **Keep what works as infrastructure**: the Go self-play engine, rollout targets and paired-seed evaluation make each of these
   tests cheap.
