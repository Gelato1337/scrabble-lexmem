# Scrabble: the problem, stated fully

## 1. Rules (English, 2-player)

- Board 15×15. Premium squares: DL, TL, DW, TW. Center square starts.
- Bag: 100 tiles, fixed distribution, public. English standard (count / point value):

  | | | | | | | | | | | | | | |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | A 9/1 | B 2/3 | C 2/3 | D 4/2 | E 12/1 | F 2/4 | G 3/2 | H 2/4 | I 9/1 | J 1/8 | K 1/5 | L 4/1 | M 2/3 | N 6/1 |
  | O 8/1 | P 2/3 | Q 1/10 | R 6/1 | S 4/1 | T 6/1 | U 4/1 | V 2/4 | W 2/4 | X 1/8 | Y 2/4 | Z 1/10 | blank 2/0 | |

  Vowels (AEIOU) 42, consonants 56, blanks 2. Singletons: J K Q X Z. Total 100. Other languages: different counts, values, alphabet; same structure.
- Each player holds 7 tiles. Draw back to 7 after every turn while the bag has tiles.
- Turn = one of:
  - **Play**: place ≥1 tiles in one row/column, forming a contiguous line connected to existing tiles. Every new contiguous string (main word + all cross-words) must be in the lexicon. Score = sum of letter values with premiums (premiums apply only to newly placed tiles); 7 tiles played = +50 (bingo).
  - **Exchange**: return 1–7 tiles to the bag, draw the same number. Only if bag ≥ 7. Score 0.
  - **Pass**: score 0.
- Blank = any letter, value 0, fixed once played.
- Game ends when one player empties their rack with the bag empty, or after 6 consecutive scoreless turns. Unplayed tiles: subtracted from holder, added to the player who went out.
- Lexicon: ~280k words (CSW) or ~190k (NWL). Fixed and public.
- (Tournament: challenges, clocks. Ignore for now.)

## 2. Game state

**Public (both players see):**
- Board contents (letters, blank positions, which premiums are still open)
- Both scores, turn number, tiles remaining in bag (count)
- Full move history
- Lexicon and tile distribution
- → therefore **unseen multiset U** = distribution − board − own rack. Exact, known.

**Private (to me):**
- My rack R (7 tiles)

**Hidden (to me):**
- Opponent rack O ⊂ U (7 tiles)
- Bag order / contents = U − O

That's it. The state is small. The *reachable future* is not.

## 3. What "solving" requires

Optimal play means choosing the move maximizing expected outcome over:
1. all opponent racks consistent with U,
2. all future draws,
3. all future lexicon-legal plays by both sides,
4. all possible future boards those plays create.

Branching: hundreds of legal plays/turn × C(|U|,7) opponent racks × draw sequences. ~25 turns. Exhaustive search is impossible from the opening; it becomes possible only near the end (bag empty → perfect information → solvable; 1–2 tiles in bag → nearly solvable).

**The information-flow requirement (the bottleneck):**
The decision-maker must have access to *what is still reachable*: which high-value plays could still exist, given
- the lexicon (280k words / letter combinations),
- the tiles still in play (U),
- what's on the board (anchors and lanes),
- who probably holds what (belief over O).

Every strategic act — fishing, blocking, setting up, dumping tiles to starve the opponent of a letter, timing the endgame — is a function of that reachable set. If the model can't see the reachable set, those tactics can only be imitated, never derived.

## 4. What prior work did, and why it stops short

| System | Move gen | Evaluation | Lookahead | Hidden info | Lexicon inside the learner? |
|---|---|---|---|---|---|
| **Maven** (1990s) | DAWG | score + hand-tuned leave table | Monte Carlo sim, 2–4 plies; exact endgame | random opponent racks | No |
| **Quackle** (2005) | GADDAG | score + leave table from self-play | MC sim, 2–5 plies; endgame | random | No |
| **Macondo / BestBot** (2020–26) | GADDAG | score + leave table → replaced by a 10-block CNN static evaluator (post-move snapshot → 5-ply spread) | MC sim; exact endgame + pre-endgame solver | Bayesian rack inference (hand-built, separate from the net) | No — CNN sees cross-check planes (local legality), never the lexicon |
| **Word Freak** (2025–) | Rust GADDAG | 51 hand-made features + NN, self-play | none published | none published | No |

**Shared limitation:** the lexicon is only ever consulted by the *engine* for the current board, or for boards reached inside a shallow simulation. The learner (leave table or CNN) sees only a snapshot and a fixed horizon. Nothing in the pipeline represents "what remains reachable" as a first-class quantity. Lookahead is bounded by simulation depth (2–5 plies), and each ply re-queries the engine on concrete sampled boards — the set of futures is *sampled*, never *represented*.

Consequences:
- Plans beyond the sim horizon are invisible.
- Rare/long words are only "seen" if a rollout happens to reach a board where they're legal.
- Tile-management strategy (which tiles to remove from play to deny the opponent) has no representation at all beyond leave values.
- Belief over O is computed outside the learner and only used to seed rollouts.

## 5. The question

How do we give the learner access to the **reachable-future set** — combinations of tiles that can still be unlocked, given lexicon + U + board + belief — in a form that
1. is complete (every possibility discoverable, no fixed horizon),
2. shrinks correctly as the rules remove possibilities (tiles leave the bag, squares fill, lanes close),
3. is cheap enough to compute every turn,
4. lets the model reason over it end-to-end (so tactics are derived, not imitated)?

Everything else — architecture, layers, training — comes after this.

## 6. The information-flow problem, stated plainly

### The claim

**Strategy in Scrabble is a function of the lexicon. Any learner that cannot query the lexicon at decision time cannot derive strategy; it can only imitate it.**

### Why

Every interesting move is "do something worse now so that a *specific word* becomes playable later, or so that a specific word becomes *unplayable* for the opponent." Examples:

- Fishing: keep ERS?, play off 3 junk tiles, because the lexicon contains many 7s/8s from ERS? + common draws.
- Setup: play a word that leaves an open T next to a triple, because you hold letters that form an 8-letter word through a T.
- Blocking: close a lane, because the opponent's likely rack + the letters still unseen + that lane = a bingo *for them*, and you can name which words.
- Starving: dump tiles now (even at a loss) to change what is left in the bag, so a letter the opponent needs is no longer available.
- Timing: keep the bag at 1 tile so you draw last and they cannot draw into the blank.

In every case the value of the move depends on **which words are still alive** — for me, for them — given the tiles that remain and the shape of the board. "Alive" is a query over all 280k words, filtered by U, the racks (known / believed), and the lanes.

A learner that only sees the current board, the current legal moves, and a scalar leave value has no way to compute "alive." It sees the *consequence* of alive words when they get played (in the outcome), but never the *cause*. So it can fit correlations ("keeping ERS? tends to score later") but cannot reason ("RETAINS goes through that E and they lack the S"). Correlation-fitting is what leave tables and the Macondo CNN do, and it plateaus at ~52% vs greedy because that is how much of strategy is visible without the lexicon.

### Why it must be available *before* the move, not discovered *during* search

Monte Carlo sim and DAG-based search consult the lexicon only on concrete sampled boards. Two failures:

1. **Horizon.** A setup that pays off in 4 turns is invisible to a 2-ply sim, and a 4-ply sim over hidden information is intractable. The tactic is never in the training signal, so it is never learned.
2. **Surprise.** Each sampled future is a different board. The learner never sees "the set of futures"; it sees one at a time, each a surprise. Nothing consistent accumulates. The model cannot learn *what kind of board is good for my tiles*, because "good for my tiles" is a property of the reachable-word set, and the reachable-word set is never represented.

So the reachable-word set must be an *input* to the decision, present every turn, not a by-product of rollouts.

### What the model must be able to do, concretely

Given (my rack, unseen multiset U, belief over opponent rack, board lanes):

1. **Retrieve** specific words from the full lexicon that are alive for me — including rare ones — keyed by partial constraints (some letters held, some on the board in a pattern, some still drawable).
2. **Retrieve** the same for the opponent, under the belief.
3. **Track** how both sets shrink as tiles leave the bag and squares fill — and therefore predict how *my* move changes both sets (opening/closing lanes, removing tiles from U).
4. **Commit** to a target: hold "I am building toward word W in lane L" across several turns, and re-evaluate it as U and the board change.

If any of 1–4 is missing, the corresponding class of tactics is unreachable. This is the bottleneck. All four require the lexicon to be *inside the loop* — queryable by the model, every turn, on partial information.

### Why prior work cannot get there by scaling

More self-play data does not help a leave table or a snapshot CNN, because the missing quantity (which words are alive) is not in their input. The bitter lesson says scale beats hand-crafted knowledge — but only when the information is *available* to the learner. Here it is not. Scaling a model that cannot see the lexicon scales its ability to fit the residue of strategy, not strategy itself.

### The design constraint that follows

The 280k-word lexicon must be a first-class, queryable part of the model's state — present at every layer where decisions form, addressable by partial letter constraints, and used by training objectives that force *specific future words* into the latent (e.g. predict which words I / the opponent will play in the next k turns). The DAG remains the legality oracle at the boundary. The strategy lives in the query.

## 7. Why this is harder than chess or Go

| | Chess / Go | Scrabble |
|---|---|---|
| Information | perfect | hidden opponent rack + random draws |
| Rules size | a page | a page **plus 280k words**; the rules are the size of a corpus |
| Legal moves / turn | ~35 / ~250 | hundreds to thousands, most differing only in leave |
| Value locality | a move's value is mostly about the position | a move's value depends on words that don't exist on the board yet |
| Endgame | still hard | exactly solvable once the bag is empty (perfect info) |
| Best human vs best engine | engines far ahead since ~2000 | engine edge is small; the top human beats the classic engine on move quality |
| Variance | low | high: going first ≈ 56% win; ~75% is a dominant win rate |

The combination that has no precedent: **imperfect information × stochastic × a rule set too large to hold in a search tree**. Poker has the first two but tiny rules. Go has huge search but complete information. Nothing has all three.

## 8. Evidence that the bottleneck is representation, not capacity

From Macondo's own experiment log (pytorch/experiments.md):

- CNN static evaluator: 64ch/6 blocks → 96ch/10 blocks → 128ch/10 blocks. Win rate vs greedy: 52.4% → 52.6% → 52.5%. **Bigger model, no gain.**
- Prediction horizon 3 / 4 / 5 / 6 / 7 plies → 50.8 / 51.2 / 52.3 / 52.0 / 52.1%. **Longer horizon made it worse**, i.e. the snapshot cannot use future information.
- Adding one extra turn of history planes: no gain or slight loss. **More history, no mechanism to use it.**
- The one thing that helped: training data with more "sacrifice equity" positions (softmax opponent). **The signal exists in the data; the model can't absorb it.**

A capacity-limited model improves with size. This one didn't. The missing input is the lexicon-conditioned reachable set (§6).

## 8b. The prevailing counter-thesis, and why we reject it

**Sheppard (Maven, 2002), the field's reference point, argues the opposite of §6:**
- Board factors are largely irrelevant (except triple-word squares).
- 2-ply simulation is enough. Deeper lookahead only helps in "exotic situations like CACIQUE," and tile turnover makes deep futures too noisy to be worth the cost.
- Maven is superhuman.

This became the default belief. DeepMind's David Silver publicly lists Scrabble among solved games. Result: almost no ML research on Scrabble in 24 years.

**Evidence against it (informal, but consistent):**
- Nigel Richards makes fewer errors than Quackle on move-quality analysis; Quackle vs grandmaster Matsumoto over 500 games: 252–248. Not superhuman.
- Matsumoto maintains a list of positions where simulation gives the wrong answer.
- A human applying a "lowest-variance-after-sim" rule reportedly beat Quackle's championship player 62–70% over 250 games — i.e. a *board-shape* consideration sim does not represent.
- Community estimate: Scrabble is ~75% play-finding and ~25% "everything else," and that 25% makes champions. Bots get the 75% for free from the GADDAG. The 25% is exactly what has no representation.
- Maven's own stated statistics for "best humans" (33 pts/move, 1.5 bingos) are now far below top human play (37.5 pts/move, 2.16 bingos).

**A specific known flaw in simulation:** it assumes the opponent always plays the top-equity move. So it overestimates comebacks (expects the opponent to randomly open a triple-triple) and cannot model an opponent who is *aiming* at something. This is the opponent-target half of the belief problem, separate from rack inference.

**How Sheppard's claim and ours are reconciled:** he is right that *sampling* concrete 4-ply futures is too noisy to pay off. He concludes deep futures don't matter. We conclude they must be *represented*, not sampled. The noise is in the method, not the game.

## 9. Constraints we hold ourselves to

1. **No hand-coded tactics.** No hook planes, lane features, threat maps, leave tables. If the model needs them it must derive them.
2. **Rules-level inputs only**: lexicon, tile distribution, board geometry, legality. These are the environment, not heuristics. (AlphaZero got the rules of chess; the lexicon *is* our rules.)
3. **Per-language by swapping data**, not architecture: lexicon + tile table in, engine out.
4. **Every possibility discoverable**: no fixed horizon baked into the representation or the target.
5. **Learned, not sampled**: the model must be able to reason about the distribution of futures, not be surprised by one rollout at a time.
6. **Endgame solved exactly**, not learned. Don't spend capacity on what search does perfectly.

## 10. Candidate directions touched so far (none committed)

- **Tiles as persistent entities.** State = 100 tile tokens, each with location (board / my rack / unseen) and belief P(opponent holds it). Monotone pipeline bag→rack→board; "removing a tile from play" becomes an explicit operation. Squares are where tiles land, not the state.
- **Split the reachable set into two factors.** Tile-feasibility (lexicon ∩ tiles-in-play): global, board-independent, shrinks monotonically. Placement-feasibility (lanes): local, changes with each move but spatially bounded. The "drastic change" problem lives only in the second, small factor.
- **Lexicon as attendable memory.** 280k word vectors (each built from its letters) that the model cross-attends into, queried by partial constraints (held letters, unseen letters, lane pattern). Letter-level input, word-level memory/output. Retrieval, not generation: a word is *looked up*, not formed.
- **Future-word prediction as the training signal.** From self-play: which words will I / the opponent actually play in the next k turns. Retrospectively labeled. Forces specific words into the latent early, and forces a belief over the opponent's targets.
- **Learned latent dynamics (MuZero-style, stochastic).** The board is an observation; the state is a learned latent trained to be smooth and predictive. The consistent thing the model reasons over is learned, not designed.
- **Determinized search on top** (sample opponent racks from the belief, search each world) — only as a compute dial, not as the source of strategy.
- **Data**: unlimited self-play from Macondo/MAGPIE; human games from Woogles; Nigel's games are too few to train on — use as validation only.

## 11. Open questions

1. Is the *reachable-word set itself* the right state? Or is it a view over the tile-entity state?
2. Enumerate or evaluate: does the model need to name the specific word it is building toward, or only know that a rich future exists? (Nigel names it. Does the optimal player need to?)
3. Can soft attention over 280k keys retrieve one rare word sharply, or does it smear into "long words with these letters"? Testable in pretraining, before any game data.
4. Can a learned latent hold lexicon-level detail ("11-letter through-play via T and D"), or does it smooth it away? This is the crux of whether learned dynamics beat sim or merely match it.
5. Word-level tokens: sparsity kills a lookup table, but does uniform lexicon pretraining fix that? Where exactly should the boundary between letter-level and word-level sit?
6. How does the belief over the opponent's *targets* (not just their rack) get represented and trained?
7. What is the right evaluation? Ratings across pools aren't comparable; variance is high; "beats Nigel" needs hundreds of games and no access to Nigel.

## 12. Benchmarks

### 12a. The engine ladder (researched 2026-09-27; sources at the end of this section)

| engine | method | speed | measured strength |
|---|---|---|---|
| **HastyBot** (Macondo, Woogles) | score + leave value (equity), no lookahead | microseconds/move | Del Solar: "maybe a few dozen players in the entire world" could beat it in a 100-game series. A Woogles club guide puts the best humans near 50 % against it. **Our fixed anchor.** |
| STEEBot (Woogles) | older sim-based bot, "knows all the words, makes some mistakes" | seconds | below HastyBot on Woogles' own tiering (BetterBot < STEEBot < HastyBot < BestBot) |
| Macondo CNN static evaluator (2025) | 10-block CNN on post-move snapshot, trained on 7.5 M HastyBot games / ~185 M positions | microseconds | **51.7 % vs HastyBot** over 255,626 games (best of a year of experiments, see §8) |
| Quackle championship player (2005–) | 2–5-ply Monte Carlo sim + superleaves, exact endgame | seconds–minutes | 50–55 % vs top humans; 248–252 vs grandmaster Matsumoto over 500 games |
| **BestBot** (Macondo, 2025) | ≥ 5-ply Monte Carlo sim with Welch t-test pruning, Bayesian rack inference, pre-endgame solver (1-in-bag: up to 8 endgames per candidate), exhaustive alpha-beta endgame | **minutes/move** | **59–60 % vs HastyBot**; 56–57 % vs Quackle championship player; ≈ 50 % vs the strongest humans; Mack Meller (2024 North American champion) 40–42 % vs BestBot in a long YouTube series. **The state of the art.** |
| Word Freak (2025–), MAGPIE | Rust GADDAG + NN; C engine used for sim/pre-endgame | – | no published head-to-head numbers found |

So: HastyBot is not the state of the art; it is the *floor* every serious engine is measured against, and it is the only one that is cheap enough to play 100k games in 36 s. BestBot is the top, and it gets there by search and solvers on top of the same equity function, not by a stronger learned evaluation (the CNN alone adds < 2 points).

### 12b. Humans

- Nigel Richards: lifetime win rate ≈ 75 % in championship play (other elite players are far below that; the WESPA average is 50 % by construction). Ratings: NASPA 2221, WESPA 2174 (current, 2026); career high 2298 (WESPA/cross-tables), the highest ever over ≥ 200 games. (An earlier draft of this doc said 2,307; the sourced figure is 2,298.)
- Matsumoto (2022): "computers in Scrabble are far from reaching the levels of excellence that they have reached in other games." Del Solar (2025): "My hope is that we can get something that can beat Nigel Richards, convincingly."


**Woogles ratings (live from the Woogles profile API, 2026-09-27; Glicko, comparable to Elo within Woogles):**

| bot | CSW regular | CSW rapid | NWL regular | NWL rapid | record (CSW rapid) |
|---|---|---|---|---|---|
| BestBot | **2410** | 2321 | 2216 | 2276 | 2112–692 (75 %) |
| HastyBot | 2375 | 2245 | 2134 | 2220 | 199,053–51,090 (80 %, mostly vs weaker humans) |
| STEEBot | 2224 | 2234 | 1959 | 2058 | 66,162–20,537 |

The bots' win records are inflated by the human pool they play; the rating gap BestBot–HastyBot (≈ 35–80 points) is the meaningful number and matches the 60 % head-to-head. (`curl -X POST https://woogles.io/api/user_service.ProfileService/GetProfile -d '{"username":"BestBot"}'`.)

### 12c. Is there an Elo for the bots?

No comparable one. NASPA and WESPA ratings are Elo-like but only cover in-person tournaments; bots are not rated there. Woogles ratings are a separate pool (Glicko-style) and not convertible. The usable currency is **head-to-head win rate**, which converts to an Elo *difference* by Δ = −400·log10(1/p − 1):

| win rate | Elo difference |
|---|---|
| 52 % | +14 |
| 55 % | +35 |
| 60 % | +70 |
| 65 % | +108 |
| 75 % | +190 |

Reading the ladder in those units: BestBot ≈ +70 over HastyBot; the CNN alone ≈ +12; the strongest humans sit somewhere between HastyBot and BestBot, i.e. within ~70 Elo of each other and of both bots. Scrabble's draw luck compresses everything: a +190 gap (Nigel over the field) is the largest that exists in the game. **Practical consequence for this project: a raw policy at 55 % vs HastyBot (+35) would already be well above the published CNN; 60 % (+70) would equal BestBot without search; anything above that with search on top would be new territory.** Because 2 points of win rate is ~14 Elo and the per-game spread standard deviation is ~100 points, every comparison needs thousands of games.

Sources: [BestBot announcement (Woogles, 2025-03)](https://blog.woogles.io/posts/2025-03-10-bestbot-its-here/) · [The mathematics and algorithms behind BestBot (Woogles, 2025-05)](https://blog.woogles.io/posts/2025-05-04-the-mathematics-and-algorithms-behind-bestbot/) · [Del Solar, Beyond leave tables: a CNN static evaluator (2025-06)](https://www.cesardelsolar.com/posts/2025-06-21-nn-scrabble/) · [Android Police interview piece on engines vs humans](https://www.androidpolice.com/i-learned-ai-cant-always-beat-the-best-scrabble-players-and-it-restored-my-faith-in-humanity/) · [Colombo Scrabble Club guide to Woogles bots](https://www.colomboscrabble.com/blog/woogles) · [WESPA rating history, Nigel Richards](https://www.wespa.org/aardvark/cgi-bin/rathist.cgi?name=Nigel+Richards) · Macondo `pytorch/experiments.md` (this repo).

### 12d. Our numbers so far (2026-09-27)
- score-greedy Python bot (no leave values) vs HastyBot, 100 games: 35 % wins, −34 spread (≈ −110 Elo).
- **imitation net, no lexicon branch** (3.9M params, 30k steps × 48 positions of HastyBot play, argmax policy, no search) vs HastyBot,
  300 games: 54.0 % (noise). **2000 games: 50.3 % ± 1.1 wins, +1.8 ± 2.2 spread** (`play_eval.py`, 60 CPU procs, 7 min).
  Top-1 agreement with HastyBot 81 %, top-3 96 %. Reading: the imitator equals its teacher exactly, as expected; the
  leave function is learned from data, nothing more. Equal to HastyBot = roughly the level of the top few dozen humans.
- imitation net **with** lexicon branch: 300 games 49.0 %; retrained (imit2, with pos-only value loss) **2000 games: 46.9 % ± 1.1, −8.4 spread**,
  i.e. ~3.5 points *below* the no-lexicon net (50.4 %) despite better imitation accuracy. As currently wired (two rack-level
  queries, frozen keys, top-32 tokens) the branch hurts play at the imitation stage. Hypotheses: the top-k selection is
  discontinuous and adds variance; the tokens dominate attention early. To test: fewer tokens / logit-weighted pooling / unfreeze keys. Top-1 agreement with HastyBot **85.2 % vs 81.0 %**
  without the branch: the reachable-7s/8s tokens help predict HastyBot's leave choices, as they should.
- **1-ply value search hurts**: no-lexicon net + expectimax over the position-only value head (K=6 candidates, S=6 sampled opponent
  racks, independent samples per candidate): **37.6 % ± 1.1, −28.6 spread** vs 50.4 % for the bare policy. The value head's error is
  46 points; ranking candidates by it ranks by noise. With common random numbers and S=16: 42.9 % ± 1.1 (−16.5 spread);
  adding the policy prior (mix 10 / 30, or K=3): 48.4 / 48.6 / 48.9 % at n=1400 — never above the bare policy. Value head retrained to predict the
  4-ply spread change (imit3): MAE 29.0 pts vs 33.1 for predicting zero — weak. Bare policies: no-lex **51.2 % ± 1.1**, lex 48.6 %.
  Corrected 1-ply lookahead on it (candidate score − 100·E[opponent's 4-ply gain], common random racks): K=6/S=16 **42.5 %**,
  K=12/S=32 **39.7 %** (wider = worse: ranking by noise), with policy prior mixed in 50.4 % = bare policy.
  **Conclusion 2026-09-28: a static value head at this precision cannot serve as the self-play teacher.** Same wall Macondo hit.
  Paths: (a) Monte-Carlo rollouts with the policy net as the rollout player — needs fast move generation (Go/ONNX integration);
  (b) policy-gradient from outcomes with the value as baseline, no lookahead. **Round pg1 (plain REINFORCE, 15k self-play games,
      6k steps) collapsed: no-lex 1.3 %, lex 42.1 % vs HastyBot; HastyBot-agreement 81 % → 19 %.** Unbounded objective + off-policy
      epochs. Replaced by PPO-clip (behaviour log-probs stored at generation, ratio clipped ±0.2, normalised advantages, <1 epoch).
      **Round pg2 (PPO, from the imit3 checkpoints): no-lex 51.4 % ± 1.1 (was 51.2), lex 47.3 % (was 48.6).** Stable, no measurable
      gain from one round of 15k games. **pg3: no-lex 50.2 %, lex 43.8 %** (lex agreement with HastyBot 85 → 71 → 66 %: the prior erodes, nothing replaces it). pg4–5 running.
      Verdict forming: outcome-only gradients at 15k games/round do not improve play. Built 2026-09-28 (`lexmem/macondo_gen.py`, `lexmem/sim.py`):
      Macondo's shell as a move-generation server (4 ms per position, all moves match our engine) and a 2-ply Monte-Carlo rollout
      search (K candidates by the net's policy, S common sampled worlds, opponent reply + my follow-up played by the net's argmax,
      exact scores summed). ~3.5 s/move at K=6,S=8 on one CPU core. Evaluations vs HastyBot pending (net rollouts K=6/S=8, K=4/S=24;
      HastyBot-equity rollouts K=4/S=24 as the code check — should reproduce sim > HastyBot).
      First run of the control gave **43.8 %**: the horizon was valued by raw score, rewarding tile dumps on the last ply.
      Fixed to value the last ply at equity (score + leave): still **43.4 %**. Remaining suspect: sampling noise (±6 pts at S=24
      vs candidate differences of a few points). Added successive halving (S=24 on K, then S2=96 on the top 2) and a control with
      candidates from Macondo's equity ranking. PPO pg4: 50.3 %, pg5: 50.9 % — flat; outcome-only path closed.
      **Rollout search results (2026-09-29, 2000 games each):** candidates = Macondo equity top-4, HastyBot-equity rollouts,
      24+96 worlds (successive halving): **52.1 % ± 1.1, +4.3 spread** — the search is correct and beats HastyBot as Quackle does.
      Same search with candidates = the net's policy top-4: **44.1 %**. Diagnosis: the net's argmax is HastyBot-level but its
      2nd–4th choices are uncalibrated (trained only on a single target move), so a sim over them picks weak moves.
      => Expert-iteration target: the sim's estimates over the candidate set (soft targets), which calibrates the ranking.
  (c) Macondo's own 5-ply sim as the expert for distillation — strong teacher, but its rollouts use HastyBot's leave table. Lesson (same as Macondo's): a static value net at this precision cannot drive move choice alone.

### 12e. Plan from 2026-09-29: the sim as a labeling machine (CPU) — where the lexicon layer gets a target that needs it
1. `label_sim.py`: for 300k HastyBot positions, sim estimates (2-ply, 64 worlds) for Macondo's top-8 candidates; stores score+leave
   equity too (job 22427679; ~2.7 positions/s on 48 procs -> cut at 12 h, ~100k positions).
2. `train_value.py`: V(post-move state) -> (sim estimate − move score). Variants: position encoder only; + lexicon branch with
   rack-level and per-lane queries on the post-move board (hottest 6 lanes). Baseline: affinely calibrated leave table.
   Dev set (2k positions): target spread 10.8 pts, leave-table baseline 6.8 pts. Jobs 22427956 (no-lex) / 22427958 (lex).
   **Result (40k positions / 320k pairs, 2k held-out positions):** target spread 10.5 pts; leave-table baseline **5.63**;
   plain value net **5.00** (corr 0.87 vs 0.82). First learned quantity here that a leave table cannot reproduce:
   board-conditioned future value of a leave. **Lexicon variant (rack-level + 6 hottest-lane queries, keys frozen, 6.8M):
   4.77 pts, corr 0.884** — best of the three. Size-matched plain control (6 transformer layers, 6.3M): **5.03 pts, corr 0.867**
   — the extra capacity alone buys nothing (5.00 → 5.03); the lexicon branch buys 0.26 pts (~5 % of the error, ~27 % of the
   net's gain over the leave table). **First target on which the sparse lexicon layer measurably helps, parameter-controlled.**
   One-ply learned-equity players: lexicon value 48.5 % vs plain value 47.2 % (both below the bare policy; the one-ply argmax
   amplifies per-state errors — the value belongs inside the sim, not alone).
3. Use V as one-ply lookahead (score + V, no sampling) and as the sim horizon, replacing the leave table.
   Sim with the learned horizon (no leave table at the horizon): **52.5 % ± 1.1, +7.2 spread (2000 games)** — better than
   the leave-table horizon (52.1 %, +4.3). One-ply learned equity alone (score + V over ~32 candidates, no sampling): 47.2 % —
   a 5-pt per-state error is amplified by the argmax; the leave table's errors are consistent across candidates, the net's are not.
   Expert-iteration round ei1 (soft targets τ=5 over 'both' candidates, 3500 games): bare net **47.1 %** (from 51.2), agreement
   with HastyBot 81 → 73 %: targets too noisy/flat, the prior eroded. Fix for ei2: τ=2, policy loss only where the sim's top-2
   margin ≥ 3 pts, KL anchor (β=0.5) to the previous policy, teacher = sim with learned horizon.
4. `train_belief.py`: opponent-rack belief from the last two opponent plays + board + unseen; baseline = uniform from unseen.
   **Result (job 22428069, 8k steps, 1M samples): per-tile NLL 2.597 vs uniform 2.594 — no gain.** The net converged to the
   uniform prior. Parked. To revisit: predict only the kept tiles (the drawn part is uniform by construction), compare with
   Macondo's Bayesian `infer` as the reference.
Expert iteration round ei1 (3500 sim games, soft targets) runs in parallel: jobs 22427697-22427700.
Round ei2 (τ=2, margin filter, KL anchor, learned-horizon teacher): bare net **50.5 % ± 1.1** (prior preserved, no gain);
sim over the net's own candidates 45.8 % (was 44.1 %). Policy improvement from sim rankings: still not demonstrated.

### 12f. Future-word α target and lattice recurrence (2026-09-30; inspired by Lattice Deduction Transformers, arXiv 2605.08605)
`label_sim.py FUT=1` records every word played in the sim's sampled futures (32 worlds × 4 candidates, my follow-up and the
opponent's reply, per lane). 150k HastyBot positions + 50k positions from our own sim games (on-policy); ~2.4M (state, side, lane)
units. `train_future.py`: for each open lane of the post-move board, rank the exact rules mask (pattern + cross-checks + letters
available to that side) against the frequency of the words actually played there. R=1 single pass vs R=3 lattice rounds (keep top
25 %, eliminated words never return, pooled survivors refine the query, every round supervised). Both init from the NWL23 retrieval
model; 12k steps × 48 on one GCD. Held-out: 8k positions, 2011 lane units (682 mine, 1329 opponent's); median mask 130 words.

| share of the future-word mass in the top 8 / top 32 | mine | opponent's |
|---|---|---|
| rank by score in the lane | 0.48 / 0.72 | 0.30 / 0.47 |
| retrieval model before this training (reachability target) | 0.65 / 0.82 | 0.24 / 0.40 |
| **trained, single pass** | **0.78 / 0.93** | **0.51 / 0.67** |
| trained, 3 lattice rounds | 0.78 / 0.92 | 0.50 / 0.67 |

Reading: the lexicon layer learns to predict the words that will actually be played next, and it does so far better than score
ranking — for my own follow-ups (0.78 vs 0.48 in the top 8) and, more tellingly, for the **opponent's hidden-rack replies (0.51 vs
0.30)**: it has learned which words the unseen pool is likely to feed them in each lane. The reachability-trained model was
*worse* than score ranking on the opponent side (0.24), so the gain is due to the new target. The 3-round lattice recurrence adds
nothing over a single pass here: with median masks of 130 words there is little to deduce iteratively; the paper's gains came from
long chains of hard constraints, which this one-step prediction does not have. Next: feed these predicted future-word sets (both
sides, per lane) as tokens into the value net and the policy, where they carry the "what can each side do next on this board"
information that the value-net ablation showed the lexicon helps with.

## 13. Related work (what exists, and what it doesn't cover)

**Scrabble-specific (nearly all of it):**
- Sheppard 2002, *World-championship-caliber Scrabble* (Maven): move gen + leave patterns + 2-ply sim + B* endgame. The counter-thesis above.
- Richards & Amir, IJCAI 2007, *Opponent Modeling in Scrabble*: Bayesian rack inference from observed plays; quantified the value of knowing the opponent's tiles. Hand-built, assumes the opponent maximizes static equity.
- Katz-Brown & O'Laughlin, Quackle (2005–): superleaves from millions of self-play games; same sim.
- Del Solar, Macondo/BestBot (2020–26): exact endgame + pre-endgame solvers, Bayesian inference, CNN static evaluator (pytorch/experiments.md). His 2022 essay lists what is missing: openness, board/bag-conditional leaves, inference, ML for "feel." All pieces of §6, framed as features rather than as one representation problem.
- 2019 bachelor thesis on imitation-learned CNN evaluation: same conclusion as Macondo — the board is a poor fit for a CNN.
- arXiv 2011: variance decomposition (two-sided draw) — useful for evaluation design.
- No published transformer, self-play RL, or learned-lexicon system for Scrabble exists.

**Adjacent methods (not applied to Scrabble):**
- Search under imperfect information: Student of Games (2023), DeepStack/ReBeL lineage, determinized/information-set MCTS.
- Model-free self-play at scale in large imperfect-information games: DeepNash (Stratego, 2022).
- Learned dynamics with chance: Stochastic MuZero (2022).
- External memory in a game agent: *Large-Scale Retrieval for RL* (NeurIPS 2022) — a Go agent attending into a retrieval store. Closest precedent for "lexicon as attendable memory."
- Large discrete / compositional action spaces: action embeddings (Dulac-Arnold 2015), text-game RL (DRRN, action elimination).
- Interpretability of game transformers: *Tracing the thought of a grandmaster-level chess transformer* (2026) — template for the probing we plan.
- LLMs at Scrabble: ~800 rating (ScrabbleBench); fail at letter counting and legality. Confirms that a general language prior is not a lexicon.

**Gap this project sits in:** imperfect information + stochastic draws + a rule set the size of a corpus, with the reachable-future set as a first-class representation. None of the above addresses that combination.

## 14. Proposed architecture (candidate — passed step-1 feasibility; steps 2–5 pending)

### The one new idea
A **lexicon memory layer** inside the transformer: 280k slots, one per word, sparse top-k retrieval (product-key memory, as in Lample et al. 2019 / PEER 2024), with the **rules applied as an exact mask before retrieval**. The model does not scan the lexicon and is not handed a list; it *queries* the lexicon under constraints, and the words it pulls become tokens the later layers reason over. Access = retrieval, not enumeration.

### Pieces
- **Environment / oracle:** Macondo or MAGPIE. Legality, scoring, exact endgame. Unchanged. Demoted from "state" to "rules."
- **Input tokens (per turn, ~20):** rack (7, unordered), exact unseen counts U (27), placed tiles with (row,col), scores/bag/turn, exchange/pass. Current board as 225 square tokens once. History of all turns. No engineered features.
- **Lower layers (~4):** mix rack + U + board + history into a query-ready residual.
- **Lexicon layer:** for each query point (rack token; each hot lane token; opponent-belief token) → query → mask → top-k words → retrieved word tokens appended, each carrying which letters I hold / need / could draw. Keys and values initialised from a letter-composition encoder pretrained on the full lexicon (so rare words start meaningful). Opponent query is masked by the belief instead of my rack.
- **Upper layers (~8):** reason over game tokens + retrieved word tokens. Compare targets, notice shared letters, weigh against belief.
- **Heads:** policy over (retrieved word × placement); value (win, spread at several horizons); opponent rack (27×3); future words I / they will play in the next k turns (over the lexicon, trained retrospectively from self-play).
- **Planning:** search in word space. At every imagined node the board, racks, U and the mask are updated **exactly**; only draws (chance node) and the opponent's move (belief-sampled or learned step) are stochastic/learned. AlphaZero-with-simulator for the rules, MuZero only for hidden information.
- **Training:** (0) lexicon pretraining, uniform over words; (1) constructed-position curriculum — build positions backwards from a target word so every word is the right retrieval somewhere, then delayed versions where the word is 1–3 draws away; (2) self-play with search; (3) expert iteration: search visit counts over word-tokens train the retrieval layer.
- **Per language:** swap lexicon file + tile table.
- **Human-facing output:** the k words considered each turn, for both sides.

### Step-1 feasibility numbers (ENABLE, 168k words; board constraint is an upper bound, cross-checks ignored)

| tiles on board | fits some lane, ever | uses ≥3 of my rack | uses ≥5 of my rack | playable now |
|---|---|---|---|---|
| 0 | 52k | 13k | 254 | 38 |
| 20 | 74k | 17k | 492 | 220 |
| 35 | 72k | 12k | 163 | 362 |
| 50 | 56k | 7.6k | 17 | 644 |
| 65 | 27k | 5.2k | 104 | 229 |

Turn-to-turn overlap of the ≥3 set: Jaccard ≈ 0.54. Tile-feasibility alone (no board) is useless early: 168k of 168k alive on turn 1.

**Reading:** the rules mask cuts ~10× (worth computing, cheap) but leaves ~10⁴ candidates (so the layer must *learn* to rank — the rules alone don't solve it). Top-k of 32–64 is ~0.5% of the planning set: a normal retrieval ratio. Half the reachable set persists across a turn: enough to plan on. Nothing rules the design out. Scripts: `feas1_alive.py`, `feas1b_board.py`.

### Remaining gates
2. Static retrieval on constructed positions: every word retrievable when it is the target; top-k vs GADDAG truth.
3. Delayed retrieval: target 1–3 draws away — does the layer pull it *early*? (The "can it plan" gate.)
4. In a policy/value net, self-play on Macondo, vs Macondo's CNN at equal compute.
5. Word-space search on top.

### Step-2 pilot (CPU, 3.5 min, 80k words ≤8 letters, d=96) — `feas2_retrieval.py`
Query = MLP(rack, blanks, unseen U, lane pattern). Key = letter-composition encoder + per-word slot. Score = dot, masked by the rules, sampled-softmax training on constructed positions.

Test that matters: **DELAYED** — the seed word needs 1–2 letters still in the bag (cannot be played yet); mask ≈ 450 words fitting the lane with ≥ L−2 rack letters. Metric: fraction of the mask ranked above the seed (random = 0.50).

| | mean | median |
|---|---|---|
| untrained | 0.52 | 0.53 |
| trained 3.5 min CPU | 0.34 | 0.28 |

Reading: a dense query pushes a word it *cannot yet play* into the top third of a ~450-word candidate set with negligible compute. The mechanism learns in the right direction. Not yet a pass: the target is ambiguous (many masked words are equally reachable), so this is a floor. Two other metrics in the script (NOW recall, coverage) are degenerate (mask ≈ label) and should be ignored.

Proper step 2 needs: GPU, full lexicon, d≥256, real boards from Macondo, and a well-posed label = all words reachable within ≤2 draws weighted by draw probability × score (exactly computable from U + lexicon).

### Step-2 result (2026-09-26, full ENABLE 168k words, d=256, real self-play boards, RTX 3080, 7 min) — `feas2_lexmem.py`

Setup. Boards from 600 self-play games (`gen_games.py`, exact legality/cross-checks/scoring, softmax-over-score bot).
Query = 3-layer transformer over [rack, unseen U, globals, board summary, 15 lane tokens (fixed letter, cross-check set, premium)].
Key = 2-layer letter transformer + per-word slot. Score = dot, masked by the rules (pattern + cross-checks).
**Label (well-posed):** for (rack, U, lane), every word fitting the lane whose letters not on the rack (≤3, blanks cover)
are available in U, weighted by P(a draw of 7−kept tiles from U completes it) × score in that lane. Exact (hypergeometric).
Train: listwise softmax, target = ½ weight-normalised + ½ uniform over the reachable set. 4000 steps × 32 positions.
5% of the lexicon held out: never a positive in training, present as candidates at test.
Test on 100 unseen games; 400 real-rack lanes + 400 constructed (rack built around a random candidate word, 0–3 letters left in U).

| metric (real racks, test games) | untrained | GADDAG-style baseline* | trained |
|---|---|---|---|
| NDCG@32 (gains = weight) | 0.44 | 0.51 | **0.92** |
| NDCG@32, masks of 5000+ candidates | 0.01 | – | **0.85** |
| recall@32 of top-32-by-weight | 0.55 | – | 0.85 |
| top-1 word is reachable | 0.65 | – | 0.98 |

*playable-now words by score, everything else random.

Percentile of a reachable word inside the mask (random = 0.50, lower = better), by letters still in the bag:

| missing letters | 0 (playable now) | 1 | 2 | 3 |
|---|---|---|---|---|
| all words | 0.07 | 0.16 | 0.24 | 0.34 |
| held-out words (never trained as positive) | 0.07 | 0.15 | 0.22 | 0.31 |
| constructed racks | 0.06 | 0.19 | 0.29 | 0.39 |

Reading. (1) The dense query ranks a 10⁴-candidate mask sharply (§11.3 answered for the static case): NDCG 0.85 on the
largest masks. (2) Words 1–3 draws away are pulled well above chance, monotonically harder with distance; the pilot's
0.28 becomes 0.16/0.24/0.34 on a well-posed label. (3) Held-out words behave like trained words: the letter-composition
encoder, not the per-word slot, carries the retrieval — rare words are not a problem. (4) Loss was still decreasing;
no hyperparameter search was done. **Gate 2 passed; gate 3 (delayed) passed at the level of "well above chance", not solved.**

Caveats: the label is an exact function of the inputs, so this is function approximation, not strategy. Boards come from
a weak bot. Blanks in U are ignored in P; blank score penalty is approximate.

### Step-4 script (written, not run): `feas4_latent.py`
MuZero-style: h(obs)→latent, g(latent, placed-tiles plane, drawn tiles)→latent, lexicon head f(latent, lane)→words,
value head, EfficientZero consistency loss. Unrolled K=4 half-moves on true trajectories. Gate: NDCG / percentile from the
k-step imagined latent vs from h(obs_k) directly, as a function of k. To be run on LUMI (init word encoder from `data/lexmem_base.pt`).

### What "learning the game" needs (gate 4, LUMI)
Policy/value net with the lexicon layer vs the same net without it, equal compute, self-play (Macondo/MAGPIE as oracle).
The layer only matters if the delayed-word signal moves win rate. Everything above is a prerequisite, not that proof.

### LUMI runs (2026-09-27, project_462001611, one MI250X GCD each; 4000 train / 300 test games)

**Step 2 at scale** (`feas2_lexmem.py`, 40k steps × 64 positions, ~2 h). With and without the per-word slot table:

| real racks, 400 test lanes | local (600 games, 4k steps) | LUMI no slots | LUMI with slots |
|---|---|---|---|
| NDCG@32 | 0.919 | **0.973** | 0.970 |
| NDCG@32 on masks ≥ 5000 words | 0.853 | 0.966 | 0.969 |
| recall@32 of top-32-by-weight | 0.854 | 0.930 | 0.927 |
| percentile, 0/1/2/3 letters missing | .07/.16/.24/.34 | .07/.19/.24/.32 | .07/.19/.24/.32 |
| held-out words, same | .07/.15/.22/.31 | .06/.13/.24/.32 | .04/.10/.20/.29 |
| GADDAG-style baseline | 0.505 | 0.537 | 0.537 |

Reading: ten times the data and steps push NDCG to 0.97 and close the large-mask gap. **Slots add nothing** (3.5M-parameter
model = 46.5M-parameter model): the letter-composition encoder carries retrieval; drop the table. The delayed-word
percentiles do **not** move with scale (≈0.19/0.24/0.32 for 1/2/3 missing): that is a floor of this label (many words
equally reachable), not of the model. A sharper delayed test needs a label with a unique answer (e.g. retrospective:
the word actually played k turns later).

**Whole-lexicon coverage sweep** (`eval_coverage.py`, no-slot checkpoint, every one of 168,551 words × 0/1/2 letters
left in the bag, on real test-board lanes; 99.4 % of items constructible; 8 min on one GCD):

| letters in bag | in top-32 of its mask | top-1 | mean percentile | unmasked median rank (of 168k) | unmasked top-32 |
|---|---|---|---|---|---|
| 0 | **1.000** | 0.88 | 0.004 | 9 | 0.60 |
| 1 | 0.96 | 0.64 | 0.016 | 169 | 0.29 |
| 2 | 0.82 | 0.48 | 0.042 | 1336 | 0.08 |

Every playable word in the dictionary is retrieved in the top-32 of its rules mask; 302 of 168k land in the bottom half and
all of those are masks of 2–5 words where a higher-scoring word also fits (ambiguity, not failure). With 2 letters missing,
short words (3–6 letters, top-32 rate 0.41–0.68) are hard because hundreds of words are equally reachable; words of ≥ 8
letters are at 0.86–1.00 because rack + fixed letters pin them down. Unmasked (no rules at all) a playable word sits at
median rank 9 of 168k: the query has internalised most of the rules, but the exact mask still does real work for
delayed words. **Gate 2 fully passed; the per-word coverage requirement of §14 is met.**

**Step 4, latent dynamics** (`feas4_latent.py`, 20k steps × 48 windows of K=4 half-moves, ~3 h; word encoder initialised
from step 2). Retrieval NDCG@32 from the observation encoder h(o_k) ("direct") vs from the latent imagined k half-moves
ahead through g(·, true action, true draw) ("imagined"); chance ≈ 0.43:

| k (half-moves) | 0 | 1 (opp. to move) | 2 (me) | 3 (opp.) | 4 (me) |
|---|---|---|---|---|---|
| direct | 0.954 | 0.951 | 0.949 | 0.952 | 0.947 |
| imagined | 0.954 | 0.638 | **0.872** | 0.823 | **0.836** |
| cos(imagined, direct) | 1.00 | 0.55 | 0.84 | 0.44 | 0.73 |

Reading: at even k the future rack and board are fully determined by the actions, and the imagined latent keeps most of
the lexicon-level detail (0.87 at two half-moves, 0.84 at four; still rising at the end of training). At odd k the mover's
rack is hidden from the starting observation, so imagined retrieval is capped by the belief problem; the rise from k=1
(0.64) to k=3 (0.82) shows the dynamics model learns some rack inference from the opponent's play and draw. **The learned
latent does not smooth away lexicon detail (§11.4); the loss is concentrated at the hidden-information step.** Value
head MAE 73 spread points (weak: single-position spread prediction from a random-ish bot's games; not the point of the test).

**Not tested by any of the above:** strategy. All labels are computable functions of the position. Gate 4 (policy/value
with vs without the layer, self-play, equal compute) remains the proof that the model learns the game.
