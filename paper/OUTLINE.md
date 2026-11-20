# Paper outline (working draft, 2026-11-20)

No deadline: the outline collects the claim, the evidence we have, and the experiments still needed. Numbers come from
`docs/benchmarks.md` (provenance there); figures from `docs/figures/`.

## Working titles
- *A Lexicon Memory for Lookahead: Learning Where Scrabble's Rules Engine Cannot Help*
- *Remembering the Dictionary: An Explicit Lexicon Memory for Planning in Scrabble*

## Thesis
A policy/value network for Scrabble gets every legal word for the current move from the rules engine, so a memory over
the dictionary adds nothing at the current move. It adds what the engine does not enumerate: which words become possible
later (the opponent's replies, my next turn). An explicit, sparse retrieval layer over the whole lexicon (letter-composition
keys, feasibility mask from the tiles that can still appear, top-k retrieval) makes that foresight learnable from less
data, and at two turns of lookahead also at full data. This frees the model from enumeration-based search for lookahead,
the precondition for learning foresight from experience (self-play) instead of computing it by simulation.

## Contributions
1. **The memory layer.** Retrieval over all 196,601 NWL23 words in one pass: NDCG@32 0.97 (GADDAG-style baseline 0.54);
   held-out words behave like trained ones; retrieval from an imagined position two half-moves ahead 0.87 (0.95 direct).
2. **An exact planning benchmark.** For each candidate move: the opponent's expected best reply and my expected best next
   play by enumeration over the dictionary on shared tile samples (one and two turns); no teacher, no value net. Metrics:
   within-position R², critical-position regret. Includes the labeller bug we found and fixed (Macondo shadow bounds).
3. **Controlled comparisons that separate the memory from its confounds.** Frozen vs trainable vs from-scratch encoders,
   capacity matched, with and without cross-check sets, two seeds, 3k / 10k / 270k positions, NWL23 and zero-shot CSW24.
4. **Where it shows up in play.** Planning policies in games (memory +2.2 ± 0.6 at 3k, gone by 270k); the net as a move
   picker; the system context (one-pass net 53.2 % vs HastyBot, + endgame solvers 58.1 %, + search 60.1 %, teacher level).
5. **What self-play did and did not do.** Q-learning collapse, gated actor-critic null, expert iteration null; why
   (outcome noise sd 0.58 vs effects 0.01–0.03; search labels of teacher quality add nothing to a net trained on 715k).

## Sections and evidence
1. **Introduction.** Scrabble: imperfect information, chance, a lexicon-defined action space (hundreds of legal moves).
   Engines enumerate moves and simulate (Maven, Quackle, Macondo/BestBot: minutes per move, hand-built leaves, Bayesian
   rack inference). Question: can a learned model get foresight about words without enumerating them?
2. **Related work.** Scrabble engines and Macondo's CNN evaluator (51.5–52.6 % vs HastyBot); memory-augmented networks
   (product-key memories, kNN-LM, Memorizing Transformers, RETRO); AlphaZero / MuZero / Stochastic MuZero, expert iteration;
   learning in stochastic games.
3. **Model.** Rules engine and position encoding; move tokens; the lexicon memory (query encoder, word encoder,
   feasibility mask, top-k); heads. Inference path (dense GPU masks, identical to the training path).
4. **The memory retrieves the right words** (contribution 1). Fig.: retrieval quality vs mask size; held-out words.
5. **Exact planning benchmark** (contribution 2). Fig.: label construction; one vs two turns (two-turn answer differs in 29 %).
6. **Controlled results** (contribution 3).
   - One turn: memory = sample-efficiency prior (R² E_opp 0.20 vs 0.00 at 10k), control catches up by 270k.
   - Isolation: no cross-check sets, no collapse (refutes our first explanation).
   - Encoder confounds: from-scratch control decides as well as the frozen-encoder memory arm; capacity-matched memory
     beats the same-start control everywhere, the best control only at 3k.
   - Two turns: memory 3.14 vs 3.47 critical regret at 270k against the same-start control (same pretrained, trained
     encoder; 3.56 against the clean control), better at every size and seed pair: the only full-data decision gain.
   - Dictionary transfer: smaller benchmark loss zero-shot on CSW24, not in games.
   Figs.: `lexplan_within_r2`, `lexplan_no_xchk`, `lexplan_clean_control`, `lexplan_capacity_matched`, `lexplan_two_turn`,
   `lexplan_csw_transfer`.
7. **In games** (contribution 4). Figs.: `lexplan_games`, `win_ladder`, `endgame_hybrid`, `cost_by_phase`.
8. **Self-play** (contribution 5). Table of attempts and costs.
9. **Discussion.** The memory adds information only where the engine does not enumerate; foresight as the use case;
   outlook: learning foresight from self-play at scale (the memory makes the long-horizon part learnable in one pass).

## Open experiments (what would make the central claim)
| Experiment | Question | Status / cost |
|---|---|---|
| Two-turn same-start control | Is the two-turn full-data gain the memory or the pretrained encoder? | **done: the memory** (−0.33 at 270k) |
| Horizon scaling (1 → 2 → 3 turns) | Does the memory's advantage grow with the lookahead horizon? | needs a three-turn labeller; cost to estimate |
| Own search leaf (memory vs control) | Can a value net learned from search labels replace Macondo's leaf, and does the memory help it? | **done: no** (+2.5 / +2.2 vs `mac` +8.5 / +10.0 agreement; memory no help) |
| Search over the memory vs control net | Does the memory make the strongest system stronger? | no memory done (+2.0 ± 1.3 over its hybrid); memory arm lost to node failures, rerun ≈ 20 GPU-h undecided |
| Strength per compute | Where does one forward pass sit on Macondo's simulation-time curve? | proposed, CPU jobs |
| Rack inference with the memory | Can the memory read the opponent's rack from what they did not play, in one pass? | proposed |
| Outcome-RL scaling test | Does the memory learn foresight from self-play outcomes faster as games grow (small scale only)? | idea; not planned |

## Venues (to check: deadlines, JUFO levels)
IEEE Conference on Games (CoG); IEEE Transactions on Games; ICGA Journal; NeurIPS / ICLR workshops on memory or games.
