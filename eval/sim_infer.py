"""Rack modelling inside Macondo's simulator, position level (CPU only; lumi/sim_infer.sh).
Positions: the rack-inference held-out positions from the side of the player who moves next (data/planinf_*.jsonl of
eval/plan_infer.py: the board after HastyBot's play, our rack; "okeep" = kept-tile samples of one inference source).
Decision: Macondo's SIMMING_BOT simulation as it is (go/lexmem_simrack MODE=choose: top 100 plays by static equity, 2 plies,
Stop99, one thread, SECS per position); the arms differ only in the opponent's rack in each iteration (uniform from the unseen
tiles = SIMMING_BOT; kept tiles from an inference source + refill; the true kept tiles + refill) or in SECS.
Value of a choice: the same simulation with the true kept tiles + refill, a fixed number of iterations and no pruning
(MODE=eval), on the plays any arm chose; the plays of a position share every iteration's opponent rack and bag order.
Reported per arm: the win probability and equity of its choice under that evaluation, paired against the base arm.
Positions with at most 8 unseen tiles are not simulated by SIMMING_BOT (solver stage): no difference between arms there;
they count as zeros in the "all positions" column.
prep-eval ARM ...   data/<PREFIX>_<arm>.jsonl -> data/<PREFIX>_evalin<EVAL>.jsonl
analyze ARM ...     <- data/<PREFIX>_eval<EVAL>.jsonl; the first arm is the base. Env PREFIX (default simrack), EVAL (a suffix
                    that names another evaluation of the same arms, e.g. _end: rollouts to the end of the game,
                    lumi/sim_depth.sh; default none).
Stdlib only (runs with /usr/bin/python3)."""
import os, sys, json, math
D = "data/"; PRE = os.environ.get("PREFIX", "simrack"); EV = os.environ.get("EVAL", "")

def load(arm):
    return {o["id"]: o for o in map(json.loads, open("%s%s_%s.jsonl" % (D, PRE, arm)))}

def ms(v):   # mean, standard error
    n = len(v); m = sum(v) / n
    return m, (math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1) / n) if n > 1 else float("nan"))

def prep(arms):
    A = {a: load(a) for a in arms}; n = 0
    with open("%s%s_evalin%s.jsonl" % (D, PRE, EV), "w") as f:
        for t in map(json.loads, open(D + "planinf_true.jsonl")):
            i = t["id"]; plays = sorted({A[a][i]["best"] for a in arms if i in A[a] and A[a][i].get("best")})
            if not plays: continue
            f.write(json.dumps(dict(id=i, cgp=t["cgp"], okeep=t["okeep"], plays=plays)) + "\n"); n += 1
    print("%d positions with a simulated choice -> %s%s_evalin%s.jsonl" % (n, D, PRE, EV), flush=True)

def analyze(arms):
    A = {a: load(a) for a in arms}; base = arms[0]
    E = {o["id"]: {c["p"]: c for c in o.get("cands", [])} for o in map(json.loads, open("%s%s_eval%s.jsonl" % (D, PRE, EV))) if not o.get("err")}
    ids = sorted(set.intersection(*[set(A[a]) for a in arms]))
    err = [i for i in ids if any(A[a][i].get("err") for a in arms)]
    solver = [i for i in ids if i not in err and all(A[a][i]["plies"] == 0 for a in arms)]
    sim = [i for i in ids if i not in err and i in E and all(A[a][i].get("best") in E[i] for a in arms)]
    frac = len(sim) / max(len(sim) + len(solver), 1)
    its = ms([next(iter(E[i].values()))["n"] for i in sim])[0] if sim else 0
    print("%d positions: %d simulated and evaluated, %d solver stage (no rack sampled), %d errors, %d not evaluated; evaluation %d iterations per play"
          % (len(ids), len(sim), len(solver), len(err), len(ids) - len(sim) - len(solver) - len(err), its))
    print("value of each arm's choice under the true kept tiles; paired vs %s (positive = better); 'all' = simulated positions x %.3f" % (base, frac))
    val = {a: ([100 * E[i][A[a][i]["best"]]["w"] for i in sim], [E[i][A[a][i]["best"]]["e"] for i in sim]) for a in arms}
    for a in arms:
        w, e = val[a]; it = ms([A[a][i]["iters"] for i in sim])[0]; sec = ms([A[a][i].get("secs", 0) for i in sim])[0]
        line = "  %-28s win-%% %.3f equity %+.2f | %5.0f iterations %.1f s" % (a, ms(w)[0], ms(e)[0], it, sec)
        if a != base:
            dw, de = ms([x - y for x, y in zip(w, val[base][0])]), ms([x - y for x, y in zip(e, val[base][1])])
            ch = sum(A[a][i]["best"] != A[base][i]["best"] for i in sim) / len(sim)
            line += " | vs base: win-%% %+.3f +- %.3f, equity %+.3f +- %.3f, other play in %.1f %% | all: win-%% %+.3f, equity %+.3f" % (
                dw[0], dw[1], de[0], de[1], 100 * ch, dw[0] * frac, de[0] * frac)
        print(line)
    lex = [a for a in arms if "_lex_" in a]; nol = [a for a in arms if "_nolex_" in a]
    if lex and nol:
        mean = lambda grp, k: [sum(val[a][k][j] for a in grp) / len(grp) for j in range(len(sim))]
        dw, de = ms([x - y for x, y in zip(mean(lex, 0), mean(nol, 0))]), ms([x - y for x, y in zip(mean(lex, 1), mean(nol, 1))])
        print("memory - control (seed means): win-%% %+.3f +- %.3f, equity %+.3f +- %.3f" % (dw[0], dw[1], de[0], de[1]))

if __name__ == "__main__":
    (prep if sys.argv[1] == "prep-eval" else analyze)(sys.argv[2:])
