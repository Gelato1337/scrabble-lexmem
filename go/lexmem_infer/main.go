// lexmem_infer: Macondo's rack inference (the rangefinder that SIMMING_INFER_BOT runs) as the reference for the rack-inference
// benchmark (train/train_lexinfer.py, eval/infer_macondo.py). One position per input line:
//   {"i":..,"cgp":"<position before the opponent's tile play: opponent on turn, its rack = the tiles it placed / our rack>",
//    "coords":"8D","tiles":"WO.D","played":"WOD","leave":"AE?RS"}
// The game is rebuilt as that CGP + the play (one history event; the opponent's refill goes back into the bag, as in a bare
// CGP), and the rangefinder infers the opponent's kept tiles, bounded by BUDGET leaves measured (default: Macondo's
// rangefinder.DefaultPairedInferenceBudget, the setting of its paired bot experiments) or, with BUDGET=0, by TIME seconds.
// Output per line: {"i":..,"n":inferred leaves,"complete":..,"marg":[27][4] = P(kept count 0 / 1 / 2 / 3+) per tile in our
// order (A..Z, blank), "post": posterior of the true leave, "prior": its prior, "lift": log2(post/prior) (-999 if undefined),
// "secs":..,"err":..}
// Env: BUDGET TIME THREADS (rangefinder threads per position) WORKERS (positions in parallel); diagnostics (default: Macondo's
// settings): PROPOSAL=prior|floor (refine rounds draw from the prior / keep a FLOOR share of prior mass, default 0.2),
// TAU (pinned softmax temperature of P(play | leave))
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/rs/zerolog"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/rangefinder"
)

type In struct {
	I      int    `json:"i"`
	CGP    string `json:"cgp"`
	Coords string `json:"coords"`
	Tiles  string `json:"tiles"`
	Played string `json:"played"`
	Leave  string `json:"leave"`
}
type Out struct {
	I        int          `json:"i"`
	N        int          `json:"n"`
	Complete bool         `json:"complete"`
	Marg     [27][4]float64 `json:"marg"`
	Post     float64      `json:"post"`
	Prior    float64      `json:"prior"`
	Lift     float64      `json:"lift"`
	Secs     float64      `json:"secs"`
	Err      string       `json:"err,omitempty"`
}

func envI(k string, d int) int {
	if v, err := strconv.Atoi(os.Getenv(k)); err == nil { return v }
	return d
}
func envF(k string, d float64) float64 {
	if v, err := strconv.ParseFloat(os.Getenv(k), 64); err == nil { return v }
	return d
}

// ours: Macondo machine letter (0 = blank, 1..26 = A..Z) -> our tile index (0..25 = A..Z, 26 = blank)
func ours(ml tilemapping.MachineLetter) int {
	if ml == 0 { return 26 }
	return int(ml) - 1
}

func infer(cfg *config.Config, calc equity.EquityCalculator, x In, budget, secs, threads int) (out Out) {
	out = Out{I: x.I, Lift: -999}
	t0 := time.Now()
	defer func() { out.Secs = time.Since(t0).Seconds() }()
	pg, err := cgp.ParseCGP(cfg, x.CGP)
	if err != nil { out.Err = err.Error(); return out }
	g := pg.Game; g.RecalculateBoard(); g.SetBackupMode(game.InteractiveGameplayMode); g.SetStateStackLength(1)
	m, err := g.CreateAndScorePlacementMove(x.Coords, x.Tiles, x.Played, false)
	if err != nil { out.Err = err.Error(); return out }
	opp := g.PlayerOnTurn()
	if err := g.PlayMove(m, true, 0); err != nil { out.Err = err.Error(); return out }
	g.ThrowRacksInFor(opp)
	rf := &rangefinder.RangeFinder{}
	rf.Init(g, []equity.EquityCalculator{calc}, cfg); rf.SetThreads(threads); rf.SetBudget(budget)
	switch os.Getenv("PROPOSAL") {
	case "prior": rf.SetProposalMode(rangefinder.ProposalPrior)
	case "floor": rf.SetProposalMode(rangefinder.ProposalFloor); rf.SetExplorationFloor(envF("FLOOR", 0.2))
	}
	if tau := envF("TAU", 0); tau > 0 { rf.SetTau(tau) }
	if err := rf.PrepareFinder(g.RackFor(g.PlayerOnTurn()).TilesOn()); err != nil { out.Err = err.Error(); return out }
	var ctx context.Context; var cancel context.CancelFunc
	if budget > 0 { ctx, cancel = context.WithCancel(context.Background()) } else { ctx, cancel = context.WithTimeout(context.Background(), time.Duration(secs)*time.Second) }
	defer cancel()
	if err := rf.Infer(ctx); err != nil { out.Err = "infer: " + err.Error() }
	inf := rf.Inferences()
	if inf == nil || len(inf.InferredRacks) == 0 { if out.Err == "" { out.Err = "no inferences" }; return out }
	out.N, out.Complete = len(inf.InferredRacks), inf.Complete
	sumW := 0.0
	for _, ir := range inf.InferredRacks {
		var cnt [27]int
		for _, ml := range ir.Leave { cnt[ours(ml)]++ }
		for t := 0; t < 27; t++ { k := cnt[t]; if k > 3 { k = 3 }; out.Marg[t][k] += ir.Weight }
		sumW += ir.Weight
	}
	if sumW > 0 { for t := 0; t < 27; t++ { for k := 0; k < 4; k++ { out.Marg[t][k] /= sumW } } }
	leave, err := tilemapping.ToMachineLetters(x.Leave, g.Alphabet())
	if err == nil {
		sc := rf.ScoreLeave(leave); out.Post, out.Prior = sc.Posterior, sc.Prior
		if !math.IsNaN(sc.LiftBits) && !math.IsInf(sc.LiftBits, 0) { out.Lift = sc.LiftBits }
	}
	return out
}

func main() {
	zerolog.SetGlobalLevel(zerolog.ErrorLevel)
	ex, _ := os.Executable(); cfg := &config.Config{}; cfg.Load(nil); cfg.AdjustRelativePaths(filepath.Dir(ex))
	budget, secs, threads, W := envI("BUDGET", rangefinder.DefaultPairedInferenceBudget), envI("TIME", 20), envI("THREADS", 4), envI("WORKERS", 8)
	in := make(chan In, 256); outc := make(chan Out, 256); var wg sync.WaitGroup
	for w := 0; w < W; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			calc, err := equity.NewCombinedStaticCalculator("NWL23", cfg, "", equity.PEGAdjustmentFilename)
			if err != nil { panic(err) }
			for x := range in { outc <- infer(cfg, calc, x, budget, secs, threads) }
		}()
	}
	go func() { wg.Wait(); close(outc) }()
	go func() {
		sc := bufio.NewScanner(os.Stdin); sc.Buffer(make([]byte, 1<<20), 1<<20)
		for sc.Scan() { var x In; if json.Unmarshal(sc.Bytes(), &x) == nil { in <- x } }
		close(in)
	}()
	bw := bufio.NewWriterSize(os.Stdout, 1<<20); enc := json.NewEncoder(bw)
	for o := range outc { enc.Encode(o); bw.Flush() }
	bw.Flush()
}
