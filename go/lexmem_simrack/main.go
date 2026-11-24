// lexmem_simrack: does modelling the opponent's rack change what Macondo's simulator plays, and is the change better?
// Each position is simulated exactly as SIMMING_BOT does it (ai/bot LexmemSim = the set-up of nonEndgameBest: the top 100
// plays by static equity, 2 plies, or the number of unseen tiles when 9-14 are unseen, Stop99, one thread); the only thing
// that varies is where the opponent's rack comes from in each iteration:
//
//	no "okeep"     a uniform rack from the tiles unseen to the mover (SIMMING_BOT as it is)
//	"okeep": [..]  kept tiles drawn from this list (equal weights; repeat a rack to weight it) + a refill from the bag:
//	               posterior samples of an inference method, or the one true kept rack (perfect inference)
//
// stdin: one JSON per line {"id", "cgp" (mover on turn, its rack only), "okeep"?, "plays"? (MODE=eval)}; the files
// data/planinf_<cond>.jsonl of eval/plan_infer.py have this form. stdout: one JSON per line.
//
//	MODE=choose (default): SECS seconds per position (default 10) -> "best", "iters", the top CANDS plays by win probability
//	               ("p" play, "w" win probability, "e" equity, "n" iterations).
//	MODE=eval:     exactly ITERS iterations (default 4000) over the plays in "plays" only, no stopping rule, nothing pruned
//	               -> every play's "w" / "e"; plays not among the top 100 are listed in "miss". Every play of an iteration
//	               meets the same opponent rack and bag order, so differences between plays are paired.
//
// "plies": 0 = at most 8 tiles unseen: SIMMING_BOT hands the position to its pre-endgame / endgame solver and samples no
// rack, so the position is skipped (no difference between rack sources there).
// Env: MODE SECS ITERS CANDS (default 5) WORKERS (default 8; one single-threaded simulation each).
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
)

type In struct {
	ID    int      `json:"id"`
	CGP   string   `json:"cgp"`
	OKeep []string `json:"okeep,omitempty"`
	Plays []string `json:"plays,omitempty"`
}

type Cand struct {
	P string  `json:"p"`
	W float64 `json:"w"`
	E float64 `json:"e"`
	N int     `json:"n"`
}

type Out struct {
	ID     int      `json:"id"`
	Plies  int      `json:"plies"`
	Best   string   `json:"best,omitempty"`
	Iters  int      `json:"iters,omitempty"`
	NMoves int      `json:"nmoves,omitempty"`
	NKeep  int      `json:"nkeep,omitempty"`
	Cands  []Cand   `json:"cands,omitempty"`
	Miss   []string `json:"miss,omitempty"`
	Secs   float64  `json:"secs,omitempty"`
	Err    string   `json:"err,omitempty"`
}

func envI(k string, d int) int {
	if v, err := strconv.Atoi(os.Getenv(k)); err == nil {
		return v
	}
	return d
}

func run(conf *bot.BotConfig, in In, eval bool, secs float64, iters, ncand int) (out Out) {
	out.ID = in.ID
	defer func() {
		if r := recover(); r != nil {
			out.Err = fmt.Sprintf("panic: %v", r)
		}
	}()
	pg, err := cgp.ParseCGP(&conf.Config, in.CGP)
	if err != nil {
		out.Err = err.Error()
		return
	}
	pg.RecalculateBoard()
	pg.SetBackupMode(game.InteractiveGameplayMode)
	pg.SetStateStackLength(1)
	tp, err := bot.NewBotTurnPlayerFromGame(pg.Game, conf, pb.BotRequest_SIMMING_BOT)
	if err != nil {
		out.Err = err.Error()
		return
	}
	out.Plies = tp.LexmemSimPlies()
	if out.Plies == 0 {
		return
	}
	var racks []montecarlo.InferredRack
	for _, k := range in.OKeep {
		mls, err := tilemapping.ToMachineLetters(k, pg.Alphabet())
		if err != nil {
			out.Err = "okeep: " + err.Error()
			return
		}
		racks = append(racks, montecarlo.InferredRack{Leave: mls, Weight: 1})
	}
	out.NKeep = len(racks)
	moves := tp.GenerateMoves(100) // SIMMING_BOT's candidate list
	out.NMoves = len(moves)
	t0 := time.Now()
	if eval {
		var sel []*move.Move
		for _, want := range in.Plays {
			found := false
			for _, m := range moves {
				if strings.TrimSpace(m.ShortDescription()) == want {
					sel, found = append(sel, m), true
					break
				}
			}
			if !found {
				out.Miss = append(out.Miss, want)
			}
		}
		if len(sel) == 0 {
			return
		}
		err = tp.LexmemSim(context.Background(), out.Plies, sel, racks, iters)
	} else {
		ctx, cancel := context.WithTimeout(context.Background(), time.Duration(secs*float64(time.Second)))
		err = tp.LexmemSim(ctx, out.Plies, moves, racks, 0)
		cancel()
	}
	if err != nil {
		out.Err = err.Error()
		return
	}
	out.Secs = time.Since(t0).Seconds()
	sm := tp.Simmer()
	out.Iters = sm.Iterations()
	if !eval {
		out.Best = strings.TrimSpace(sm.WinningPlay().Move().ShortDescription())
	}
	for i, sp := range sm.PlaysByWinProb().PlaysNoLock() {
		if !eval && i >= ncand {
			break
		}
		out.Cands = append(out.Cands, Cand{P: strings.TrimSpace(sp.Move().ShortDescription()), W: sp.WinProb(), E: sp.EquityMean(), N: sp.WinProbIterations()})
	}
	return
}

func main() {
	zerolog.SetGlobalLevel(zerolog.ErrorLevel)
	ex, _ := os.Executable()
	cfg := &config.Config{}
	cfg.Load(os.Args[1:])
	cfg.AdjustRelativePaths(filepath.Dir(ex))
	conf := &bot.BotConfig{Config: *cfg, SimThreads: 1} // one shared config, built before the workers start (Config holds a lock)
	eval := os.Getenv("MODE") == "eval"
	secs := 10.0
	if v, err := strconv.ParseFloat(os.Getenv("SECS"), 64); err == nil {
		secs = v
	}
	iters, ncand, W := envI("ITERS", 4000), envI("CANDS", 5), envI("WORKERS", 8)
	in := make(chan In, 1024)
	outc := make(chan Out, 1024)
	var wg sync.WaitGroup
	for w := 0; w < W; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for x := range in {
				outc <- run(conf, x, eval, secs, iters, ncand)
			}
		}()
	}
	go func() { wg.Wait(); close(outc) }()
	go func() {
		sc := bufio.NewScanner(os.Stdin)
		sc.Buffer(make([]byte, 1<<22), 1<<22)
		for sc.Scan() {
			var x In
			if err := json.Unmarshal(sc.Bytes(), &x); err != nil {
				fmt.Fprintf(os.Stderr, "bad line: %v\n", err)
				continue
			}
			in <- x
		}
		close(in)
	}()
	w := bufio.NewWriter(os.Stdout)
	defer w.Flush()
	n, nsim, nerr, its := 0, 0, 0, 0
	for o := range outc {
		b, _ := json.Marshal(o)
		w.Write(b)
		w.WriteByte('\n')
		n++
		if o.Err != "" {
			nerr++
		} else if o.Plies > 0 {
			nsim++
			its += o.Iters
		}
	}
	mean := 0
	if nsim > 0 {
		mean = its / nsim
	}
	fmt.Fprintf(os.Stderr, "%d positions: %d simulated (mean %d iterations), %d solver stage, %d errors\n", n, nsim, mean, n-nsim-nerr, nerr)
}
