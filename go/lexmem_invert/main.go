// lexmem_invert: exact rack inference against a static HastyBot, the information ceiling of the rack-inference benchmark
// (train/train_lexinfer.py, eval/infer_macondo.py). Input lines as go/lexmem_infer:
//   {"i":..,"cgp":"<position before the opponent's play: opponent on turn, its rack = the tiles it placed / our rack>",
//    "coords":"8D","tiles":"WO.D","played":"WOD","leave":"AE?RS"}
// For every possible kept set l (every multiset of the true kept count n from the tiles unseen to us minus the placed tiles P),
// HastyBot (Macondo HASTY_BOT in-process, our rack thrown into the bag as in go/lexmem_search) gets the rack P + l on the
// position before its play; l is consistent if HastyBot's move is the recorded play. Posterior over l ∝ prior(l) · consistent(l),
// prior(l) ∝ Π_x C(U_x, P_x + l_x) (the exact no-inference baseline, U = the tiles unseen to us before the play).
// Output per line (fields as go/lexmem_infer, so eval/infer_macondo.py analyzes either): {"i","n": kept sets checked,
// "complete","marg":[27][4] posterior P(kept count 0 / 1 / 2 / 3+) per tile in our order (A..Z, blank), "post": posterior of
// the true kept set, "prior": its prior, "lift": log2(post/prior) (-999 if 0), "nmatch": consistent kept sets,
// "truematch": the true kept set reproduces the play, "secs","err"}. No consistent kept set: marg = the prior, err set.
// Env: WORKERS (positions in parallel, default 8) MAXLEAVES (positions with more kept sets are skipped, default 400000)
// SAMPLES (> 0: also "samples", that many kept sets drawn from the posterior, seeded by the position id; e.g. "AE?")
package main

import (
	"bufio"
	"encoding/json"
	"math"
	"math/rand/v2"
	"sort"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/rs/zerolog"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
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
	I         int            `json:"i"`
	N         int            `json:"n"`
	Complete  bool           `json:"complete"`
	Marg      [27][4]float64 `json:"marg"`
	Post      float64        `json:"post"`
	Prior     float64        `json:"prior"`
	Lift      float64        `json:"lift"`
	NMatch    int            `json:"nmatch"`
	TrueMatch bool           `json:"truematch"`
	Secs      float64        `json:"secs"`
	Err       string         `json:"err,omitempty"`
	Samples   []string       `json:"samples,omitempty"`
}

func envI(k string, d int) int {
	if v, err := strconv.Atoi(os.Getenv(k)); err == nil {
		return v
	}
	return d
}

// ours: Macondo machine letter (0 = blank, 1..26 = A..Z) -> our tile index (0..25 = A..Z, 26 = blank)
func ours(ml tilemapping.MachineLetter) int {
	if ml == 0 {
		return 26
	}
	return int(ml) - 1
}
func mlOf(t int) tilemapping.MachineLetter {
	if t == 26 {
		return 0
	}
	return tilemapping.MachineLetter(t + 1)
}

func binom(n, k int) float64 {
	if k < 0 || k > n {
		return 0
	}
	r := 1.0
	for j := 1; j <= k; j++ {
		r = r * float64(n-k+j) / float64(j)
	}
	return r
}

// number of multisets of size n with count_t <= pool_t
func countLeaves(pool [27]int, n int) int {
	c := make([]int, n+1)
	c[0] = 1
	for t := 0; t < 27; t++ {
		nc := make([]int, n+1)
		for k := 0; k <= n; k++ {
			if c[k] == 0 {
				continue
			}
			for j := 0; j <= pool[t] && k+j <= n; j++ {
				nc[k+j] += c[k]
			}
		}
		c = nc
	}
	return c[n]
}

func invert(cfg *config.Config, conf *bot.BotConfig, x In, maxLeaves, nSamples int) (out Out) {
	out = Out{I: x.I, Lift: -999}
	t0 := time.Now()
	defer func() { out.Secs = time.Since(t0).Seconds() }()
	pg, err := cgp.ParseCGP(cfg, x.CGP)
	if err != nil {
		out.Err = err.Error()
		return out
	}
	g := pg.Game
	g.RecalculateBoard()
	g.SetBackupMode(game.NoBackup)
	hasty := g.PlayerOnTurn()
	played, err := g.CreateAndScorePlacementMove(x.Coords, x.Tiles, x.Played, false)
	if err != nil {
		out.Err = err.Error()
		return out
	}
	want := strings.TrimSpace(played.ShortDescription())
	// pool = the tiles unseen to us minus P = the CGP's bag (both racks are set in the CGP)
	var pool, P, trueL [27]int
	for _, ml := range g.Bag().Peek() {
		pool[ours(ml)]++
	}
	for _, ml := range g.RackFor(hasty).TilesOn() {
		P[ours(ml)]++
	}
	lv, err := tilemapping.ToMachineLetters(x.Leave, g.Alphabet())
	if err != nil {
		out.Err = err.Error()
		return out
	}
	for _, ml := range lv {
		trueL[ours(ml)]++
	}
	n := len(lv)
	out.N = countLeaves(pool, n)
	if out.N > maxLeaves {
		out.Err = "too many kept sets"
		return out
	}
	g.ThrowRacksInFor(1 - hasty) // HastyBot does not see our rack (as go/lexmem_search's botMove)
	tp, err := bot.NewBotTurnPlayerFromGame(g, conf, pb.BotRequest_HASTY_BOT)
	if err != nil {
		out.Err = err.Error()
		return out
	}
	var prior, post [27][4]float64
	zAll, zMatch, wTrue := 0.0, 0.0, 0.0
	var l [27]int
	mls := make([]tilemapping.MachineLetter, 0, 7)
	var keptL [][27]int
	var keptW []float64
	var rec func(t, left int, w float64)
	rec = func(t, left int, w float64) {
		if t == 27 {
			if left != 0 {
				return
			}
			mls = mls[:0]
			for u := 0; u < 27; u++ {
				for j := 0; j < P[u]+l[u]; j++ {
					mls = append(mls, mlOf(u))
				}
			}
			g.ThrowRacksInFor(hasty)
			rk := tilemapping.NewRack(g.Alphabet())
			rk.Set(mls)
			if err := g.SetRackForOnly(hasty, rk); err != nil {
				out.Err = "set rack: " + err.Error()
				return
			}
			match := strings.TrimSpace(tp.GenerateMoves(1)[0].ShortDescription()) == want
			isTrue := l == trueL
			zAll += w
			for u := 0; u < 27; u++ {
				k := min(l[u], 3)
				prior[u][k] += w
				if match {
					post[u][k] += w
				}
			}
			if match {
				zMatch += w
				out.NMatch++
				if nSamples > 0 {
					keptL = append(keptL, l)
					keptW = append(keptW, zMatch)
				}
			}
			if isTrue {
				wTrue = w
				out.TrueMatch = match
			}
			return
		}
		for j := 0; j <= pool[t] && j <= left; j++ {
			l[t] = j
			// the prior factor of tile t: C(U_t, P_t + j), U_t = pool_t + P_t
			rec(t+1, left-j, w*binom(pool[t]+P[t], P[t]+j))
		}
		l[t] = 0
	}
	rec(0, n, 1.0)
	out.Complete = out.Err == ""
	if zAll > 0 {
		out.Prior = wTrue / zAll
	}
	if zMatch == 0 {
		if out.Err == "" {
			out.Err = "no consistent kept set"
		}
		for u := 0; u < 27; u++ {
			for k := 0; k < 4; k++ {
				out.Marg[u][k] = prior[u][k] / zAll
			}
		}
		return out
	}
	for u := 0; u < 27; u++ {
		for k := 0; k < 4; k++ {
			out.Marg[u][k] = post[u][k] / zMatch
		}
	}
	if out.TrueMatch {
		out.Post = wTrue / zMatch
		out.Lift = math.Log2(out.Post / out.Prior)
	}
	if nSamples > 0 { // keptW holds the cumulative posterior weight
		rng := rand.New(rand.NewPCG(uint64(x.I), 7))
		for k := 0; k < nSamples; k++ {
			j := sort.SearchFloat64s(keptW, rng.Float64()*zMatch)
			if j >= len(keptL) {
				j = len(keptL) - 1
			}
			var sb strings.Builder
			for u := 0; u < 27; u++ {
				for c := 0; c < keptL[j][u]; c++ {
					if u == 26 {
						sb.WriteByte('?')
					} else {
						sb.WriteByte(byte('A' + u))
					}
				}
			}
			out.Samples = append(out.Samples, sb.String())
		}
	}
	return out
}

func main() {
	zerolog.SetGlobalLevel(zerolog.ErrorLevel)
	ex, _ := os.Executable()
	cfg := &config.Config{}
	cfg.Load(nil)
	cfg.AdjustRelativePaths(filepath.Dir(ex))
	W, maxLeaves, nSamples := envI("WORKERS", 8), envI("MAXLEAVES", 400000), envI("SAMPLES", 0)
	in := make(chan In, 256)
	outc := make(chan Out, 256)
	var wg sync.WaitGroup
	for w := 0; w < W; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			conf := &bot.BotConfig{Config: *cfg}
			for x := range in {
				outc <- invert(cfg, conf, x, maxLeaves, nSamples)
			}
		}()
	}
	go func() { wg.Wait(); close(outc) }()
	go func() {
		sc := bufio.NewScanner(os.Stdin)
		sc.Buffer(make([]byte, 1<<20), 1<<20)
		for sc.Scan() {
			var x In
			if json.Unmarshal(sc.Bytes(), &x) == nil {
				in <- x
			}
		}
		close(in)
	}()
	bw := bufio.NewWriterSize(os.Stdout, 1<<20)
	enc := json.NewEncoder(bw)
	for o := range outc {
		enc.Encode(o)
		bw.Flush()
	}
	bw.Flush()
}
