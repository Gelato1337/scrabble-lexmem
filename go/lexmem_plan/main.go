// lexmem_plan: exact-by-enumeration planning labels for the lexicon-memory benchmark.
// For each position (one CGP per input line, mover on turn) and each of its top K plays by score (one per lane + word: fewest
// blanks, then highest score), over S shared samples (same seed per sample index for every candidate):
//   E_opp = mean best score the opponent can make next turn (rack: min(7, unseen) tiles drawn from the tiles unseen to the mover),
//   E_my  = mean best score the mover can make on the same board with leave + refill from what is left (never more than the real
//           bag holds; ignoring the opponent's move),
// best = highest-scoring legal play over the whole lexicon (Macondo shadow top-play, score only; bingo bonus included).
// Planning value PV = score - E_opp + E_my. No policy, no value net, no leave table: only the rules and the dictionary.
// stdin: {"id":..,"cgp":".."} per line; stdout: {"id":..,"cands":[{"d":[0,r,c,d,L,widx,score,blanks],"eo":..,"em":..,"so":..,"sm":..}]}
// Env: K S LEXWORDS WORKERS
package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog"

	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

func envI(k string, d int) int {
	if v, err := strconv.Atoi(os.Getenv(k)); err == nil { return v }
	return d
}

var wordIdx map[string]int32

type In struct {
	ID  json.RawMessage `json:"id"`
	CGP string          `json:"cgp"`
}
type Cand struct {
	D  []int   `json:"d"`
	EO float64 `json:"eo"`
	EM float64 `json:"em"`
	SO float64 `json:"so"`
	SM float64 `json:"sm"`
}
type Out struct {
	ID    json.RawMessage `json:"id"`
	Cands []Cand          `json:"cands"`
	Err   string          `json:"err,omitempty"`
}

func bestScore(gen *movegen.GordonGenerator, rack *tilemapping.Rack) int {
	ps := gen.GenAll(rack, false)
	if len(ps) == 0 || ps[0].Action() != move.MoveTypePlay { return 0 }
	return ps[0].Score()
}

func label(cfg *config.Config, gd *kwg.KWG, in In, K, S int) Out {
	out := Out{ID: in.ID}
	pg, err := cgp.ParseCGP(cfg, in.CGP)
	if err != nil { out.Err = err.Error(); return out }
	g := pg.Game; g.RecalculateBoard(); g.SetBackupMode(game.InteractiveGameplayMode); g.SetStateStackLength(1)
	me := g.PlayerOnTurn(); bd := g.Board(); ld := g.Rules().LetterDistribution()
	gen := movegen.NewGordonGenerator(gd, bd, ld); gen.SetGame(g)
	plays := gen.GenAll(g.RackFor(me), false)
	type c struct { m *move.Move; d []int; blanks, score int }
	best := map[[4]int]c{}
	for _, m := range plays {
		if m.Action() != move.MoveTypePlay { continue }
		r, cc, vert := m.CoordsAndVertical(); d := 0; dr, dc := 0, 1
		if vert { d = 1; dr, dc = 1, 0 }
		var sb strings.Builder; bl := 0
		for j, t := range m.Tiles() {
			var l tilemapping.MachineLetter
			if t == 0 { l = bd.GetLetter(r+dr*j, cc+dc*j).Unblank() } else { l = t.Unblank(); if t.IsBlanked() { bl++ } }
			sb.WriteByte(byte('A' + int(l) - 1))
		}
		w, ok := wordIdx[sb.String()]
		if !ok { continue }
		k := [4]int{r, cc, d, int(w)}; s := m.Score()
		if o, seen := best[k]; !seen || bl < o.blanks || (bl == o.blanks && s > o.score) {
			best[k] = c{m, []int{0, r, cc, d, len(m.Tiles()), int(w), s, bl}, bl, s}
		}
	}
	cs := make([]c, 0, len(best))
	for _, v := range best { cs = append(cs, v) }
	sort.Slice(cs, func(i, j int) bool { if cs[i].score != cs[j].score { return cs[i].score > cs[j].score }; return cs[i].d[5] < cs[j].d[5] })
	if len(cs) > K { cs = cs[:K] }
	for _, cd := range cs {
		gc := g.Copy(); gc.SetBackupMode(game.InteractiveGameplayMode)
		if err := gc.PlayMove(cd.m, false, 0); err != nil { continue }
		leave := cd.m.Leave(); tg := movegen.NewGordonGenerator(gd, gc.Board(), ld); tg.SetGame(gc)
		tg.SetSortingParameter(movegen.SortByScore); tg.SetPlayRecorderTopPlay()
		// pool = every tile the mover cannot see after the move (the CGP bag holds the real bag + the opponent's rack)
		gc.ThrowRacksIn()
		if err := gc.Bag().RemoveTiles(leave); err != nil { continue }
		alph := g.Alphabet(); bag := gc.Bag(); bufO := make([]tilemapping.MachineLetter, 7); bufM := make([]tilemapping.MachineLetter, 7)
		rO, rM := tilemapping.NewRack(alph), tilemapping.NewRack(alph)
		so, so2, sm, sm2 := 0.0, 0.0, 0.0, 0.0
		for s := 0; s < S; s++ {
			var sd [32]byte; binary.LittleEndian.PutUint64(sd[:], uint64(s)+1); gc.SeedBag(sd); bag.Shuffle()
			pool := bag.TilesRemaining(); ko := pool; if ko > 7 { ko = 7 }
			km := 7 - len(leave); if km > pool-ko { km = pool - ko }; if km < 0 { km = 0 }
			if err := bag.Draw(ko, bufO); err != nil { continue }
			if err := bag.Draw(km, bufM); err != nil { bag.PutBack(bufO[:ko]); continue }
			rO.Set(bufO[:ko]); rM.Set(append(append([]tilemapping.MachineLetter{}, leave...), bufM[:km]...))
			o := float64(bestScore(tg, rO)); mm := float64(bestScore(tg, rM))
			bag.PutBack(bufO[:ko]); bag.PutBack(bufM[:km])
			so += o; so2 += o * o; sm += mm; sm2 += mm * mm
		}
		n := float64(S); eo, em := so/n, sm/n
		out.Cands = append(out.Cands, Cand{D: cd.d, EO: eo, EM: em, SO: math.Sqrt(math.Max(so2/n-eo*eo, 0)), SM: math.Sqrt(math.Max(sm2/n-em*em, 0))})
	}
	return out
}

func main() {
	zerolog.SetGlobalLevel(zerolog.ErrorLevel)
	ex, _ := os.Executable(); cfg := &config.Config{}; cfg.Load(nil); cfg.AdjustRelativePaths(filepath.Dir(ex))
	words, err := os.ReadFile(os.Getenv("LEXWORDS"))
	if err != nil { panic(err) }
	wordIdx = map[string]int32{}
	for i, w := range strings.Split(strings.TrimSpace(string(words)), "\n") { wordIdx[w] = int32(i) }
	gd, err := kwg.GetKWG(cfg.WGLConfig(), "NWL23")
	if err != nil { panic(err) }
	K, S, W := envI("K", 16), envI("S", 256), envI("WORKERS", 8)
	in := make(chan In, 1024); outc := make(chan Out, 1024); var wg sync.WaitGroup
	for w := 0; w < W; w++ { wg.Add(1); go func() { defer wg.Done(); for x := range in { outc <- label(cfg, gd, x, K, S) } }() }
	go func() { wg.Wait(); close(outc) }()
	go func() {
		sc := bufio.NewScanner(os.Stdin); sc.Buffer(make([]byte, 1<<20), 1<<20)
		for sc.Scan() { var x In; if json.Unmarshal(sc.Bytes(), &x) == nil { in <- x } }
		close(in)
	}()
	bw := bufio.NewWriterSize(os.Stdout, 1<<20); enc := json.NewEncoder(bw)
	for o := range outc { enc.Encode(o) }
	bw.Flush()
}
