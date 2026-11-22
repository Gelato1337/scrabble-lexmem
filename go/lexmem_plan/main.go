// lexmem_plan: exact-by-enumeration planning labels for the lexicon-memory benchmark.
// For each position (one CGP per input line, mover on turn) and each of its top K plays by score (one per lane + word: fewest
// blanks, then highest score), over S shared samples (same seed per sample index for every candidate):
//   E_opp = mean best score the opponent can make next turn (rack: min(7, unseen) tiles drawn from the tiles unseen to the mover),
//   E_my  = mean best score the mover can make on the same board with leave + refill from what is left (never more than the real
//           bag holds; ignoring the opponent's move),
// best = highest-scoring legal play over the whole lexicon (Macondo shadow top-play, score only; bingo bonus included).
// Planning value PV = score - E_opp + E_my. No policy, no value net, no leave table: only the rules and the dictionary.
// H2=1 adds a two-turn horizon on the same samples, every side playing its best-scoring move: the opponent's best reply is played,
//   E_my2  = mean best score of the mover (leave + refill) on the board after that reply,
//   E_opp2 = mean best score of the opponent's next play after the mover's best play (its leave + refill from the bag);
//   two-turn planning value PV2 = score - E_opp + E_my2 - E_opp2.
// H3=1 (implies H2) continues the same line one more exchange: the opponent's second reply is played and it draws, the mover
//   (leave of its second play + refill) answers again, the opponent (leave + refill) replies again:
//   E_my3 = mean best score of the mover's third play, E_opp3 = mean best score of the opponent's third play;
//   three-turn planning value PV3 = PV2 + E_my3 - E_opp3. CHECK=1 recomputes cross-sets and anchors from scratch for every
//   two-turn board and reports mismatches with the incremental update on stderr.
// stdin: {"id":..,"cgp":".."} per line; stdout: {"id":..,"cands":[{"d":[0,r,c,d,L,widx,score,blanks],"eo":..,"em":..,"so":..,"sm":..,
//   "h2":{"em2":..,"eo2":..,"sm2":..,"so2":..},"h3":{"em3":..,"eo3":..,"sm3":..,"so3":..}}]}
// Optional input "okeep" (one turn; the planning-with-inference test, eval/plan_infer.py): sample s gives the opponent the kept
//   tiles okeep[s % len] plus a refill from the rest of the pool instead of 7 tiles drawn from the whole pool.
// Env: K S LEXWORDS (our word order for widx) LEXICON (KWG name, default NWL23) WORKERS H2 H3 CHECK SEED0 (offset of the sample seeds)
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
	"sync/atomic"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog"

	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

func envI(k string, d int) int {
	if v, err := strconv.Atoi(os.Getenv(k)); err == nil { return v }
	return d
}

var wordIdx map[string]int32
var h2, h3, check bool
var nCheck, nShadow, nIncr, nB1, nB1Check, nKeepFail int64
var seed0 uint64

type In struct {
	ID    json.RawMessage `json:"id"`
	CGP   string          `json:"cgp"`
	OKeep []string        `json:"okeep,omitempty"`   // optional (one turn): sample s gives the opponent the kept tiles OKeep[s % len] + a refill
}
type Cand struct {
	D  []int   `json:"d"`
	EO float64 `json:"eo"`
	EM float64 `json:"em"`
	SO float64 `json:"so"`
	SM float64 `json:"sm"`
	H2 *Two    `json:"h2,omitempty"`
	H3 *Three  `json:"h3,omitempty"`
}
type Three struct {
	EM3 float64 `json:"em3"`
	EO3 float64 `json:"eo3"`
	SM3 float64 `json:"sm3"`
	SO3 float64 `json:"so3"`
}
type Two struct {
	EM2 float64 `json:"em2"`
	EO2 float64 `json:"eo2"`
	SM2 float64 `json:"sm2"`
	SO2 float64 `json:"so2"`
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

// bestMove: best-scoring play for rack, copied into dst (the top-play recorder reuses its move); ok = false when no play exists.
func bestMove(gen *movegen.GordonGenerator, rack *tilemapping.Rack, dst *move.Move) (int, bool) {
	ps := gen.GenAll(rack, false)
	if len(ps) == 0 || ps[0].Action() != move.MoveTypePlay { return 0, false }
	dst.CopyFrom(ps[0])
	return ps[0].Score(), true
}

// takeTiles: remove the tiles m placed from rack (a blank designated as a letter leaves the rack as a blank).
func takeTiles(rack *tilemapping.Rack, m *move.Move) {
	for _, t := range m.Tiles() { if t == 0 { continue }; if t.IsBlanked() { rack.Take(0) } else { rack.Take(t) } }
}

// playOn: put m on b and bring its cross-sets up to date (anchors are updated by PlayMove).
func playOn(b *board.GameBoard, m *move.Move, gd *kwg.KWG, ld *tilemapping.LetterDistribution) {
	b.PlayMove(m); cross_set.UpdateCrossSetsForMove(b, m, gd, ld)
}

// maxScore: best score for rack by full enumeration (no shadow pruning) on a copy of b; scratch = recompute all cross-sets and anchors.
func maxScore(b *board.GameBoard, rack *tilemapping.Rack, scratch bool, gd *kwg.KWG, ld *tilemapping.LetterDistribution) int {
	bc := b.Copy()
	if scratch { cross_set.GenAllCrossSets(bc, gd, ld); bc.UpdateAllAnchors() }
	gen := movegen.NewGordonGenerator(gd, bc, ld); gen.SetSortingParameter(movegen.SortByScore); best := 0
	for _, m := range gen.GenAll(rack, false) { if m.Action() == move.MoveTypePlay && m.Score() > best { best = m.Score() } }
	return best
}

// checkBest: the shadow top play on the incrementally updated board (got) vs full enumeration on it, and full enumeration on it vs on a
// from-scratch copy (incremental cross-set update).
func checkBest(b *board.GameBoard, rack *tilemapping.Rack, got int, gd *kwg.KWG, ld *tilemapping.LetterDistribution) {
	fi, fs := maxScore(b, rack, false, gd, ld), maxScore(b, rack, true, gd, ld)
	atomic.AddInt64(&nCheck, 1)
	if got != fi { atomic.AddInt64(&nShadow, 1) }
	if fi != fs { atomic.AddInt64(&nIncr, 1) }
}

func label(cfg *config.Config, gd *kwg.KWG, in In, K, S int) Out {
	out := Out{ID: in.ID}
	pg, err := cgp.ParseCGP(cfg, in.CGP)
	if err != nil { out.Err = err.Error(); return out }
	g := pg.Game; g.RecalculateBoard(); g.SetBackupMode(game.InteractiveGameplayMode); g.SetStateStackLength(1)
	me := g.PlayerOnTurn(); bd := g.Board(); ld := g.Rules().LetterDistribution()
	var keeps [][]tilemapping.MachineLetter
	if len(in.OKeep) > 0 {
		if h2 { out.Err = "okeep is one-turn only"; return out }
		for _, k := range in.OKeep {
			mls, err := tilemapping.ToMachineLetters(k, g.Alphabet())
			if err != nil { out.Err = "okeep: " + err.Error(); return out }
			keeps = append(keeps, mls)
		}
	}
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
	sort.Slice(cs, func(i, j int) bool {   // full key (score, word, square, direction): map order must not decide which plays make the top K
		for _, t := range [][2]int{{cs[j].score, cs[i].score}, {cs[i].d[5], cs[j].d[5]}, {cs[i].d[1], cs[j].d[1]}, {cs[i].d[2], cs[j].d[2]}, {cs[i].d[3], cs[j].d[3]}} {
			if t[0] != t[1] { return t[0] < t[1] }
		}
		return false
	})
	if len(cs) > K { cs = cs[:K] }
	for _, cd := range cs {
		gc := g.Copy(); gc.SetBackupMode(game.InteractiveGameplayMode)
		if err := gc.PlayMove(cd.m, false, 0); err != nil { continue }
		// best-score generators get no game: with one, an empty bag switches the shadow pass to endgame bounds (not-playing-out
		// penalties) while plays are ranked by raw score, so it can prune the highest-scoring play
		leave := cd.m.Leave(); tg := movegen.NewGordonGenerator(gd, gc.Board(), ld)
		tg.SetSortingParameter(movegen.SortByScore); tg.SetPlayRecorderTopPlay()
		// pool = every tile the mover cannot see after the move (the CGP bag holds the real bag + the opponent's rack)
		gc.ThrowRacksIn()
		if err := gc.Bag().RemoveTiles(leave); err != nil { continue }
		alph := g.Alphabet(); bag := gc.Bag(); bufO := make([]tilemapping.MachineLetter, 7); bufM := make([]tilemapping.MachineLetter, 7)
		rO, rM := tilemapping.NewRack(alph), tilemapping.NewRack(alph)
		so, so2, sm, sm2 := 0.0, 0.0, 0.0, 0.0
		// two-turn scratch: boards after the opponent's reply (b2) and after the mover's next play (b3), one generator each
		var b2, b3 *board.GameBoard; var g2, g3 *movegen.GordonGenerator; var m1, m2 *move.Move; var rO2 *tilemapping.Rack
		bufR := make([]tilemapping.MachineLetter, 7); tm2, tm22, to2, to22 := 0.0, 0.0, 0.0, 0.0
		newGen := func(b *board.GameBoard) *movegen.GordonGenerator {
			x := movegen.NewGordonGenerator(gd, b, ld); x.SetSortingParameter(movegen.SortByScore); x.SetPlayRecorderTopPlay(); return x
		}
		if h2 {
			b2, b3 = gc.Board().Copy(), gc.Board().Copy(); m1, m2 = new(move.Move), new(move.Move); rO2 = tilemapping.NewRack(alph)
			g2, g3 = newGen(b2), newGen(b3)
		}
		// three-turn scratch: boards after the opponent's second reply (b4) and after the mover's third play (b5)
		var b4, b5 *board.GameBoard; var g4, g5 *movegen.GordonGenerator; var m3, m4 *move.Move; var rM3, rO3 *tilemapping.Rack
		bufM3, bufO3 := make([]tilemapping.MachineLetter, 7), make([]tilemapping.MachineLetter, 7); tm3, tm32, to3, to32 := 0.0, 0.0, 0.0, 0.0
		if h3 {
			b4, b5 = gc.Board().Copy(), gc.Board().Copy(); m3, m4 = new(move.Move), new(move.Move); rM3, rO3 = tilemapping.NewRack(alph), tilemapping.NewRack(alph)
			g4, g5 = newGen(b4), newGen(b5)
		}
		for s := 0; s < S; s++ {
			// canonical tile order before the seeded shuffle: reproducible, and candidates share their draws as far as their pools allow
			var sd [32]byte; binary.LittleEndian.PutUint64(sd[:], uint64(s)+1+seed0); gc.SeedBag(sd)
			bt := bag.Tiles(); sort.Slice(bt, func(i, j int) bool { return bt[i] < bt[j] }); bag.Shuffle()
			pool := bag.TilesRemaining(); ko := pool; if ko > 7 { ko = 7 }
			// given kept tiles: the opponent holds them + a refill from the rest (its draw after its play), the mover draws after
			var kp []tilemapping.MachineLetter
			if len(keeps) > 0 {
				kp = keeps[s%len(keeps)]
				if err := bag.RemoveTiles(kp); err != nil { atomic.AddInt64(&nKeepFail, 1); continue }
				pool = bag.TilesRemaining(); ko = 7 - len(kp); if ko > pool { ko = pool }
			}
			km := 7 - len(leave); if km > pool-ko { km = pool - ko }; if km < 0 { km = 0 }
			if err := bag.Draw(ko, bufO); err != nil { if kp != nil { bag.PutBack(kp) }; continue }
			if err := bag.Draw(km, bufM); err != nil { bag.PutBack(bufO[:ko]); if kp != nil { bag.PutBack(kp) }; continue }
			rO.Set(append(append([]tilemapping.MachineLetter{}, kp...), bufO[:ko]...)); rM.Set(append(append([]tilemapping.MachineLetter{}, leave...), bufM[:km]...))
			var o float64; ok1 := false
			if h2 { var oi int; oi, ok1 = bestMove(tg, rO, m1); o = float64(oi) } else { o = float64(bestScore(tg, rO)) }
			mm := float64(bestScore(tg, rM))
			if check {   // one-turn labels on the board after the candidate: shadow top play vs full enumeration
				atomic.AddInt64(&nB1Check, 2)
				if int(o) != maxScore(gc.Board(), rO, false, gd, ld) { atomic.AddInt64(&nB1, 1) }
				if int(mm) != maxScore(gc.Board(), rM, false, gd, ld) { atomic.AddInt64(&nB1, 1) }
			}
			if h2 {
				// opponent plays its best reply (or passes), then draws; the mover answers with leave + refill; the opponent replies again
				b2.CopyFrom(gc.Board()); t1 := 0
				if ok1 { playOn(b2, m1, gd, ld); t1 = m1.TilesPlayed() }
				my2, ok2 := bestMove(g2, rM, m2)
				if check { checkBest(b2, rM, my2, gd, ld) }
				b3.CopyFrom(b2)
				if ok2 { playOn(b3, m2, gd, ld) }
				rO2.CopyFrom(rO)
				if ok1 { takeTiles(rO2, m1) }
				n3 := t1; if r := bag.TilesRemaining(); n3 > r { n3 = r }
				if n3 > 0 { if err := bag.Draw(n3, bufR); err != nil { n3 = 0 } else { for _, t := range bufR[:n3] { rO2.Add(t) } } }
				var o2 int; ok3 := false
				if h3 { o2, ok3 = bestMove(g3, rO2, m3) } else { o2 = bestScore(g3, rO2) }
				if check { checkBest(b3, rO2, o2, gd, ld) }
				if h3 {
					// the opponent plays its second reply; the mover refills after its second play (drawing after the opponent's
					// refill, as in the game) and answers; the opponent refills and replies again
					b4.CopyFrom(b3); t3 := 0
					if ok3 { playOn(b4, m3, gd, ld); t3 = m3.TilesPlayed() }
					rM3.CopyFrom(rM); t2 := 0
					if ok2 { takeTiles(rM3, m2); t2 = m2.TilesPlayed() }
					n4 := t2; if r := bag.TilesRemaining(); n4 > r { n4 = r }
					if n4 > 0 { if err := bag.Draw(n4, bufM3); err != nil { n4 = 0 } else { for _, t := range bufM3[:n4] { rM3.Add(t) } } }
					my3, ok4 := bestMove(g4, rM3, m4)
					if check { checkBest(b4, rM3, my3, gd, ld) }
					b5.CopyFrom(b4)
					if ok4 { playOn(b5, m4, gd, ld) }
					rO3.CopyFrom(rO2)
					if ok3 { takeTiles(rO3, m3) }
					n5 := t3; if r := bag.TilesRemaining(); n5 > r { n5 = r }
					if n5 > 0 { if err := bag.Draw(n5, bufO3); err != nil { n5 = 0 } else { for _, t := range bufO3[:n5] { rO3.Add(t) } } }
					o3 := bestScore(g5, rO3)
					if check { checkBest(b5, rO3, o3, gd, ld) }
					if n5 > 0 { bag.PutBack(bufO3[:n5]) }
					if n4 > 0 { bag.PutBack(bufM3[:n4]) }
					tm3 += float64(my3); tm32 += float64(my3 * my3); to3 += float64(o3); to32 += float64(o3 * o3)
				}
				if n3 > 0 { bag.PutBack(bufR[:n3]) }
				tm2 += float64(my2); tm22 += float64(my2 * my2); to2 += float64(o2); to22 += float64(o2 * o2)
			}
			bag.PutBack(bufO[:ko]); bag.PutBack(bufM[:km]); if kp != nil { bag.PutBack(kp) }
			so += o; so2 += o * o; sm += mm; sm2 += mm * mm
		}
		n := float64(S); eo, em := so/n, sm/n
		cand := Cand{D: cd.d, EO: eo, EM: em, SO: math.Sqrt(math.Max(so2/n-eo*eo, 0)), SM: math.Sqrt(math.Max(sm2/n-em*em, 0))}
		if h2 {
			a, b := tm2/n, to2/n
			cand.H2 = &Two{EM2: a, EO2: b, SM2: math.Sqrt(math.Max(tm22/n-a*a, 0)), SO2: math.Sqrt(math.Max(to22/n-b*b, 0))}
		}
		if h3 {
			a, b := tm3/n, to3/n
			cand.H3 = &Three{EM3: a, EO3: b, SM3: math.Sqrt(math.Max(tm32/n-a*a, 0)), SO3: math.Sqrt(math.Max(to32/n-b*b, 0))}
		}
		out.Cands = append(out.Cands, cand)
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
	lexName := os.Getenv("LEXICON")   // KWG name; must match the "lex" in the input CGPs (default NWL23)
	if lexName == "" { lexName = "NWL23" }
	gd, err := kwg.GetKWG(cfg.WGLConfig(), lexName)
	if err != nil { panic(err) }
	K, S, W := envI("K", 16), envI("S", 256), envI("WORKERS", 8)
	h3 = os.Getenv("H3") == "1"; h2, check = os.Getenv("H2") == "1" || h3, os.Getenv("CHECK") == "1"; seed0 = uint64(envI("SEED0", 0))
	in := make(chan In, 1024); outc := make(chan Out, 1024); var wg sync.WaitGroup
	for w := 0; w < W; w++ { wg.Add(1); go func() { defer wg.Done(); for x := range in { outc <- label(cfg, gd, x, K, S) } }() }
	go func() { wg.Wait(); close(outc) }()
	go func() {
		sc := bufio.NewScanner(os.Stdin); sc.Buffer(make([]byte, 1<<20), 1<<20)
		for sc.Scan() { var x In; if json.Unmarshal(sc.Bytes(), &x) == nil { in <- x } }
		close(in)
	}()
	bw := bufio.NewWriterSize(os.Stdout, 1<<20); enc := json.NewEncoder(bw)
	for o := range outc { enc.Encode(o); bw.Flush() }   // flush per position: live callers (play_eval PLAN=exact) wait for each answer
	bw.Flush()
	if nKeepFail > 0 { os.Stderr.WriteString("okeep: kept tiles not in the pool in " + strconv.FormatInt(nKeepFail, 10) + " samples\n") }
	if check {
		f := func(x int64) string { return strconv.FormatInt(x, 10) }
		os.Stderr.WriteString("CHECK one-turn (shadow vs full): " + f(nB1) + " of " + f(nB1Check) + "; two/three-turn: shadow vs full " + f(nShadow) + ", incremental vs scratch " + f(nIncr) + " of " + f(nCheck) + "\n")
	}
}
