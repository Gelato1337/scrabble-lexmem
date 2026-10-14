// lexmem_selfplay: fast self-play for scrabble-lexmem. Macondo plays the rules (move generation, scoring, bag, end of game);
// our policy net, served on a GPU by server_infer.py over a Unix socket, chooses every move.
// Many games run concurrently (one goroutine each); their positions are batched into one request per round trip.
// Move tokens are built exactly as lexmem/policy.py MoveDataset.build: top 400 plays by score (one per lane+word: fewest
// blanks, then highest score) + every exchange when the bag has >= 7 tiles.
// Output: one JSON line per game (positions, chosen moves, final scores). PARITY=1 also stores each position's move list and logits.
// Branching (P_BRANCH > 0): counterfactual pairs with shared draws, see rollout(). BK alternatives, H mover turns (0 = to the end).
// Env: SOCK NGAMES CONC BATCH PTEMP SEED OUT LEXWORDS PARITY P_BRANCH BK H
package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

const MAXM = 400

func envI(k string, d int) int {
	if v, err := strconv.Atoi(os.Getenv(k)); err == nil { return v }
	return d
}
func envF(k string, d float64) float64 {
	if v, err := strconv.ParseFloat(os.Getenv(k), 64); err == nil { return v }
	return d
}

// ---- features of one position, as server_infer.py expects them
type Token struct {
	Widx  int32
	MF    [9]float32
	Leave [27]uint8
	Mv    *move.Move
	Desc  []int // [kind 0 play / 1 exchange, r, c, d, L, widx, score, blanks] or [1, keep27...]
}
type Pos struct {
	Board  [225]uint8
	Rack   [27]uint8
	Unseen [27]uint8
	Glob   [3]float32
	Tokens []Token
}
type Reply struct {
	Logits []float32
	Value  [2]float32
}
type req struct {
	p      *Pos
	out    chan Reply
	logits []float32
}

// ---- batching client: game goroutines send positions; one writer batches them; one reader returns results in order
type Client struct {
	conn    net.Conn
	in      chan req
	pending chan []req
	batch   int
}

func NewClient(sock string, batch int) *Client {
	c, err := net.Dial("unix", sock)
	if err != nil { panic(err) }
	cl := &Client{conn: c, in: make(chan req, 4096), pending: make(chan []req, 64), batch: batch}
	go cl.writer(); go cl.reader(); return cl
}
func (c *Client) Infer(p *Pos) Reply { ch := make(chan Reply, 1); c.in <- req{p: p, out: ch}; return <-ch }
func (c *Client) writer() {
	w := bufio.NewWriterSize(c.conn, 1<<22)
	for {
		reqs := []req{<-c.in}
		deadline := time.After(2 * time.Millisecond)
	collect:
		for len(reqs) < c.batch {
			select {
			case r := <-c.in: reqs = append(reqs, r)
			case <-deadline: break collect
			}
		}
		T := 0
		for _, r := range reqs { T += len(r.p.Tokens) }
		binary.Write(w, binary.LittleEndian, [2]uint32{uint32(len(reqs)), uint32(T)})
		for _, r := range reqs { w.Write(r.p.Board[:]) }
		for _, r := range reqs { w.Write(r.p.Rack[:]) }
		for _, r := range reqs { w.Write(r.p.Unseen[:]) }
		for _, r := range reqs { binary.Write(w, binary.LittleEndian, r.p.Glob) }
		for _, r := range reqs { binary.Write(w, binary.LittleEndian, uint16(len(r.p.Tokens))) }
		for _, r := range reqs { for _, t := range r.p.Tokens { binary.Write(w, binary.LittleEndian, t.Widx) } }
		for _, r := range reqs { for _, t := range r.p.Tokens { binary.Write(w, binary.LittleEndian, t.MF) } }
		for _, r := range reqs { for _, t := range r.p.Tokens { w.Write(t.Leave[:]) } }
		if err := w.Flush(); err != nil { panic(err) }
		c.pending <- reqs
	}
}
func (c *Client) reader() {
	rd := bufio.NewReaderSize(c.conn, 1<<22)
	for reqs := range c.pending {
		for i := range reqs {
			l := make([]float32, len(reqs[i].p.Tokens))
			if err := binary.Read(rd, binary.LittleEndian, l); err != nil { panic(err) }
			reqs[i].logits = l
		}
		vals := make([]float32, 2*len(reqs))
		if err := binary.Read(rd, binary.LittleEndian, vals); err != nil { panic(err) }
		for i := range reqs { reqs[i].out <- Reply{Logits: reqs[i].logits, Value: [2]float32{vals[2*i], vals[2*i+1]}} }
	}
}

// ---- tiles
func idx27(ml tilemapping.MachineLetter) int {
	if ml == 0 || ml.IsBlanked() && ml.Unblank() == 0 { return 26 }
	return int(ml.Unblank()) - 1
}
func counts(ws ...tilemapping.MachineWord) (c [27]uint8) {
	for _, w := range ws { for _, t := range w { c[idx27(t)]++ } }
	return
}

var wordIdx map[string]int32

func buildPos(g *game.Game, turn int) *Pos {
	p := &Pos{}; bd := g.Board(); me := g.PlayerOnTurn()
	for r := 0; r < 15; r++ { for c := 0; c < 15; c++ { if ml := bd.GetLetter(r, c); ml != 0 { p.Board[r*15+c] = uint8(ml.Unblank()) } } }
	p.Rack = counts(g.RackFor(me).TilesOn())
	p.Unseen = counts(tilemapping.MachineWord(g.Bag().Peek()), g.RackFor(1-me).TilesOn())
	p.Glob = [3]float32{float32(g.Bag().TilesRemaining()) / 100, float32(g.PointsFor(me)-g.PointsFor(1-me)) / 100, float32(turn) / 30}
	return p
}
func addTokens(p *Pos, g *game.Game, plays []*move.Move) {
	bd := g.Board()
	type cand struct { t Token; key [4]int; blanks, score int }
	best := map[[4]int]cand{}
	var exch []Token
	for _, m := range plays {
		switch m.Action() {
		case move.MoveTypePlay:
			r, c, vert := m.CoordsAndVertical(); d := 0; dr, dc := 0, 1
			if vert { d = 1; dr, dc = 1, 0 }
			tiles := m.Tiles(); var sb strings.Builder; blanks := 0
			for j, t := range tiles {
				var l tilemapping.MachineLetter
				if t == 0 { l = bd.GetLetter(r+dr*j, c+dc*j).Unblank() } else { l = t.Unblank(); if t.IsBlanked() { blanks++ } }
				sb.WriteByte(byte('A' + int(l) - 1))
			}
			w, ok := wordIdx[sb.String()]
			if !ok { continue }
			s := m.Score(); L := len(tiles)
			t := Token{Widx: w, Mv: m, Desc: []int{0, r, c, d, L, int(w), s, blanks}}
			t.MF = [9]float32{float32(s) / 50, float32(L) / 15, float32(r) / 14, float32(c) / 14, float32(d), b2f(s >= 50), float32(m.TilesPlayed()) / 7, float32(blanks), 0}
			t.Leave = counts(m.Leave())
			k := [4]int{r, c, d, int(w)}
			if o, seen := best[k]; !seen || blanks < o.blanks || (blanks == o.blanks && s > o.score) { best[k] = cand{t, k, blanks, s} }
		case move.MoveTypeExchange:
			keep := counts(m.Leave())
			t := Token{Widx: 0, Mv: m, Leave: keep}; t.MF[8] = 1
			t.Desc = append([]int{1}, func() []int { o := make([]int, 27); for i, v := range keep { o[i] = int(v) }; return o }()...)
			exch = append(exch, t)
		}
	}
	ps := make([]cand, 0, len(best))
	for _, v := range best { ps = append(ps, v) }
	if bd.IsEmpty() { // opening: add the mirrored vertical copy of every play (Python generates both orientations)
		n := len(ps)
		for i := 0; i < n; i++ {
			o := ps[i]; m := o.t.Mv; r, c, _ := m.CoordsAndVertical()
			tm := move.NewScoringMove(m.Score(), m.Tiles(), m.Leave(), true, m.TilesPlayed(), g.Alphabet(), c, r)
			t := o.t; t.Mv = tm; t.Desc = append([]int{}, o.t.Desc...); t.Desc[1], t.Desc[2], t.Desc[3] = c, r, 1
			t.MF[2], t.MF[3], t.MF[4] = float32(c)/14, float32(r)/14, 1
			ps = append(ps, cand{t, [4]int{c, r, 1, int(t.Widx)}, o.blanks, o.score})
		}
	}
	key := func(c cand) [5]int { return [5]int{-c.score, int(c.t.Widx), c.t.Desc[1], c.t.Desc[2], c.t.Desc[3]} }
	sort.Slice(ps, func(i, j int) bool { a, b := key(ps[i]), key(ps[j]); for k := 0; k < 5; k++ { if a[k] != b[k] { return a[k] < b[k] } }; return false })
	if len(ps) > MAXM { ps = ps[:MAXM] }
	for _, v := range ps { p.Tokens = append(p.Tokens, v.t) }
	if g.Bag().TilesRemaining() >= 7 { p.Tokens = append(p.Tokens, exch...) }
}
func b2f(b bool) float32 { if b { return 1 }; return 0 }

// ---- counterfactual branches: the same position continued after move a and after move b, with shared draws
// (paired bag: fixed tile order, each seat draws from its own end; same seed for exchange re-inserts in both copies).
// Both continue with the greedy policy for H turns of the mover (0 = to the end); a truncated branch is scored with the value head.
// U = win (1 / 0.5 / 0) + spread / 400 from the mover's view; the pair's result is dU = U(a) - U(b).
func outcomeU(spread float64) float64 {
	w := 0.5
	if spread > 0 { w = 1 } else if spread < 0 { w = 0 }
	return w + spread/400
}
func rollout(g0 *game.Game, first *move.Move, mover, turn, H int, seed [32]byte, mk func(*game.Game) *movegen.GordonGenerator, cl *Client) float64 {
	gc := g0.Copy(); gc.SetPairedBagMode(true); gc.Bag().SetFixedOrder(true); gc.SeedBag(seed)
	if err := gc.PlayMove(first, false, 0); err != nil { panic(err) }
	gen := mk(gc); own := 0; t := turn + 1
	for gc.Playing() != pb.PlayState_GAME_OVER && t < 100 {
		me := gc.PlayerOnTurn()
		if me == mover { own++ }
		p := buildPos(gc, t); gen.SetGame(gc)
		addTokens(p, gc, gen.GenAll(gc.RackFor(me), gc.Bag().TilesRemaining() >= 7))
		if len(p.Tokens) == 0 { gc.PlayMove(move.NewPassMove(gc.RackFor(me).TilesOn(), gc.Alphabet()), false, 0); t++; continue }
		rep := cl.Infer(p)
		if H > 0 && own >= H { // truncate: value of this position from the mover's view
			sp := float64(gc.PointsFor(mover) - gc.PointsFor(1-mover)); vs := float64(rep.Value[0]) * 100; pw := 1 / (1 + math.Exp(-float64(rep.Value[1])))
			if me == mover { return pw + (sp+vs)/400 }
			return (1 - pw) + (sp-vs)/400
		}
		k := 0
		for i := range rep.Logits { if rep.Logits[i] > rep.Logits[k] { k = i } }
		if err := gc.PlayMove(p.Tokens[k].Mv, false, 0); err != nil { panic(err) }
		t++
	}
	return outcomeU(float64(gc.PointsFor(mover) - gc.PointsFor(1-mover)))
}

// ---- one game
type PosRec struct {
	Board  string     `json:"b"`
	Rack   [27]uint8  `json:"rk"`
	Unseen [27]uint8  `json:"un"`
	Scores [2]int     `json:"sc"`
	Bag    int        `json:"bag"`
	Turn   int        `json:"t"`
	Mover  int        `json:"p"`
	Move   []int      `json:"mv"`
	Moves  [][]int    `json:"moves,omitempty"`
	Branch []float64  `json:"br,omitempty"` // [dU, then the alternative move's descriptor]
	Logits []float32  `json:"logits,omitempty"`
}
type GameRec struct {
	Seed   int64      `json:"seed"`
	Final  [2]int     `json:"final"`
	Pos    []PosRec   `json:"pos"`
}
func boardString(g *game.Game) string {
	var sb strings.Builder; bd := g.Board()
	for r := 0; r < 15; r++ { for c := 0; c < 15; c++ {
		ml := bd.GetLetter(r, c)
		switch { case ml == 0: sb.WriteByte('.'); case ml.IsBlanked(): sb.WriteByte(byte('a' + int(ml.Unblank()) - 1)); default: sb.WriteByte(byte('A' + int(ml) - 1)) }
	} }
	return sb.String()
}
func playGame(g *game.Game, gen *movegen.GordonGenerator, cl *Client, seed int64, ptemp float64, parity bool, pBranch float64, bk, H int, mk func(*game.Game) *movegen.GordonGenerator) GameRec {
	var s [32]byte; binary.LittleEndian.PutUint64(s[:], uint64(seed))
	g.SeedBag(s); g.StartGame(); rng := rand.New(rand.NewSource(seed))
	rec := GameRec{Seed: seed}; turn := 0
	for g.Playing() != pb.PlayState_GAME_OVER && turn < 100 {
		me := g.PlayerOnTurn(); p := buildPos(g, turn)
		gen.SetGame(g)
		addTokens(p, g, gen.GenAll(g.RackFor(me), g.Bag().TilesRemaining() >= 7))
		pr := PosRec{Board: boardString(g), Rack: p.Rack, Unseen: p.Unseen, Scores: [2]int{g.PointsFor(me), g.PointsFor(1 - me)}, Bag: g.Bag().TilesRemaining(), Turn: turn, Mover: me}
		var chosen *move.Move
		if len(p.Tokens) == 0 {
			chosen = move.NewPassMove(g.RackFor(me).TilesOn(), g.Alphabet()); pr.Move = []int{2}
		} else {
			rep := cl.Infer(p); k := 0
			if ptemp <= 0 {
				for i := range rep.Logits { if rep.Logits[i] > rep.Logits[k] { k = i } }
			} else {
				mx := rep.Logits[0]; for _, v := range rep.Logits { if v > mx { mx = v } }
				w := make([]float64, len(rep.Logits)); tot := 0.0
				for i, v := range rep.Logits { w[i] = math.Exp(float64(v-mx) / ptemp); tot += w[i] }
				x := rng.Float64() * tot
				for k = 0; k < len(w)-1 && x > w[k]; k++ { x -= w[k] }
			}
			chosen = p.Tokens[k].Mv; pr.Move = p.Tokens[k].Desc
			if len(p.Tokens) > 1 && rng.Float64() < pBranch { // alternative b: sampled from the policy's top bk moves other than k
				idx := make([]int, len(p.Tokens)); for i := range idx { idx[i] = i }
				sort.Slice(idx, func(i, j int) bool { return rep.Logits[idx[i]] > rep.Logits[idx[j]] })
				var alts []int
				for _, i := range idx { if i != k { alts = append(alts, i) }; if len(alts) == bk { break } }
				mx := rep.Logits[alts[0]]; ws := make([]float64, len(alts)); tot := 0.0
				for i, a := range alts { ws[i] = math.Exp(float64(rep.Logits[a]-mx) / math.Max(ptemp, 1e-3)); tot += ws[i] }
				x := rng.Float64() * tot; b := alts[len(alts)-1]
				for i, a := range alts { if x <= ws[i] { b = a; break }; x -= ws[i] }
				var bs [32]byte; binary.LittleEndian.PutUint64(bs[:], rng.Uint64()); binary.LittleEndian.PutUint64(bs[8:], rng.Uint64())
				dU := rollout(g, p.Tokens[k].Mv, me, turn, H, bs, mk, cl) - rollout(g, p.Tokens[b].Mv, me, turn, H, bs, mk, cl)
				pr.Branch = []float64{dU}
				for _, v := range p.Tokens[b].Desc { pr.Branch = append(pr.Branch, float64(v)) }
			}
			if parity { for _, t := range p.Tokens { pr.Moves = append(pr.Moves, t.Desc) }; pr.Logits = rep.Logits }
		}
		rec.Pos = append(rec.Pos, pr)
		if err := g.PlayMove(chosen, false, 0); err != nil { panic(err) }
		turn++
	}
	rec.Final = [2]int{g.PointsFor(0), g.PointsFor(1)}
	return rec
}

func main() {
	zerolog.SetGlobalLevel(zerolog.ErrorLevel)
	ex, _ := os.Executable(); cfg := &config.Config{}; cfg.Load(nil); cfg.AdjustRelativePaths(filepath.Dir(ex))
	words, err := os.ReadFile(os.Getenv("LEXWORDS"))
	if err != nil { panic(err) }
	wordIdx = map[string]int32{}
	for i, w := range strings.Split(strings.TrimSpace(string(words)), "\n") { wordIdx[w] = int32(i) }
	ngames, conc, seed0 := envI("NGAMES", 100), envI("CONC", 256), int64(envI("SEED", 1)); ptemp := envF("PTEMP", 1.0); parity := os.Getenv("PARITY") == "1"
	pBranch, bk, horizon := envF("P_BRANCH", 0), envI("BK", 6), envI("H", 3)
	cl := NewClient(os.Getenv("SOCK"), envI("BATCH", 512))
	out, err := os.Create(os.Getenv("OUT"))
	if err != nil { panic(err) }
	bw := bufio.NewWriterSize(out, 1<<20); enc := json.NewEncoder(bw); var mu sync.Mutex
	jobs := make(chan int64, ngames)
	for i := 0; i < ngames; i++ { jobs <- seed0 + int64(i) }
	close(jobs)
	gd, err := kwg.GetKWG(cfg.WGLConfig(), "NWL23")
	if err != nil { panic(err) }
	t0 := time.Now(); done := 0; npos := 0; var wg sync.WaitGroup
	for w := 0; w < conc; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rules, err := game.NewBasicGameRules(cfg, "NWL23", board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
			if err != nil { panic(err) }
			g, err := game.NewGame(rules, []*pb.PlayerInfo{{Nickname: "p1", RealName: "p1"}, {Nickname: "p2", RealName: "p2"}})
			if err != nil { panic(err) }
			gen := movegen.NewGordonGenerator(gd, g.Board(), rules.LetterDistribution())
			mk := func(gc *game.Game) *movegen.GordonGenerator { return movegen.NewGordonGenerator(gd, gc.Board(), rules.LetterDistribution()) }
			for seed := range jobs {
				r := playGame(g, gen, cl, seed, ptemp, parity, pBranch, bk, horizon, mk)
				mu.Lock(); enc.Encode(r); done++; npos += len(r.Pos)
				if done%500 == 0 { el := time.Since(t0).Seconds(); fmt.Fprintf(os.Stderr, "%d games %.0fs  %.1f games/s  %.0f positions/s\n", done, el, float64(done)/el, float64(npos)/el) }
				mu.Unlock()
			}
		}()
	}
	wg.Wait(); mu.Lock(); el := time.Since(t0).Seconds()
	fmt.Fprintf(os.Stderr, "DONE %d games, %d positions in %.0fs: %.1f games/s, %.0f positions/s\n", done, npos, el, float64(done)/el, float64(npos)/el)
	mu.Unlock()
	bw.Flush(); out.Close()
}
