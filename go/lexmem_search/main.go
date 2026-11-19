// lexmem_search: whole games in Go between our side and a Macondo bot (HastyBot by default), for paired evaluations of the
// policy net, of a search over the net, or of a Macondo bot; optionally Macondo's SIMMING_BOT plays the end of the game for
// our side. The net is served on a GPU by selfplay/server_infer.py over a Unix socket (VHEAD=val for LEAF=net).
//
// Search (OURS=search): candidates = the policy's top K moves; S worlds shared by all candidates (bag order and the
// opponent's rack redrawn from the unseen tiles, as the mover would know them); in each world the candidate is played, then
// D plies of net argmax for both sides; the leaf gives my win probability from Macondo's win % table (indexed by the spread
// from the player on turn and the tiles unseen by that player), with
//   LEAF=mac: spread + leave value of the last play (Macondo's own sim leaf), or
//   LEAF=net: spread + the value head's predicted spread change over the next 4 half-moves of the player on turn;
// a finished game counts 1 / 0.5 / 0. The candidate with the highest mean is played, but it replaces the policy's top move
// only if its mean paired (same-world) difference exceeds Z standard errors (Z=0: plain argmax). PM caps the playout move list
// (top PM plays by score; the root always uses MAXM). REC=1 stores each search decision (position, candidates, estimates).
//
// Macondo bots: HASTY_BOT runs in-process (static, no solver); other codes (SIMMING_BOT, ...) run in a pool of NBOT
// bin/lexmem_bot processes (each its own GOMEMLIMIT, like the Python harness), fed a CGP with only the mover's rack.
//
// Env: SOCK LEXWORDS MACONDO NGAMES SEED CONC BATCH OUT EXCH OURS OURS_SECS OPP OPP_SECS HANDOFF EG_SECS NBOT
//      K S D LEAF PM Z REC PAR (max playouts in flight, default 4096) POSITIONS NPOS (positions mode, see runPositions)
// Game g (0-based): seed SEED+g; our side moves first when g is even. Output: one JSON line per game.
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
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/domino14/word-golib/cache"
	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

const MAXM = 400

func envI(k string, d int) int {
	if v, err := strconv.Atoi(os.Getenv(k)); err == nil {
		return v
	}
	return d
}
func envF(k string, d float64) float64 {
	if v, err := strconv.ParseFloat(os.Getenv(k), 64); err == nil {
		return v
	}
	return d
}
func envS(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

// ---- features of one position, as server_infer.py expects them (same as cmd/lexmem_selfplay)
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

type Client struct {
	conn    net.Conn
	in      chan req
	pending chan []req
	batch   int
}

func NewClient(sock string, batch int) *Client {
	c, err := net.Dial("unix", sock)
	if err != nil {
		panic(err)
	}
	cl := &Client{conn: c, in: make(chan req, 1<<16), pending: make(chan []req, 64), batch: batch}
	go cl.writer()
	go cl.reader()
	return cl
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
			case r := <-c.in:
				reqs = append(reqs, r)
			case <-deadline:
				break collect
			}
		}
		T := 0
		for _, r := range reqs {
			T += len(r.p.Tokens)
		}
		binary.Write(w, binary.LittleEndian, [2]uint32{uint32(len(reqs)), uint32(T)})
		for _, r := range reqs {
			w.Write(r.p.Board[:])
		}
		for _, r := range reqs {
			w.Write(r.p.Rack[:])
		}
		for _, r := range reqs {
			w.Write(r.p.Unseen[:])
		}
		for _, r := range reqs {
			binary.Write(w, binary.LittleEndian, r.p.Glob)
		}
		for _, r := range reqs {
			binary.Write(w, binary.LittleEndian, uint16(len(r.p.Tokens)))
		}
		for _, r := range reqs {
			for _, t := range r.p.Tokens {
				binary.Write(w, binary.LittleEndian, t.Widx)
			}
		}
		for _, r := range reqs {
			for _, t := range r.p.Tokens {
				binary.Write(w, binary.LittleEndian, t.MF)
			}
		}
		for _, r := range reqs {
			for _, t := range r.p.Tokens {
				w.Write(t.Leave[:])
			}
		}
		if err := w.Flush(); err != nil {
			panic(err)
		}
		c.pending <- reqs
	}
}
func (c *Client) reader() {
	rd := bufio.NewReaderSize(c.conn, 1<<22)
	for reqs := range c.pending {
		for i := range reqs {
			l := make([]float32, len(reqs[i].p.Tokens))
			if err := binary.Read(rd, binary.LittleEndian, l); err != nil {
				panic(err)
			}
			reqs[i].logits = l
		}
		vals := make([]float32, 2*len(reqs))
		if err := binary.Read(rd, binary.LittleEndian, vals); err != nil {
			panic(err)
		}
		for i := range reqs {
			reqs[i].out <- Reply{Logits: reqs[i].logits, Value: [2]float32{vals[2*i], vals[2*i+1]}}
		}
	}
}

func idx27(ml tilemapping.MachineLetter) int {
	if ml == 0 || ml.IsBlanked() && ml.Unblank() == 0 {
		return 26
	}
	return int(ml.Unblank()) - 1
}
func counts(ws ...tilemapping.MachineWord) (c [27]uint8) {
	for _, w := range ws {
		for _, t := range w {
			c[idx27(t)]++
		}
	}
	return
}

var wordIdx map[string]int32

func buildPos(g *game.Game, turn int) *Pos {
	p := &Pos{}
	bd := g.Board()
	me := g.PlayerOnTurn()
	for r := 0; r < 15; r++ {
		for c := 0; c < 15; c++ {
			if ml := bd.GetLetter(r, c); ml != 0 {
				p.Board[r*15+c] = uint8(ml.Unblank())
			}
		}
	}
	p.Rack = counts(g.RackFor(me).TilesOn())
	p.Unseen = counts(tilemapping.MachineWord(g.Bag().Peek()), g.RackFor(1-me).TilesOn())
	p.Glob = [3]float32{float32(g.Bag().TilesRemaining()) / 100, float32(g.PointsFor(me)-g.PointsFor(1-me)) / 100, float32(turn) / 30}
	return p
}
func b2f(b bool) float32 {
	if b {
		return 1
	}
	return 0
}
func addTokens(p *Pos, g *game.Game, plays []*move.Move, maxm int, withExch bool) {
	bd := g.Board()
	type cand struct {
		t             Token
		blanks, score int
	}
	best := map[[4]int]cand{}
	var exch []Token
	for _, m := range plays {
		switch m.Action() {
		case move.MoveTypePlay:
			r, c, vert := m.CoordsAndVertical()
			d := 0
			dr, dc := 0, 1
			if vert {
				d = 1
				dr, dc = 1, 0
			}
			tiles := m.Tiles()
			var sb strings.Builder
			blanks := 0
			for j, t := range tiles {
				var l tilemapping.MachineLetter
				if t == 0 {
					l = bd.GetLetter(r+dr*j, c+dc*j).Unblank()
				} else {
					l = t.Unblank()
					if t.IsBlanked() {
						blanks++
					}
				}
				sb.WriteByte(byte('A' + int(l) - 1))
			}
			w, ok := wordIdx[sb.String()]
			if !ok {
				continue
			}
			s := m.Score()
			L := len(tiles)
			t := Token{Widx: w, Mv: m, Desc: []int{0, r, c, d, L, int(w), s, blanks}}
			t.MF = [9]float32{float32(s) / 50, float32(L) / 15, float32(r) / 14, float32(c) / 14, float32(d), b2f(s >= 50), float32(m.TilesPlayed()) / 7, float32(blanks), 0}
			t.Leave = counts(m.Leave())
			k := [4]int{r, c, d, int(w)}
			if o, seen := best[k]; !seen || blanks < o.blanks || (blanks == o.blanks && s > o.score) {
				best[k] = cand{t, blanks, s}
			}
		case move.MoveTypeExchange:
			keep := counts(m.Leave())
			t := Token{Widx: 0, Mv: m, Leave: keep}
			t.MF[8] = 1
			t.Desc = []int{1}
			for _, v := range keep {
				t.Desc = append(t.Desc, int(v))
			}
			exch = append(exch, t)
		}
	}
	ps := make([]cand, 0, len(best))
	for _, v := range best {
		ps = append(ps, v)
	}
	if bd.IsEmpty() { // opening: the mirrored vertical copy of every play (the Python engine generates both orientations)
		n := len(ps)
		for i := 0; i < n; i++ {
			o := ps[i]
			m := o.t.Mv
			r, c, _ := m.CoordsAndVertical()
			tm := move.NewScoringMove(m.Score(), m.Tiles(), m.Leave(), true, m.TilesPlayed(), g.Alphabet(), c, r)
			t := o.t
			t.Mv = tm
			t.Desc = append([]int{}, o.t.Desc...)
			t.Desc[1], t.Desc[2], t.Desc[3] = c, r, 1
			t.MF[2], t.MF[3], t.MF[4] = float32(c)/14, float32(r)/14, 1
			ps = append(ps, cand{t, o.blanks, o.score})
		}
	}
	key := func(c cand) [5]int { return [5]int{-c.score, int(c.t.Widx), c.t.Desc[1], c.t.Desc[2], c.t.Desc[3]} }
	sort.Slice(ps, func(i, j int) bool {
		a, b := key(ps[i]), key(ps[j])
		for k := 0; k < 5; k++ {
			if a[k] != b[k] {
				return a[k] < b[k]
			}
		}
		return false
	})
	if len(ps) > maxm {
		ps = ps[:maxm]
	}
	for _, v := range ps {
		p.Tokens = append(p.Tokens, v.t)
	}
	if withExch && g.Bag().TilesRemaining() >= 7 {
		p.Tokens = append(p.Tokens, exch...)
	}
}

// ---- globals
var (
	cfg     *config.Config
	gd      *kwg.KWG
	rules   *game.GameRules
	winpcts [][]float32
	leaves  *equity.CombinedStaticCalculator
	playSem chan struct{}
	exchOK  bool
	cl      *Client
	pool    chan *BotProc
	hastyConf *bot.BotConfig // built once: Config holds a lock and must not be copied per move
)

func newGen(g *game.Game) *movegen.GordonGenerator {
	return movegen.NewGordonGenerator(gd, g.Board(), rules.LetterDistribution())
}

// tokens of the position for the player on turn
func tokensFor(g *game.Game, gen *movegen.GordonGenerator, turn, maxm int) *Pos {
	p := buildPos(g, turn)
	gen.SetGame(g)
	addTokens(p, g, gen.GenAll(g.RackFor(g.PlayerOnTurn()), exchOK && g.Bag().TilesRemaining() >= 7), maxm, exchOK)
	return p
}
func argmax(l []float32) int {
	k := 0
	for i := range l {
		if l[i] > l[k] {
			k = i
		}
	}
	return k
}

// ---- Macondo bots
type BotProc struct {
	in  *bufio.Writer
	out *bufio.Reader
}

func startBot(macondo string) *BotProc {
	cmd := exec.Command(filepath.Join(macondo, "bin/lexmem_bot"))
	cmd.Dir = macondo
	cmd.Env = append(os.Environ(), "GOMAXPROCS=2", "SIM_THREADS=1", "GOMEMLIMIT="+envS("BOT_MEM", "2GiB"))
	cmd.Stderr = nil
	wi, err := cmd.StdinPipe()
	if err != nil {
		panic(err)
	}
	ro, err := cmd.StdoutPipe()
	if err != nil {
		panic(err)
	}
	if err := cmd.Start(); err != nil {
		panic(err)
	}
	return &BotProc{in: bufio.NewWriter(wi), out: bufio.NewReader(ro)}
}
func letter(ml tilemapping.MachineLetter) byte {
	if ml == 0 {
		return '?'
	}
	if ml.IsBlanked() {
		return byte('a' + int(ml.Unblank()) - 1)
	}
	return byte('A' + int(ml) - 1)
}

// CGP of the position for the player on turn; the opponent's rack is left out (it stays among the unseen tiles)
func toCGP(g *game.Game) string {
	bd := g.Board()
	rows := make([]string, 15)
	for r := 0; r < 15; r++ {
		var sb strings.Builder
		run := 0
		for c := 0; c < 15; c++ {
			ml := bd.GetLetter(r, c)
			if ml == 0 {
				run++
				continue
			}
			if run > 0 {
				sb.WriteString(strconv.Itoa(run))
				run = 0
			}
			sb.WriteByte(letter(ml))
		}
		if run > 0 {
			sb.WriteString(strconv.Itoa(run))
		}
		rows[r] = sb.String()
	}
	me := g.PlayerOnTurn()
	var rk strings.Builder
	for _, t := range g.RackFor(me).TilesOn() {
		rk.WriteByte(letter(t))
	}
	return fmt.Sprintf("%s %s/ %d/%d %d lex NWL23;", strings.Join(rows, "/"), rk.String(), g.PointsFor(me), g.PointsFor(1-me), g.ScorelessTurns())
}

// the move a Macondo bot plays for the player on turn
func botMove(g *game.Game, gen *movegen.GordonGenerator, code string, secs float64) *move.Move {
	if code == "HASTY_BOT" {
		gc := g.Copy()
		gc.ThrowRacksInFor(1 - gc.PlayerOnTurn())
		tp, err := bot.NewBotTurnPlayerFromGame(gc, hastyConf, pb.BotRequest_HASTY_BOT)
		if err != nil {
			panic(err)
		}
		return tp.GenerateMoves(1)[0]
	}
	b := <-pool
	fmt.Fprintf(b.in, "%s %g %s\n", code, secs, toCGP(g))
	b.in.Flush()
	line, err := b.out.ReadString('\n')
	pool <- b
	if err != nil {
		panic(fmt.Sprintf("lexmem_bot died: %v", err))
	}
	play := ""
	if strings.HasPrefix(line, "MOVE ") {
		play, _, _ = strings.Cut(strings.TrimSpace(line[5:]), "\t")
		play = strings.TrimSpace(play)
	}
	me := g.PlayerOnTurn()
	if play == "(Pass)" {
		return move.NewPassMove(g.RackFor(me).TilesOn(), g.Alphabet())
	}
	gen.SetGame(g)
	for _, m := range gen.GenAll(g.RackFor(me), g.Bag().TilesRemaining() >= 7) {
		if strings.TrimSpace(m.ShortDescription()) == play {
			return m
		}
	}
	fmt.Fprintf(os.Stderr, "NOMATCH %s %q -> HASTY_BOT\n", code, line)
	return botMove(g, gen, "HASTY_BOT", 0)
}

// ---- search
func winFor(onTurnSpread float64, unseen int) float64 { // Macondo's table: win probability of the player on turn
	x := int(math.Round(onTurnSpread))
	if x > equity.MaxRepresentedWinSpread {
		x = equity.MaxRepresentedWinSpread
	}
	if x < -equity.MaxRepresentedWinSpread {
		x = -equity.MaxRepresentedWinSpread
	}
	if unseen > 93 {
		unseen = 93
	}
	return float64(winpcts[equity.MaxRepresentedWinSpread-x][unseen])
}
func outcome(spread int) float64 {
	if spread > 0 {
		return 1
	}
	if spread < 0 {
		return 0
	}
	return 0.5
}

type Search struct {
	K, S, D, PM int
	Leaf        string
	Z           float64
}

// one playout: the candidate in world seed sd, D plies of net argmax, leaf -> my win probability
func (s *Search) playout(gc *game.Game, cand *move.Move, me, turn int, sd [32]byte) float64 {
	gc.SeedBag(sd)
	gc.Bag().Shuffle()
	if _, err := gc.SetRandomRack(1-me, nil); err != nil {
		panic(err)
	}
	if err := gc.PlayMove(cand, false, 0); err != nil {
		panic(err)
	}
	gen := newGen(gc)
	last, lastMover := cand, me
	for ply := 0; ply < s.D && gc.Playing() != pb.PlayState_GAME_OVER; ply++ {
		mover := gc.PlayerOnTurn()
		p := tokensFor(gc, gen, turn+1+ply, s.PM)
		var m *move.Move
		if len(p.Tokens) == 0 {
			m = move.NewPassMove(gc.RackFor(mover).TilesOn(), gc.Alphabet())
		} else {
			m = p.Tokens[argmax(cl.Infer(p).Logits)].Mv
		}
		if err := gc.PlayMove(m, false, 0); err != nil {
			panic(err)
		}
		last, lastMover = m, mover
	}
	if gc.Playing() == pb.PlayState_GAME_OVER {
		return outcome(gc.PointsFor(me) - gc.PointsFor(1-me))
	}
	q := gc.PlayerOnTurn()
	unseen := gc.Bag().TilesRemaining() + int(gc.RackFor(1-q).NumTiles())
	sq := float64(gc.PointsFor(q) - gc.PointsFor(1-q))
	var x float64 // spread from q's view, including the leaf estimate
	if s.Leaf == "net" {
		x = sq + float64(cl.Infer(buildPos(gc, turn+1+s.D)).Value[0])*100
	} else {
		lv := 0.0
		if last.Action() == move.MoveTypePlay || last.Action() == move.MoveTypeExchange {
			lv = leaves.LeaveValue(last.Leave())
		}
		if lastMover == q {
			x = sq + lv
		} else {
			x = sq - lv
		}
	}
	w := winFor(x, unseen)
	if q == me {
		return w
	}
	return 1 - w
}

// playout values of the policy's top K moves (token indices, by logit) in the same S worlds: vals[ci*S + si]; nil if K = 1
func (s *Search) evalCands(g *game.Game, p *Pos, logits []float32, me, turn int, rng *rand.Rand) ([]int, []float64) {
	idx := make([]int, len(logits))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(i, j int) bool { return logits[idx[i]] > logits[idx[j]] })
	k := s.K
	if k > len(idx) {
		k = len(idx)
	}
	cands := idx[:k]
	if k == 1 {
		return cands, nil
	}
	seeds := make([][32]byte, s.S)
	for i := range seeds {
		binary.LittleEndian.PutUint64(seeds[i][:], rng.Uint64())
		binary.LittleEndian.PutUint64(seeds[i][8:], rng.Uint64())
	}
	vals := make([]float64, k*s.S)
	var wg sync.WaitGroup
	for ci := range cands {
		for si := range seeds {
			playSem <- struct{}{} // global cap on playouts in flight (PAR): memory grows with them, GPU throughput does not
			gc := g.Copy()
			wg.Add(1)
			go func(ci, si int, gc *game.Game) {
				defer func() { <-playSem; wg.Done() }()
				vals[ci*s.S+si] = s.playout(gc, p.Tokens[cands[ci]].Mv, me, turn, seeds[si])
			}(ci, si, gc)
		}
	}
	wg.Wait()
	return cands, vals
}

// choose among the policy's top K by the mean playout value over S shared worlds; returns token index, candidates, estimates
func (s *Search) choose(g *game.Game, p *Pos, logits []float32, me, turn int, rng *rand.Rand) (int, []int, []float64) {
	cands, vals := s.evalCands(g, p, logits, me, turn, rng)
	if vals == nil {
		return cands[0], cands, []float64{1}
	}
	k := len(cands)
	// a candidate replaces the policy's top move (ci = 0) only if its mean paired difference (same worlds) exceeds Z standard
	// errors; among those, the highest mean. Z = 0: plain argmax of the mean.
	est := make([]float64, k)
	bi, bm := 0, 0.0
	for ci := range cands {
		sum, sumsq := 0.0, 0.0
		for si := 0; si < s.S; si++ {
			est[ci] += vals[ci*s.S+si]
			d := vals[ci*s.S+si] - vals[si]
			sum += d
			sumsq += d * d
		}
		est[ci] /= float64(s.S)
		if ci == 0 {
			continue
		}
		m := sum / float64(s.S)
		se := math.Sqrt(math.Max(sumsq/float64(s.S)-m*m, 0) / float64(s.S))
		if m > bm && m > s.Z*se {
			bi, bm = ci, m
		}
	}
	return cands[bi], cands, est
}

// ---- one game
type DecRec struct { // one search decision (REC=1): enough to rebuild the position and the search's targets
	Board  string    `json:"b"`
	Rack   [27]uint8 `json:"rk"`
	Unseen [27]uint8 `json:"un"`
	Scores [2]int    `json:"sc"`
	Bag    int       `json:"bag"`
	Turn   int       `json:"t"`
	Cands  [][]int   `json:"c"`
	Est    []float64 `json:"e"`
	Logits []float32 `json:"l"`
	Chosen int       `json:"k"`
}
type GameRec struct {
	G      int      `json:"g"`
	Seed   int64    `json:"seed"`
	First  bool     `json:"first"`
	Spread int      `json:"spread"`
	Final  [2]int   `json:"final"`
	NDec   [3]int   `json:"n"` // our moves by: net or search / Macondo bot (OURS or handoff) / pass with no tokens
	Dec    []DecRec `json:"dec,omitempty"`
}

func boardString(g *game.Game) string {
	var sb strings.Builder
	bd := g.Board()
	for r := 0; r < 15; r++ {
		for c := 0; c < 15; c++ {
			ml := bd.GetLetter(r, c)
			switch {
			case ml == 0:
				sb.WriteByte('.')
			default:
				sb.WriteByte(letter(ml))
			}
		}
	}
	return sb.String()
}

type Setup struct {
	Ours, Opp          string
	OursSecs, OppSecs  float64
	Handoff            int
	EGSecs             float64
	Srch               *Search
	Rec                bool
}

func playGame(g *game.Game, gen *movegen.GordonGenerator, st *Setup, gi int, seed int64) GameRec {
	var sd [32]byte
	binary.LittleEndian.PutUint64(sd[:], uint64(seed))
	g.SeedBag(sd)
	g.StartGame()
	us := g.PlayerOnTurn()
	if gi%2 == 1 {
		us = 1 - us
	}
	rec := GameRec{G: gi, Seed: seed, First: gi%2 == 0}
	rng := rand.New(rand.NewSource(seed*7919 + 17))
	turn := 0
	for g.Playing() != pb.PlayState_GAME_OVER && turn < 100 {
		me := g.PlayerOnTurn()
		var m *move.Move
		if me != us {
			m = botMove(g, gen, st.Opp, st.OppSecs)
		} else if st.Ours != "net" && st.Ours != "search" {
			m = botMove(g, gen, st.Ours, st.OursSecs)
			rec.NDec[1]++
		} else if st.Handoff > 0 && g.Bag().TilesRemaining()+int(g.RackFor(1-me).NumTiles()) <= 6+st.Handoff {
			m = botMove(g, gen, "SIMMING_BOT", st.EGSecs)
			rec.NDec[1]++
		} else {
			p := tokensFor(g, gen, turn, MAXM)
			if len(p.Tokens) == 0 {
				m = move.NewPassMove(g.RackFor(me).TilesOn(), g.Alphabet())
				rec.NDec[2]++
			} else {
				rep := cl.Infer(p)
				k := argmax(rep.Logits)
				if st.Ours == "search" {
					var cands []int
					var est []float64
					k, cands, est = st.Srch.choose(g, p, rep.Logits, me, turn, rng)
					if st.Rec {
						d := DecRec{Board: boardString(g), Rack: p.Rack, Unseen: p.Unseen, Scores: [2]int{g.PointsFor(me), g.PointsFor(1 - me)},
							Bag: g.Bag().TilesRemaining(), Turn: turn, Est: est, Chosen: k}
						for _, c := range cands {
							d.Cands = append(d.Cands, p.Tokens[c].Desc)
							d.Logits = append(d.Logits, rep.Logits[c])
						}
						rec.Dec = append(rec.Dec, d)
					}
				}
				m = p.Tokens[k].Mv
				rec.NDec[0]++
			}
		}
		if err := g.PlayMove(m, false, 0); err != nil {
			panic(err)
		}
		turn++
	}
	rec.Final = [2]int{g.PointsFor(us), g.PointsFor(1 - us)}
	rec.Spread = rec.Final[0] - rec.Final[1]
	return rec
}

// ---- positions mode (POSITIONS=file.jsonl with lines {"i", "cgp", "turn", ...}; other fields are ignored; NPOS = first N only):
// for each position the policy's top K moves and their playout values in each of the S worlds (the same worlds for every
// candidate; world seeds are drawn in sequence, so the first S' worlds of an S run are an S' run). A CGP leaves the opponent's
// rack among the unseen tiles; a random one is dealt before the root tokens are built so the bag count matches the position
// (each playout redeals it).
type PosIn struct {
	I    int    `json:"i"`
	CGP  string `json:"cgp"`
	Turn int    `json:"turn"`
}
type PosOut struct {
	I      int         `json:"i"`
	NTok   int         `json:"ntok"`
	Cands  [][]int     `json:"c"` // Token.Desc of each candidate, highest logit first
	Logits []float32   `json:"l"`
	Vals   [][]float64 `json:"v"` // [candidate][world]: win probability of the player on turn (null if one candidate)
}

func runPositions(path string, s *Search, conc int, seed0 int64, enc *json.Encoder, mu *sync.Mutex) {
	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	var ps []PosIn
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for sc.Scan() {
		var p PosIn
		if err := json.Unmarshal(sc.Bytes(), &p); err != nil {
			panic(err)
		}
		ps = append(ps, p)
	}
	f.Close()
	if n := envI("NPOS", 0); n > 0 && n < len(ps) {
		ps = ps[:n]
	}
	jobs := make(chan PosIn, len(ps))
	for _, p := range ps {
		jobs <- p
	}
	close(jobs)
	t0 := time.Now()
	done := 0
	var wg sync.WaitGroup
	for w := 0; w < conc; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for pi := range jobs {
				pg, err := cgp.ParseCGP(cfg, pi.CGP)
				if err != nil {
					panic(fmt.Sprintf("position %d: %v", pi.I, err))
				}
				pg.RecalculateBoard()
				g := pg.Game
				me := g.PlayerOnTurn()
				var sd [32]byte
				binary.LittleEndian.PutUint64(sd[:], uint64(seed0)+uint64(pi.I))
				g.SeedBag(sd)
				g.Bag().Shuffle()
				if _, err := g.SetRandomRack(1-me, nil); err != nil {
					panic(err)
				}
				p := tokensFor(g, newGen(g), pi.Turn, MAXM)
				o := PosOut{I: pi.I, NTok: len(p.Tokens)}
				if len(p.Tokens) > 0 {
					rep := cl.Infer(p)
					rng := rand.New(rand.NewSource((seed0+int64(pi.I))*7919 + 17))
					cands, vals := s.evalCands(g, p, rep.Logits, me, pi.Turn, rng)
					for ci, c := range cands {
						o.Cands = append(o.Cands, p.Tokens[c].Desc)
						o.Logits = append(o.Logits, rep.Logits[c])
						if vals != nil {
							row := make([]float64, s.S)
							for si := range row {
								row[si] = math.Round(vals[ci*s.S+si]*1e4) / 1e4
							}
							o.Vals = append(o.Vals, row)
						}
					}
				}
				mu.Lock()
				enc.Encode(o)
				done++
				if done%50 == 0 {
					fmt.Fprintf(os.Stderr, "%d/%d positions  %.0fs\n", done, len(ps), time.Since(t0).Seconds())
				}
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	fmt.Printf("FINAL positions %d search=K%d,S%d,D%d,%s,PM%d: %.0fs\n", len(ps), s.K, s.S, s.D, s.Leaf, s.PM, time.Since(t0).Seconds())
}

func main() {
	zerolog.SetGlobalLevel(zerolog.ErrorLevel)
	ex, _ := os.Executable()
	macondo := envS("MACONDO", filepath.Dir(filepath.Dir(ex)))
	cfg = &config.Config{}
	cfg.Load(nil)
	cfg.AdjustRelativePaths(filepath.Dir(ex))
	words, err := os.ReadFile(os.Getenv("LEXWORDS"))
	if err != nil {
		panic(err)
	}
	wordIdx = map[string]int32{}
	for i, w := range strings.Split(strings.TrimSpace(string(words)), "\n") {
		wordIdx[w] = int32(i)
	}
	st := &Setup{Ours: envS("OURS", "net"), Opp: envS("OPP", "HASTY_BOT"), OursSecs: envF("OURS_SECS", 10), OppSecs: envF("OPP_SECS", 10),
		Handoff: envI("HANDOFF", 0), EGSecs: envF("EG_SECS", 10), Rec: os.Getenv("REC") == "1",
		Srch: &Search{K: envI("K", 8), S: envI("S", 32), D: envI("D", 2), PM: envI("PM", MAXM), Leaf: envS("LEAF", "mac"), Z: envF("Z", 0)}}
	exchOK = os.Getenv("EXCH") != "0"
	ngames, conc, seed0 := envI("NGAMES", 100), envI("CONC", 64), int64(envI("SEED", 777000))
	gd, err = kwg.GetKWG(cfg.WGLConfig(), "NWL23")
	if err != nil {
		panic(err)
	}
	rules, err = game.NewBasicGameRules(cfg, "NWL23", board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		panic(err)
	}
	wp, err := cache.Load(cfg.WGLConfig(), "winpctfile:NWL20:winpct.csv", equity.WinPCTLoadFunc)
	if err != nil {
		panic(err)
	}
	winpcts = wp.([][]float32)
	hastyConf = &bot.BotConfig{Config: *cfg}
	leaves, err = equity.NewCombinedStaticCalculator("NWL23", cfg, "", "")
	if err != nil {
		panic(err)
	}
	playSem = make(chan struct{}, envI("PAR", 4096))
	needNet := st.Ours == "net" || st.Ours == "search"
	if needNet {
		cl = NewClient(os.Getenv("SOCK"), envI("BATCH", 512))
	}
	needPool := st.Opp != "HASTY_BOT" || (st.Ours != "net" && st.Ours != "search" && st.Ours != "HASTY_BOT") || (needNet && st.Handoff > 0)
	if needPool {
		nb := envI("NBOT", 24)
		pool = make(chan *BotProc, nb)
		for i := 0; i < nb; i++ {
			pool <- startBot(macondo)
		}
	}
	out, err := os.Create(os.Getenv("OUT"))
	if err != nil {
		panic(err)
	}
	bw := bufio.NewWriterSize(out, 1<<20)
	enc := json.NewEncoder(bw)
	var mu sync.Mutex
	fmt.Fprintf(os.Stderr, "setup %+v search %+v exch %v games %d from seed %d\n", *st, *st.Srch, exchOK, ngames, seed0)
	if pf := os.Getenv("POSITIONS"); pf != "" {
		runPositions(pf, st.Srch, conc, seed0, enc, &mu)
		bw.Flush()
		out.Close()
		return
	}
	jobs := make(chan int, ngames)
	for i := 0; i < ngames; i++ {
		jobs <- i
	}
	close(jobs)
	t0 := time.Now()
	var spreads []int
	var wg sync.WaitGroup
	for w := 0; w < conc; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			g, err := game.NewGame(rules, []*pb.PlayerInfo{{Nickname: "p1", RealName: "p1"}, {Nickname: "p2", RealName: "p2"}})
			if err != nil {
				panic(err)
			}
			gen := newGen(g)
			for gi := range jobs {
				r := playGame(g, gen, st, gi, seed0+int64(gi))
				mu.Lock()
				enc.Encode(r)
				spreads = append(spreads, r.Spread)
				if len(spreads)%50 == 0 {
					wins := 0.0
					for _, s := range spreads {
						if s > 0 {
							wins++
						}
					}
					fmt.Fprintf(os.Stderr, "%d games: win %.3f  %.0fs\n", len(spreads), wins/float64(len(spreads)), time.Since(t0).Seconds())
				}
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	bw.Flush()
	out.Close()
	n := float64(len(spreads))
	wins, ties, sum, sumsq := 0.0, 0.0, 0.0, 0.0
	for _, s := range spreads {
		if s > 0 {
			wins++
		} else if s == 0 {
			ties++
		}
		sum += float64(s)
		sumsq += float64(s * s)
	}
	wr := wins / n
	mean := sum / n
	fmt.Printf("FINAL ours=%s opp=%s(%gs) handoff=%d search=K%d,S%d,D%d,%s,PM%d,Z%g: %d games, win rate %.3f ± %.3f, ties %.3f, spread %+.1f ± %.1f, %.0fs\n",
		st.Ours, st.Opp, st.OppSecs, st.Handoff, st.Srch.K, st.Srch.S, st.Srch.D, st.Srch.Leaf, st.Srch.PM, st.Srch.Z, len(spreads), wr,
		math.Sqrt(wr*(1-wr)/n), ties/n, mean, math.Sqrt(math.Max(sumsq/n-mean*mean, 0)/n), time.Since(t0).Seconds())
}
