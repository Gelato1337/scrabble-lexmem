// lexmem_track: exact tracking of the opponent's rack over the whole move history against a static HastyBot: the ceiling of
// rack inference from everything a player sees (go/lexmem_invert is the same for the last move alone).
// Input lines (eval/track_infer.py prep):
//
//	{"i":.., "leave": the true kept tiles after the last move, "steps":[..]}   steps = the opponent's moves in order, each
//	  "cgp"    the position before the move, the opponent on turn with its true rack / our rack at that time
//	  "type"   play ("coords","tiles","played" = the placed tiles) | exch ("k" = the number of tiles) | pass
//	  "upre"   the tiles unseen to us right after its previous move (first move: at the start of the game)
//	  "draw"   the tiles we drew between that moment and this move
//
// We never see the opponent's rack; the true rack in the CGP gives the unseen tiles (bag + rack) and the diagnostics only.
// State after a move: a weight for every kept set L the opponent can hold (with the tiles E it put back when it exchanged).
// At its next move every rack R it can hold then (R = the placed tiles + a kept set; all 7-tile racks for an exchange or
// pass) gets the weight
//
//	prior(R) = Σ_{L ⊆ R} w(L, E) · Π_x C((upre − L − E)_x, (R − L)_x)   it drew R − L from the bag
//	           · Π_x C((upre − R)_x, draw_x)                              we then drew our tiles from what was left
//
// and HastyBot (Macondo HASTY_BOT in-process, on the real bag and our rack) is asked what it plays with R:
//
//	play        likelihood 1/t if the recorded play is one of the t moves tied at the top equity, else 0 (as lexmem_invert)
//	exch, pass  likelihood 1 if its top move (Macondo's own tie-break) is an exchange of k tiles / a pass, else 0; only
//	            the number of exchanged tiles is seen, so the kept set is the one HastyBot keeps with R
//
// The top move comes from Macondo's top-play generator (GenBestStaticTurn, the function its autoplay games use); a rack is
// rejected there when some move beats the recorded play's equity by more than 0.001, everything else is decided on the full
// move list. FAST=0 uses the full move list for every rack (check: the same output).
// No history (HIST=k uses the last k moves only; a step skipped for size or with no consistent rack): the step starts from
// the no-inference prior Π_x C(U_x, R_x), U = the tiles unseen to us before the move, so HIST=1 = lexmem_invert. A game with
// at most k moves of the opponent is its whole history: there HIST=k also knows what we drew before its first move (after an
// exchange of ours on the first turn the tiles we put back cannot be on its rack; lexmem_invert's prior does not see that).
// Output per line: the fields of lexmem_invert for the last move ("n","complete","marg","post","prior" = the no-inference
// prior of the true kept set,"lift","nmatch","truematch","tietrue","samples","secs","err"), and "steps" (moves used),
// "cands" (racks put to HastyBot over all steps), "slow" (of them on the full move list), "lost" (first step at which the
// true rack lost all its weight, -1 = never), "skipped" (steps over MAXCAND), "empty" (steps restarted without history
// because no rack was consistent).
// Env: WORKERS (default 8) HIST (default 0 = the whole history) MAXCAND (racks per step, default 4000000) SAMPLES FAST
// DEBUG (1: stderr lists the racks the top play rejects although the full move list accepts them)
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"math/bits"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/rs/zerolog"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/ai/bot"
	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

type Step struct {
	CGP    string `json:"cgp"`
	Type   string `json:"type"`
	Coords string `json:"coords"`
	Tiles  string `json:"tiles"`
	Played string `json:"played"`
	K      int    `json:"k"`
	Upre   string `json:"upre"`
	Draw   string `json:"draw"`
}
type In struct {
	I     int    `json:"i"`
	Steps []Step `json:"steps"`
	Leave string `json:"leave"`
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
	TieTrue   int            `json:"tietrue"`
	Steps     int            `json:"steps"`
	Cands     int            `json:"cands"`
	Slow      int            `json:"slow"`
	Lost      int            `json:"lost"`
	Skipped   int            `json:"skipped"`
	Empty     int            `json:"empty"`
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

// ours: Macondo machine letter (0 = blank, 1..26 = A..Z; a played blank has the high bit) -> our tile index (0..25, 26 = blank)
func ours(ml tilemapping.MachineLetter) int {
	if ml == 0 || ml.IsBlanked() {
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

// counts of a rack string ("AE?RS")
func counts(s string) (c [27]int, err error) {
	for _, ch := range s {
		switch {
		case ch == '?':
			c[26]++
		case ch >= 'A' && ch <= 'Z':
			c[ch-'A']++
		default:
			return c, fmt.Errorf("bad tile %q in %q", ch, s)
		}
	}
	return c, nil
}
func total(c *[27]int) int {
	n := 0
	for _, v := range c {
		n += v
	}
	return n
}
func rackString(c *[27]int) string {
	var sb strings.Builder
	for u := 0; u < 27; u++ {
		for j := 0; j < c[u]; j++ {
			if u == 26 {
				sb.WriteByte('?')
			} else {
				sb.WriteByte(byte('A' + u))
			}
		}
	}
	return sb.String()
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

// a multiset of at most 7 tiles as one number: the tiles in ascending order, 5 bits each (tile + 1)
func pack(c *[27]int) uint64 {
	k, sh := uint64(0), 0
	for u := 0; u < 27; u++ {
		for j := 0; j < c[u]; j++ {
			k |= uint64(u+1) << sh
			sh += 5
		}
	}
	return k
}
func unpack(k uint64) (c [27]int) {
	for ; k != 0; k >>= 5 {
		c[int(k&31)-1]++
	}
	return c
}

type ent struct {
	w float64
	e uint64 // the tiles the opponent put back into the bag after its draw (exchange), packed; 0 = none
}

// State: the opponent after one of its moves, before it draws
type State struct {
	nL   int              // kept tiles (the same number in every entry)
	m    map[uint64][]ent // kept set -> weights
	post [27]int          // the tiles unseen to us right after the move
	chk  bool             // post is set (not the start of the game)
}

var masks [8][]int // 7-bit masks by number of bits

func init() {
	for m := 0; m < 1<<7; m++ {
		n := bits.OnesCount(uint(m))
		masks[n] = append(masks[n], m)
	}
}

// prior weight of the rack R at the opponent's next move (see the header)
func (s *State) prior(R, upre, draw *[27]int) float64 {
	ev := 1.0
	for x := 0; x < 27; x++ {
		if draw[x] > 0 {
			ev *= binom(upre[x]-R[x], draw[x])
			if ev == 0 {
				return 0
			}
		}
	}
	var ts [7]int
	n := 0
	for u := 0; u < 27; u++ {
		for j := 0; j < R[u]; j++ {
			ts[n] = u
			n++
		}
	}
	sum := 0.0
	for _, mask := range masks[s.nL] {
		ok := true // each sub-multiset once: of equal tiles the first ones are taken
		for p := 1; p < 7; p++ {
			if mask>>p&1 == 1 && mask>>(p-1)&1 == 0 && ts[p] == ts[p-1] {
				ok = false
				break
			}
		}
		if !ok {
			continue
		}
		key, sh := uint64(0), 0
		var L [27]int
		for p := 0; p < 7; p++ {
			if mask>>p&1 == 1 {
				key |= uint64(ts[p]+1) << sh
				sh += 5
				L[ts[p]]++
			}
		}
		for _, en := range s.m[key] {
			var E [27]int
			if en.e != 0 {
				E = unpack(en.e)
			}
			t := en.w
			for x := 0; x < 27 && t != 0; x++ {
				if d := R[x] - L[x]; d > 0 {
					t *= binom(upre[x]-L[x]-E[x], d)
				}
			}
			sum += t
		}
	}
	return sum * ev
}

const nTie = 16       // moves requested on the full move list; the top-equity ties are counted among them
const fastTol = 0.001 // the top-play generator rejects a rack only when it beats the recorded play by more than this

var debug = os.Getenv("DEBUG") == "1" // list the racks the top play rejects although the full move list accepts them

type stepRes struct {
	next    *State
	n       int // racks in the enumeration
	cands   int // of them with a prior weight (put to HastyBot)
	slow    int
	nmatch  int
	z       float64
	trueOK  bool
	tieTrue int
	tooMany bool
	err     string
	// the last step
	marg, prior0 [27][4]float64
	z0, w0True   float64
	wTrue        float64
	keptL        [][27]int
	keptW        []float64
}

func doStep(cfg *config.Config, conf *bot.BotConfig, s Step, st *State, last bool, maxCand int, wantSamples, fast bool) (r stepRes) {
	pg, err := cgp.ParseCGP(cfg, s.CGP)
	if err != nil {
		r.err = err.Error()
		return
	}
	g := pg.Game
	g.RecalculateBoard()
	g.SetBackupMode(game.NoBackup)
	hasty := g.PlayerOnTurn()
	opp := 1 - hasty
	var pool, P, trueR, upre, draw [27]int // pool = the tiles unseen to us before the move = bag + the opponent's rack
	for _, ml := range g.Bag().Peek() {
		pool[ours(ml)]++
	}
	for _, ml := range g.RackFor(hasty).TilesOn() {
		trueR[ours(ml)]++
		pool[ours(ml)]++
	}
	if total(&trueR) != 7 {
		r.err = fmt.Sprintf("rack of %d tiles", total(&trueR))
		return
	}
	var played *move.Move
	want := ""
	switch s.Type {
	case "play":
		if P, err = counts(s.Played); err != nil {
			r.err = err.Error()
			return
		}
		if played, err = g.CreateAndScorePlacementMove(s.Coords, s.Tiles, s.Played, false); err != nil {
			r.err = err.Error()
			return
		}
		want = strings.TrimSpace(played.ShortDescription())
	case "exch", "pass":
	default:
		r.err = "step type " + s.Type
		return
	}
	nKeep := 7 - total(&P)
	var free [27]int // pool minus the placed tiles: what the kept set is drawn from
	for u := 0; u < 27; u++ {
		if free[u] = pool[u] - P[u]; free[u] < 0 || trueR[u] < P[u] {
			r.err = "placed tiles not in the rack"
			return
		}
	}
	r.n = countLeaves(free, nKeep)
	hist := st != nil
	if hist {
		if upre, err = counts(s.Upre); err != nil {
			r.err = err.Error()
			return
		}
		if draw, err = counts(s.Draw); err != nil {
			r.err = err.Error()
			return
		}
		if st.chk && upre != st.post {
			r.err = "upre does not follow from the previous step"
			return
		}
	}
	tp, err := bot.NewBotTurnPlayerFromGame(g, conf, pb.BotRequest_HASTY_BOT) // the full move list
	if err != nil {
		r.err = err.Error()
		return
	}
	tpF, err := bot.NewBotTurnPlayerFromGame(g, conf, pb.BotRequest_HASTY_BOT) // the top play only, as Macondo's autoplay
	if err != nil {
		r.err = err.Error()
		return
	}
	one := []*move.Move{played}
	fullList := func() (float64, int) { // the recorded play among the moves tied at the top equity of the full move list
		mv := tp.GenerateMoves(nTie)
		top := math.Inf(-1)
		for _, m := range mv {
			top = math.Max(top, m.Equity())
		}
		tied, match := 0, false
		for _, m := range mv {
			if m.Equity() >= top-1e-6 {
				tied++
				match = match || strings.TrimSpace(m.ShortDescription()) == want
			}
		}
		if match {
			return 1 / float64(tied), tied
		}
		return 0, tied
	}
	r.next = &State{nL: nKeep, m: map[uint64][]ent{}, chk: true}
	if s.Type == "exch" {
		r.next.nL = 7 - s.K
	}
	for u := 0; u < 27; u++ {
		r.next.post[u] = pool[u] - P[u]
	}
	var l, R [27]int
	mls := make([]tilemapping.MachineLetter, 0, 7)
	lml := make([]tilemapping.MachineLetter, 0, 7)
	leaf := func(w0 float64, countOnly bool) {
		for u := 0; u < 27; u++ {
			R[u] = P[u] + l[u]
		}
		pr := w0
		if hist {
			pr = st.prior(&R, &upre, &draw)
		}
		if countOnly {
			if pr > 0 {
				r.cands++
			}
			return
		}
		isTrue := R == trueR
		if last {
			r.z0 += w0
			for u := 0; u < 27; u++ {
				r.prior0[u][min(l[u], 3)] += w0
			}
			if isTrue {
				r.w0True = w0
			}
		}
		if pr <= 0 {
			return
		}
		r.cands++
		mls = mls[:0]
		for u := 0; u < 27; u++ {
			for j := 0; j < R[u]; j++ {
				mls = append(mls, mlOf(u))
			}
		}
		g.ThrowRacksInFor(hasty)
		rk := tilemapping.NewRack(g.Alphabet())
		rk.Set(mls)
		if err := g.SetRackForOnly(hasty, rk); err != nil {
			r.err = "set rack: " + err.Error()
			return
		}
		lik, tied := 0.0, 0
		var L [27]int
		var E uint64
		switch s.Type {
		case "play":
			L = l
			if fast { // the recorded play's equity with this rack against the top play's
				lml = lml[:0] // the kept tiles in Macondo's order (blank first)
				for j := 0; j < l[26]; j++ {
					lml = append(lml, 0)
				}
				for u := 0; u < 26; u++ {
					for j := 0; j < l[u]; j++ {
						lml = append(lml, mlOf(u))
					}
				}
				played.SetLeave(lml)
				tp.AssignEquity(one, g.Board(), g.Bag(), g.RackFor(opp))
				if top := aiturnplayer.GenBestStaticTurn(g, tpF, hasty); top.Equity() > played.Equity()+fastTol {
					if debug { // does the full move list accept the rack the top play rejects?
						desc, eq := strings.TrimSpace(top.ShortDescription()), top.Equity()
						if lk, _ := fullList(); lk > 0 {
							fmt.Fprintf(os.Stderr, "DISAGREE rack %s%s: top play %q %.3f, recorded play %q %.3f, bag %d\n", rackString(&R),
								map[bool]string{true: " (the true rack)"}[isTrue], desc, eq, want, played.Equity(), g.Bag().TilesRemaining())
						}
					}
					break
				}
			}
			r.slow++
			lik, tied = fullList()
		case "exch", "pass":
			top := aiturnplayer.GenBestStaticTurn(g, tpF, hasty)
			if s.Type == "exch" && top.Action() == move.MoveTypeExchange && top.TilesPlayed() == s.K {
				lik, tied = 1, 1
				var e [27]int
				for _, ml := range top.Leave() {
					L[ours(ml)]++
				}
				for _, ml := range top.Tiles() {
					e[ours(ml)]++
				}
				E = pack(&e)
			} else if s.Type == "pass" && top.Action() == move.MoveTypePass {
				lik, tied, L = 1, 1, R
			}
		}
		if isTrue {
			r.trueOK, r.tieTrue = lik > 0, tied
		}
		if lik == 0 {
			return
		}
		w := pr * lik
		r.z += w
		r.nmatch++
		if last {
			for u := 0; u < 27; u++ {
				r.marg[u][min(l[u], 3)] += w
			}
			if isTrue {
				r.wTrue = w
			}
			if wantSamples {
				r.keptL = append(r.keptL, l)
				r.keptW = append(r.keptW, r.z)
			}
		} else {
			key := pack(&L)
			r.next.m[key] = append(r.next.m[key], ent{w, E})
		}
	}
	var rec func(t, left int, w0 float64, countOnly bool)
	rec = func(t, left int, w0 float64, countOnly bool) {
		if r.err != "" {
			return
		}
		if t == 27 {
			if left == 0 {
				leaf(w0, countOnly)
			}
			return
		}
		for j := 0; j <= free[t] && j <= left; j++ {
			l[t] = j
			rec(t+1, left-j, w0*binom(pool[t], P[t]+j), countOnly) // the no-inference prior's factor of tile t
		}
		l[t] = 0
	}
	if r.n > maxCand { // the history may leave far fewer racks: count those with a prior weight first
		if hist {
			rec(0, nKeep, 1.0, true)
		}
		if !hist || r.cands > maxCand {
			r.tooMany = true
			return
		}
		r.cands = 0
	}
	rec(0, nKeep, 1.0, false)
	if r.z > 0 {
		for _, es := range r.next.m {
			for i := range es {
				es[i].w /= r.z
			}
		}
	}
	return
}

func track(cfg *config.Config, conf *bot.BotConfig, x In, hist, maxCand, nSamples int, fast bool) (out Out) {
	out = Out{I: x.I, Lift: -999, Lost: -1}
	t0 := time.Now()
	defer func() {
		if e := recover(); e != nil {
			out.Err = fmt.Sprintf("panic: %v", e)
		}
		out.Secs = time.Since(t0).Seconds()
	}()
	steps := x.Steps
	if len(steps) == 0 {
		out.Err = "no steps"
		return
	}
	var st *State // nil = no history: the next step starts from the no-inference prior
	if hist > 0 && hist < len(steps) {
		steps = steps[len(steps)-hist:]
	} else {
		st = &State{m: map[uint64][]ent{0: {{w: 1}}}} // the start of the game: the opponent holds nothing and draws 7
	}
	out.Steps = len(steps)
	for si, s := range steps {
		last := si == len(steps)-1
		r := doStep(cfg, conf, s, st, last, maxCand, nSamples > 0, fast)
		out.Cands += r.cands
		out.Slow += r.slow
		if r.err == "" && !r.tooMany && r.nmatch == 0 && st != nil { // no rack is consistent with the history: without it
			out.Empty++
			r = doStep(cfg, conf, s, nil, last, maxCand, nSamples > 0, fast)
			out.Cands += r.cands
			out.Slow += r.slow
		}
		if r.err != "" {
			out.Err = fmt.Sprintf("step %d: %s", si, r.err)
			return
		}
		if !r.tooMany && !r.trueOK && out.Lost < 0 {
			out.Lost = si
		}
		if !last {
			st = r.next
			if r.tooMany {
				out.Skipped++
				st = nil
			} else if r.nmatch == 0 {
				st = nil
			}
			continue
		}
		out.N = r.n
		if r.tooMany {
			out.Err = "too many kept sets"
			return
		}
		out.Complete, out.NMatch, out.TrueMatch, out.TieTrue = true, r.nmatch, r.trueOK, r.tieTrue
		if r.z0 > 0 {
			out.Prior = r.w0True / r.z0
		}
		if r.z == 0 {
			out.Err = "no consistent kept set"
			for u := 0; u < 27; u++ {
				for k := 0; k < 4; k++ {
					out.Marg[u][k] = r.prior0[u][k] / r.z0
				}
			}
			return
		}
		for u := 0; u < 27; u++ {
			for k := 0; k < 4; k++ {
				out.Marg[u][k] = r.marg[u][k] / r.z
			}
		}
		if r.wTrue > 0 {
			out.Post = r.wTrue / r.z
			out.Lift = math.Log2(out.Post / out.Prior)
		}
		if nSamples > 0 { // keptW holds the cumulative posterior weight
			rng := rand.New(rand.NewPCG(uint64(x.I), 7))
			for k := 0; k < nSamples; k++ {
				j := sort.SearchFloat64s(r.keptW, rng.Float64()*r.z)
				if j >= len(r.keptL) {
					j = len(r.keptL) - 1
				}
				out.Samples = append(out.Samples, rackString(&r.keptL[j]))
			}
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
	W, hist, maxCand, nSamples := envI("WORKERS", 8), envI("HIST", 0), envI("MAXCAND", 4000000), envI("SAMPLES", 0)
	fast := os.Getenv("FAST") != "0"
	conf := &bot.BotConfig{Config: *cfg} // one bot config for all workers, built before they start (Config holds a lock)
	in := make(chan In, 256)
	outc := make(chan Out, 256)
	var wg sync.WaitGroup
	for w := 0; w < W; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for x := range in {
				outc <- track(cfg, conf, x, hist, maxCand, nSamples, fast)
			}
		}()
	}
	go func() { wg.Wait(); close(outc) }()
	go func() {
		sc := bufio.NewScanner(os.Stdin)
		sc.Buffer(make([]byte, 1<<22), 1<<22)
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
	n, nerr, cands, slow := 0, 0, 0, 0
	secs := 0.0
	for o := range outc {
		enc.Encode(o)
		bw.Flush()
		n++
		cands += o.Cands
		slow += o.Slow
		secs += o.Secs
		if o.Err != "" {
			nerr++
		}
	}
	bw.Flush()
	fmt.Fprintf(os.Stderr, "%d positions, %d with an error; %d racks put to HastyBot (%d on the full move list), %.0f s summed over positions\n", n, nerr, cands, slow, secs)
}
