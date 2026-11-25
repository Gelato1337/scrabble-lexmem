// lexmem_bot: one Macondo bot per process for scrabble-lexmem evaluations.
// stdin, one request per line:  <BOTCODE> <seconds> <cgp...>     e.g.  SIMMING_BOT 15 15/15/... RACK/ 0/0 0 lex NWL23;
// stdout, one line per request: MOVE <ShortDescription>           e.g.  MOVE  8D WORD | MOVE (exch AB) | MOVE (Pass) | ERR <msg>
// Optional opponent history for rack inference (SIMMING_INFER_BOT reads the game history; a bare CGP has none):
//   <cgp> TAB <cgp before the opponent's last tile play: opponent on turn, its rack = the tiles it played / our rack>
//         TAB <coords> TAB <tiles ('.' = played through, lowercase = blank)> TAB <the played tiles as a rack>
// The position is then rebuilt as that CGP + the opponent's play (one history event) and checked against <cgp>.
// BOT_LOG=info prints Macondo's log (inference lines) to stderr; every history use prints HIST ok / HIST err.
// Variants of the elite bots (more thinking at the same place in the game):
//   MIN_SIM_PLIES=n  simulate n plies instead of 2 while more than 14 tiles are unseen (Macondo's MinSimPlies)
//   PEG_UNSEEN=n     n > 8: the pre-endgame solver plays from n unseen tiles on (9 = two tiles in the bag, 10 = three) instead
//                    of the simulation (ai/bot LexmemBestPlay), SIM_THREADS threads, PEG_SECS seconds (default: the
//                    request's); stderr gets one line "PEG <unseen> <seconds used> <play>" per such move
package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/rs/zerolog"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

// CANDS=K > 0: after a simmed choice, append TAB + "C " + up to K candidates sorted by win probability,
// each "winprob,equity,iterations,ignored,<play>" separated by ";" (none when no sim ran: endgame / pre-endgame / static bots).
var cands int

var minPlies, pegUnseen int // MIN_SIM_PLIES, PEG_UNSEEN
var pegSecs float64         // PEG_SECS

func simCands(tp *bot.BotTurnPlayer) (out string) {
	defer func() {
		if r := recover(); r != nil {
			out = ""
		}
	}()
	sm := tp.Simmer()
	if cands <= 0 || sm == nil || sm.Iterations() == 0 {
		return ""
	}
	parts := []string{}
	for i, sp := range sm.PlaysByWinProb().PlaysNoLock() {
		if i >= cands {
			break
		}
		ign := 0
		if sp.IsIgnored() {
			ign = 1
		}
		parts = append(parts, fmt.Sprintf("%.5f,%.3f,%d,%d,%s", sp.WinProb(), sp.EquityMean(), sp.WinProbIterations(), ign,
			strings.TrimSpace(sp.Move().ShortDescription())))
	}
	return "\tC " + strings.Join(parts, ";")
}

func main() {
	zerolog.SetGlobalLevel(zerolog.ErrorLevel)
	if os.Getenv("BOT_LOG") == "info" {
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
	}
	ex, _ := os.Executable()
	cfg := &config.Config{}
	cfg.Load(os.Args[1:])
	cfg.AdjustRelativePaths(filepath.Dir(ex))
	cands, _ = strconv.Atoi(os.Getenv("CANDS"))
	minPlies, _ = strconv.Atoi(os.Getenv("MIN_SIM_PLIES"))
	pegUnseen, _ = strconv.Atoi(os.Getenv("PEG_UNSEEN"))
	pegSecs, _ = strconv.ParseFloat(os.Getenv("PEG_SECS"), 64)
	threads, _ := strconv.Atoi(os.Getenv("SIM_THREADS"))
	if threads <= 0 {
		threads = 1
	}
	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 1<<20), 1<<20)
	out := bufio.NewWriter(os.Stdout)
	for in.Scan() {
		f := strings.SplitN(strings.TrimSpace(in.Text()), " ", 3)
		if len(f) < 3 {
			continue
		}
		mv, err := best(cfg, f[0], f[1], f[2], threads)
		if err != nil {
			fmt.Fprintf(out, "ERR %v\n", strings.ReplaceAll(err.Error(), "\n", " "))
		} else {
			fmt.Fprintf(out, "MOVE %s\n", strings.TrimSpace(mv))
		}
		out.Flush()
	}
}

// withHistory: the current position rebuilt as (position before the opponent's last play) + that play, so that the game
// history holds the opponent's move; nil if anything does not match the current position cur.
func withHistory(cfg *config.Config, cur *game.Game, h []string) (*game.Game, error) {
	pg, err := cgp.ParseCGP(cfg, h[0])
	if err != nil {
		return nil, err
	}
	g2 := pg.Game
	g2.RecalculateBoard()
	g2.SetBackupMode(game.InteractiveGameplayMode)
	g2.SetStateStackLength(1)
	m, err := g2.CreateAndScorePlacementMove(h[1], h[2], h[3], false)
	if err != nil {
		return nil, err
	}
	opp := g2.PlayerOnTurn()
	if err := g2.PlayMove(m, true, 0); err != nil {
		return nil, err
	}
	g2.ThrowRacksInFor(opp) // the opponent's refill is unknown to us: back into the bag, as in a bare CGP
	me := g2.PlayerOnTurn()
	if g2.PointsFor(me) != cur.PointsFor(cur.PlayerOnTurn()) || g2.PointsFor(opp) != cur.PointsFor(1-cur.PlayerOnTurn()) {
		return nil, fmt.Errorf("scores %d/%d vs %d/%d", g2.PointsFor(me), g2.PointsFor(opp), cur.PointsFor(cur.PlayerOnTurn()), cur.PointsFor(1-cur.PlayerOnTurn()))
	}
	if g2.RackFor(me).String() != cur.RackFor(cur.PlayerOnTurn()).String() || g2.Bag().TilesRemaining() != cur.Bag().TilesRemaining() {
		return nil, fmt.Errorf("rack or bag differs")
	}
	for r := 0; r < 15; r++ {
		for c := 0; c < 15; c++ {
			if g2.Board().GetLetter(r, c) != cur.Board().GetLetter(r, c) {
				return nil, fmt.Errorf("board differs at %d,%d", r, c)
			}
		}
	}
	return g2, nil
}

func best(cfg *config.Config, code, secs, cgpStr string, threads int) (res string, rerr error) {
	parts := strings.Split(cgpStr, "\t")
	cgpStr = parts[0]
	bc, ok := pb.BotRequest_BotCode_value[code]
	if !ok {
		return "", fmt.Errorf("unknown bot code %s", code)
	}
	s, err := strconv.ParseFloat(secs, 64)
	if err != nil {
		return "", err
	}
	g, err := cgp.ParseCGP(cfg, cgpStr)
	if err != nil {
		return "", err
	}
	g.RecalculateBoard()
	g.SetBackupMode(game.InteractiveGameplayMode)
	g.SetStateStackLength(1)
	gm := g.Game
	if len(parts) == 5 {
		if g2, err := withHistory(cfg, gm, parts[1:]); err == nil {
			gm = g2
			fmt.Fprintln(os.Stderr, "HIST ok")
		} else {
			fmt.Fprintf(os.Stderr, "HIST err %v\n", err)
		}
	}
	tp, err := bot.NewBotTurnPlayerFromGame(gm, &bot.BotConfig{Config: *cfg, SimThreads: threads, MinSimPlies: minPlies}, pb.BotRequest_BotCode(bc))
	if err != nil {
		return "", err
	}
	unseen := int(gm.RackFor(gm.NextPlayer()).NumTiles()) + gm.Bag().TilesRemaining()
	early := pegUnseen > 8 && unseen > 8 && unseen <= pegUnseen // the pre-endgame solver earlier than the bot's own stage
	if early && pegSecs > 0 {
		s = pegSecs
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(s*float64(time.Second)))
	defer cancel()
	// Macondo bug: an endgame solve that times out can return an empty sequence and BestPlay panics
	// (ai/bot/elite.go endGameBest seq[0]). Fall back to the static (HastyBot) best play for that turn.
	defer func() {
		if r := recover(); r != nil {
			fmt.Fprintf(os.Stderr, "FALLBACK %s panic: %v\n", code, r)
			res, rerr = tp.GenerateMoves(1)[0].ShortDescription(), nil
		}
	}()
	var m *move.Move
	if early {
		t0 := time.Now()
		m, err = tp.LexmemBestPlay(ctx, pegUnseen, threads)
		if err == nil && m != nil {
			fmt.Fprintf(os.Stderr, "PEG %d %.1f %s\n", unseen, time.Since(t0).Seconds(), strings.TrimSpace(m.ShortDescription()))
		}
	} else {
		m, err = tp.BestPlay(ctx)
	}
	if err != nil || m == nil {
		fmt.Fprintf(os.Stderr, "FALLBACK %s error: %v\n", code, err)
		return tp.GenerateMoves(1)[0].ShortDescription(), nil
	}
	return m.ShortDescription() + simCands(tp), nil
}
