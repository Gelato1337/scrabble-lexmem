// lexmem_bot: one Macondo bot per process for scrabble-lexmem evaluations.
// stdin, one request per line:  <BOTCODE> <seconds> <cgp...>     e.g.  SIMMING_BOT 15 15/15/... RACK/ 0/0 0 lex NWL23;
// stdout, one line per request: MOVE <ShortDescription>           e.g.  MOVE  8D WORD | MOVE (exch AB) | MOVE (Pass) | ERR <msg>
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
)

// CANDS=K > 0: after a simmed choice, append TAB + "C " + up to K candidates sorted by win probability,
// each "winprob,equity,iterations,ignored,<play>" separated by ";" (none when no sim ran: endgame / pre-endgame / static bots).
var cands int

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
	ex, _ := os.Executable()
	cfg := &config.Config{}
	cfg.Load(os.Args[1:])
	cfg.AdjustRelativePaths(filepath.Dir(ex))
	cands, _ = strconv.Atoi(os.Getenv("CANDS"))
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

func best(cfg *config.Config, code, secs, cgpStr string, threads int) (res string, rerr error) {
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
	tp, err := bot.NewBotTurnPlayerFromGame(g.Game, &bot.BotConfig{Config: *cfg, SimThreads: threads}, pb.BotRequest_BotCode(bc))
	if err != nil {
		return "", err
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
	m, err := tp.BestPlay(ctx)
	if err != nil || m == nil {
		fmt.Fprintf(os.Stderr, "FALLBACK %s error: %v\n", code, err)
		return tp.GenerateMoves(1)[0].ShortDescription(), nil
	}
	return m.ShortDescription() + simCands(tp), nil
}
