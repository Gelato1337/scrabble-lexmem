package bot

import (
	"context"
	"math"

	"github.com/domino14/word-golib/kwg"

	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
)

// Simmer exposes the bot's Monte Carlo simmer (for scrabble-lexmem: per-candidate win probabilities after BestPlay).
func (p *BotTurnPlayer) Simmer() *montecarlo.Simmer { return p.simmer }

// LexmemSimPlies: the stage decision of eliteBestPlay for the position: the number of simulation plies, or 0 when the
// endgame or pre-endgame solver would play instead (at most 8 unseen tiles), where no opponent rack is sampled.
func (p *BotTurnPlayer) LexmemSimPlies() int {
	unseen := int(p.Game.RackFor(p.Game.NextPlayer()).NumTiles()) + p.Game.Bag().TilesRemaining()
	switch {
	case unseen <= 8:
		return 0
	case unseen <= 14:
		return unseen
	case p.minSimPlies > 2:
		return p.minSimPlies
	}
	return 2
}

// LexmemSim: the simulation of nonEndgameBest (the same simmer set-up, in the same order) over plays, with one change:
// when racks is not empty the opponent's kept tiles are sampled from racks on every iteration (complete-posterior mode:
// kept tiles + a refill from the bag) instead of a uniform rack from the unseen tiles.
// iters = 0: the bot's own search (Stop99 until ctx ends). iters > 0: exactly that many iterations on the calling
// goroutine with no stopping rule, so no play is pruned (evaluation of given plays).
func (p *BotTurnPlayer) LexmemSim(ctx context.Context, plies int, plays []*move.Move, racks []montecarlo.InferredRack, iters int) error {
	p.simmer.Init(p.Game, p.simmerCalcs, p.simmerCalcs[0].(*equity.CombinedStaticCalculator), p.Config())
	p.simmer.TryLoadWMP(p.Config().WGLConfig(), p.Game.LexiconName())
	if p.simThreads != 0 {
		p.simmer.SetThreads(p.simThreads)
	}
	if err := p.simmer.PrepareSim(plies, plays); err != nil {
		return err
	}
	if iters > 0 {
		p.simmer.SetStoppingCondition(montecarlo.StopNone)
	} else {
		p.simmer.SetStoppingCondition(montecarlo.Stop99)
	}
	if p.cfg.StochasticStaticEval {
		p.simmer.SetStochasticStaticEval(true)
	}
	if len(racks) > 0 { // after PrepareSim, which switches inference off
		p.simmer.SetInferences(racks, len(racks[0].Leave), montecarlo.InferenceCompletePosterior)
	}
	if iters > 0 {
		p.simmer.SimSingleThread(iters, plies)
		return nil
	}
	return p.simmer.Simulate(ctx)
}

// LexmemBestPlay: BestPlay of an elite bot with the pre-endgame solver starting earlier. With more than 8 and at most
// pegUnseen unseen tiles (9 = two tiles in the bag, 10 = three) Macondo's pre-endgame solver plays instead of the
// simulation, set up as in preendgameBest (endgame plies by the spread, iterative deepening, early cutoff); threads > 0 sets
// its thread count (its default is every CPU). Any other position: BestPlay.
func (p *BotTurnPlayer) LexmemBestPlay(ctx context.Context, pegUnseen, threads int) (*move.Move, error) {
	unseen := int(p.Game.RackFor(p.Game.NextPlayer()).NumTiles()) + p.Game.Bag().TilesRemaining()
	if unseen <= 8 || unseen > pegUnseen || !HasPreendgame(p.botType) {
		return p.BestPlay(ctx)
	}
	gd, err := kwg.GetKWG(p.Game.Config().WGLConfig(), p.Game.LexiconName())
	if err != nil {
		return nil, err
	}
	if err = p.preendgamer.Init(p.Game, gd); err != nil {
		return nil, err
	}
	if threads > 0 {
		p.preendgamer.SetThreads(threads)
	}
	switch ourSpread := math.Abs(float64(p.Game.SpreadFor(p.Game.PlayerOnTurn()))); {
	case ourSpread >= 100:
		p.preendgamer.SetEndgamePlies(2)
	case ourSpread >= 75:
		p.preendgamer.SetEndgamePlies(3)
	case ourSpread >= 50:
		p.preendgamer.SetEndgamePlies(4)
	default:
		p.preendgamer.SetEndgamePlies(5)
	}
	p.preendgamer.SetIterativeDeepening(true)
	p.preendgamer.SetEarlyCutoffOptim(true)
	moves, err := p.preendgamer.Solve(ctx)
	if err != nil {
		return nil, err
	}
	if len(moves) == 0 {
		return nil, context.DeadlineExceeded
	}
	p.lastCalculatedDetails = p.preendgamer.ShortDetails()
	return moves[0].Play, nil
}
