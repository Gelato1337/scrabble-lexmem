package bot

import (
	"context"

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
