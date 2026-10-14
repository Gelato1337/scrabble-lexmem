package bot

import "github.com/domino14/macondo/montecarlo"

// Simmer exposes the bot's Monte Carlo simmer (for scrabble-lexmem: per-candidate win probabilities after BestPlay).
func (p *BotTurnPlayer) Simmer() *montecarlo.Simmer { return p.simmer }
