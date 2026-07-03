# WC_2026_predictor_poisson-Montecarlo
This repository intends to create (just for fun), a posible outcome for what will happen in the current World Cup.
## How it works

For every match, an expected goals value (λ) is calculated for each team, blending three sources:

| Source | Weight | What it captures |
|---|---|---|
| Current tournament stats (API-Football) | 50% | Most recent, tournament-specific form |
| Weighted historical form (last 10 games, 0.85 decay) | 35% | Recent momentum outside the tournament |
| Elo ratio | 15% | Long-term team strength, stabilizes small samples |

With both λ values (home and away), the model runs Monte Carlo simulations over a Poisson process
(30k–100k runs depending on the notebook) to get the full outcome distribution — not just the most
likely result, but Over/Under, BTTS and exact scoreline probabilities too.

**Real home advantage** (1.08–1.12x) only applies to **the United States, Mexico and Canada**,
whether they're listed as home or away — they're the only teams with an actual host advantage in
this World Cup. Every other match uses a much smaller nominal factor or none at all.

**Knockout rounds:** there's no draw possible, so the model separates 90 minutes, extra time and
penalties. Penalty odds sit close to 50/50, with a small adjustment for the team with the better Elo
rating.

## Repo structure

```
wc_predictor.py                    # engine: get_lambda, simular_partido, build_elo, recent_form
usa_vs_bosnia_wc2026.ipynb         # original single-match demo (USA vs Bosnia, Round of 32)
wc2026_partidos_hoy.ipynb          # generalized single-match predictor
wc2026_simulacion_restante.ipynb   # full remaining bracket simulation, no API needed
*.png                              # saved dashboards per match (e.g. Spain_vs_Austria_2026-07-02.png)
requirements.txt
README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/WC_2026_predictor_poisson-Montecarlo
cd WC_2026_predictor_poisson-Montecarlo
pip install -r requirements.txt
```

`usa_vs_bosnia_wc2026.ipynb` and `wc2026_partidos_hoy.ipynb` need an API-Football key (free tier
works, with a daily request limit). `wc2026_simulacion_restante.ipynb` doesn't need one.
## Data sources

- **Historical international results:** [martj42/international_results](https://github.com/martj42/international_results)
- **Live tournament data:** [API-Football](https://www.api-football.com/)

## Installation

```bash
git clone https://github.com/<your-username>/WC_2026_predictor_poisson-Montecarlo
cd WC_2026_predictor_poisson-Montecarlo
pip install -r requirements.txt
```

`partido_individual.ipynb` needs an API-Football key (free tier works, with a daily request limit).
`bracket_completo.ipynb` doesn't need one.

## Quick usage

```python
from wc_predictor import load_history, build_elo, simular_partido

df = load_history()
elo = build_elo(df)

result = simular_partido(df, elo, 'Spain', 'Austria', ronda='R32')
print(result)
```

## Assumptions and limitations

- Elo is calculated from 2000 onward, with a variable K-factor by competition type (World Cup >
  continental cups > qualifiers > friendlies).
- With little or no direct head-to-head history, that signal is weighted low (10%) on purpose — one
  or two matches from years ago shouldn't skew the prediction.
- API-Football lineups and injury data are only available ~1 hour before kickoff; running earlier
  just falls back to tournament stats.
- This is a support tool, not a guarantee — Poisson assumes independent goal events, a reasonable
  but imperfect simplification of real football.

## Credits

This project started from the idea behind [mar-antaya/world_cup_predictions](https://github.com/mar-antaya/world_cup_predictions),
which is also where the historical dataset (martj42) came from. From there the approach went a
different way: instead of a per-match XGBoost classifier, this uses a Poisson model with Monte Carlo
simulation, adds API-Football for live tournament data, and simulates the full knockout bracket
(Elo, recent form, extra time and penalties included).

## License

MIT
