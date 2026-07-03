"""
wc_predictor.py
Dos modos de uso:
  - simular_partido(...)   → solo con histórico (Elo + forma), sin API.
                              Es lo que usa bracket_completo.ipynb.

  - get_lambda(...) recibe stats de API-Football directamente si las tienes
                              (gf_avg / ga_avg del torneo actual). Es lo que
                              usa partido_individual.ipynb, combinando esas
                              stats con Elo y forma de este mismo módulo.
"""

import numpy as np
import pandas as pd

# ── Constantes del modelo ─────────────────────────────────────────────────────

N_SIMS       = 30_000
AVG_WC_GOALS = 2.52
ELO_INIT     = 1500
ELO_K        = 30

HOST_NATIONS = {'United States', 'USA', 'Canada', 'Mexico'}
HOST_FACTOR  = 1.08

HIST_URL = 'https://raw.githubusercontent.com/martj42/international_results/master/results.csv'

NAME_MAP = {
    'USA': 'United States',
    'United States of America': 'United States',
    'Korea Republic': 'South Korea',
    'Türkiye': 'Turkey',
    'Côte d’Ivoire': 'Ivory Coast',
    "Côte d'Ivoire": 'Ivory Coast',
    'Czechia': 'Czech Republic',
    'Curaçao': 'Curacao',
    'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
    'Bosnia and Herzegowina': 'Bosnia and Herzegovina',
    'Congo DR': 'DR Congo',
    'Democratic Republic of Congo': 'DR Congo',
}

# Se usa solo si un equipo no tiene historial suficiente en el CSV
ELO_FALLBACK = {
    'Argentina': 2140, 'France': 2090, 'Spain': 2070, 'Brazil': 2040,
    'England': 2030, 'Portugal': 2020, 'Belgium': 1960, 'Colombia': 1940,
    'Croatia': 1920, 'Netherlands': 1980, 'Germany': 1970, 'Morocco': 1840,
    'Mexico': 1850, 'United States': 1830, 'Switzerland': 1820, 'Austria': 1810,
    'Canada': 1760, 'Paraguay': 1780, 'Norway': 1770, 'Algeria': 1760,
    'Egypt': 1750, 'Ghana': 1720, 'Australia': 1710, 'Cape Verde': 1660,
    'South Africa': 1650, 'Japan': 1800, 'Ivory Coast': 1730, 'Sweden': 1760,
    'Ecuador': 1800, 'DR Congo': 1650, 'Senegal': 1800,
    'Bosnia and Herzegovina': 1680,
}

# Ajuste manual por bajas confirmadas — se llena a mano antes de simular,
# ej. INJURY_FACTOR['France'] = 0.95
INJURY_FACTOR = {}


# ── Carga de histórico ────────────────────────────────────────────────────────

def load_history(url=HIST_URL):
    """Descarga y limpia el histórico de martj42. Si falla, regresa un
    DataFrame vacío con las columnas esperadas para que el resto del
    pipeline no truene."""
    try:
        df = pd.read_csv(url, parse_dates=['date'])
        df['home_team'] = df['home_team'].map(lambda x: NAME_MAP.get(x, x))
        df['away_team'] = df['away_team'].map(lambda x: NAME_MAP.get(x, x))
        df = df.dropna(subset=['home_score', 'away_score'])
        return df
    except Exception as e:
        print('No se pudo cargar el histórico, uso fallback:', e)
        return pd.DataFrame(columns=['date', 'home_team', 'away_team',
                                      'home_score', 'away_score'])


# ── Elo ────────────────────────────────────────────────────────────────────────

def expected_score(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))


def update_elo(r_a, r_b, score_a, k=ELO_K):
    exp_a = expected_score(r_a, r_b)
    return r_a + k * (score_a - exp_a)


def build_elo(df, elo_init=ELO_INIT, fallback=ELO_FALLBACK):
    """Recorre el histórico cronológicamente y devuelve el rating Elo final
    de cada selección. Los equipos sin partidos suficientes caen a
    ELO_FALLBACK."""
    ratings = {}
    for _, row in df.sort_values('date').iterrows():
        h, a = row['home_team'], row['away_team']
        hs, as_ = row['home_score'], row['away_score']

        rh = ratings.get(h, elo_init)
        ra = ratings.get(a, elo_init)

        if hs > as_:
            sh, sa = 1, 0
        elif hs < as_:
            sh, sa = 0, 1
        else:
            sh, sa = 0.5, 0.5

        ratings[h] = update_elo(rh, ra, sh)
        ratings[a] = update_elo(ra, rh, sa)

    for team, val in fallback.items():
        ratings.setdefault(team, val)

    return ratings


# ── Forma reciente ────────────────────────────────────────────────────────────

def recent_form(df, team, n=10, decay=0.85):
    """Goles a favor / en contra ponderados de los últimos n partidos,
    con decay geométrico (el más reciente pesa más)."""
    if df.empty:
        return {'gf': 1.3, 'ga': 1.1}

    tmp = df[(df['home_team'] == team) | (df['away_team'] == team)].sort_values('date').tail(n)
    if tmp.empty:
        return {'gf': 1.3, 'ga': 1.1}

    gf, ga = [], []
    for _, row in tmp.iterrows():
        if row['home_team'] == team:
            gf.append(row['home_score']); ga.append(row['away_score'])
        else:
            gf.append(row['away_score']); ga.append(row['home_score'])

    w = np.array([decay ** (len(gf) - 1 - i) for i in range(len(gf))])
    w = w / w.sum()

    return {'gf': float(np.average(gf, weights=w)),
            'ga': float(np.average(ga, weights=w))}


def team_base_stats(df, team):
    """Stats mínimas para get_lambda cuando no hay datos de API-Football
    (modo solo-histórico, usado en bracket_completo.ipynb)."""
    f = recent_form(df, team)
    return {'gf_avg': max(f['gf'], 0.4), 'ga_avg': max(f['ga'], 0.4)}


def host_factor(team):
    return HOST_FACTOR if team in HOST_NATIONS else 1.0


# ── λ (goles esperados) ────────────────────────────────────────────────────────

def get_lambda(attacker_stats, defender_stats, attacker_form, defender_form,
               elo_att, elo_def, home_factor=1.0, injury_factor=1.0,
               avg_goals=AVG_WC_GOALS):
    """
    λ combina tres fuentes:
      50% stats del torneo actual (attacker_stats / defender_stats — de API-Football
          si las tienes, o las mismas de recent_form si no)
      35% forma histórica ponderada (attacker_form / defender_form)
      15% ratio de Elo

    attacker_stats / defender_stats deben traer 'gf_avg' y 'ga_avg'.
    """
    gf_api = attacker_stats.get('gf_avg', avg_goals)
    ga_api = defender_stats.get('ga_avg', avg_goals)
    lam_api = (gf_api / avg_goals) * (ga_api / avg_goals) * avg_goals

    gf_hist = attacker_form['gf']
    ga_hist = defender_form['ga']
    lam_hist = (gf_hist / avg_goals) * (ga_hist / avg_goals) * avg_goals

    elo_ratio = 10 ** ((elo_att - elo_def) / 800)
    lam_elo = avg_goals * elo_ratio

    lam = 0.50 * lam_api + 0.35 * lam_hist + 0.15 * lam_elo
    lam *= home_factor
    lam *= injury_factor

    return round(max(lam, 0.2), 3)


# ── Simulación de un partido ───────────────────────────────────────────────────

def simular_partido(df, elo, home, away, ronda='', n_sims=N_SIMS,
                     stats_home=None, stats_away=None):
    """
    Corre Monte Carlo sobre Poisson(λ_home) vs Poisson(λ_away) y devuelve
    un dict con probabilidades a 90 min, prórroga/penales y marcador más
    probable.

    stats_home / stats_away son opcionales: si vienen de API-Football
    (gf_avg, ga_avg del torneo actual) se usan tal cual; si no, se calculan
    del histórico con team_base_stats (modo bracket_completo.ipynb).
    """
    s_h = stats_home or team_base_stats(df, home)
    s_a = stats_away or team_base_stats(df, away)
    f_h = recent_form(df, home)
    f_a = recent_form(df, away)

    elo_h = elo.get(home, ELO_FALLBACK.get(home, ELO_INIT))
    elo_a = elo.get(away, ELO_FALLBACK.get(away, ELO_INIT))

    lam_h = get_lambda(s_h, s_a, f_h, f_a, elo_h, elo_a,
                        home_factor=host_factor(home),
                        injury_factor=INJURY_FACTOR.get(home, 1.0))
    lam_a = get_lambda(s_a, s_h, f_a, f_h, elo_a, elo_h,
                        home_factor=host_factor(away),
                        injury_factor=INJURY_FACTOR.get(away, 1.0))

    gh = np.random.poisson(lam_h, n_sims)
    ga = np.random.poisson(lam_a, n_sims)

    p_h = np.mean(gh > ga)
    p_d = np.mean(gh == ga)
    p_a = np.mean(gh < ga)

    total = gh + ga
    over25 = np.mean(total > 2.5)
    over35 = np.mean(total > 3.5)
    btts   = np.mean((gh > 0) & (ga > 0))

    # Prórroga y penales — no hay empate posible en rondas eliminatorias
    p_pen_h = 0.51 if elo_h >= elo_a else 0.49
    p_pen_a = 1 - p_pen_h
    extra = p_d * 0.50

    p_h_total = p_h + extra * p_pen_h + extra * (lam_h / (lam_h + lam_a))
    p_a_total = p_a + extra * p_pen_a + extra * (lam_a / (lam_h + lam_a))

    ganador  = home if p_h_total >= p_a_total else away
    perdedor = away if ganador == home else home

    marcador = (pd.Series([f'{h}-{a}' for h, a in zip(gh, ga)])
                  .value_counts(normalize=True).idxmax())

    return {
        'Ronda': ronda,
        'Local': home,
        'Visitante': away,
        'Ganador': ganador,
        'Perdedor': perdedor,
        'P local 90': p_h,
        'P empate 90': p_d,
        'P visita 90': p_a,
        'P pasa local': p_h_total,
        'P pasa visita': p_a_total,
        'Prob ganador': max(p_h_total, p_a_total),
        'Marcador probable': marcador,
        'Over 2.5': over25,
        'Over 3.5': over35,
        'BTTS': btts,
        'xG local': lam_h,
        'xG visita': lam_a,
        'Elo local': elo_h,
        'Elo visita': elo_a,
    }
