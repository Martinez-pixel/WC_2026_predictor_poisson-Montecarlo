# World Cup 2026 Predictor

Modelo de predicción para partidos y bracket del Mundial 2026, basado en ratings Elo,
forma reciente y simulación Monte Carlo sobre un proceso de Poisson.

Dos formas de usarlo:

- **Predicción de un partido específico** con datos en vivo del torneo (API-Football): estadísticas
  de goles, forma, alineaciones y bajas confirmadas.
- **Simulación del bracket completo restante**, sin depender de API: solo con Elo histórico y forma
  reciente, corre desde ronda de 32 hasta la final.

## Cómo funciona el modelo

Para cada partido se calcula un λ (goles esperados) por equipo, mezclando tres fuentes:

| Fuente | Peso | Qué captura |
|---|---|---|
| Stats del torneo actual (API-Football) | 50% | Forma más reciente y directa, específica del Mundial |
| Forma histórica ponderada (últimos 10 partidos, decay 0.85) | 35% | Momentum reciente fuera del torneo |
| Ratio de Elo | 15% | Nivel de largo plazo, estabiliza cuando hay poca muestra |

Con los dos λ (local y visitante) se corren simulaciones de Poisson (30k–100k según el notebook)
para sacar la distribución completa de resultados — no solo el resultado más probable, también
Over/Under, BTTS y marcadores exactos.

**Localía real** (factor 1.08–1.12) solo se aplica a **Estados Unidos, México y Canadá**, sean local
o visitante en la ficha del partido — son los únicos con ventaja de anfitrión real en este Mundial.
El resto de los partidos usa un factor nominal mucho menor o ninguno.

**Rondas eliminatorias:** no hay empate posible, así que el modelo separa 90 minutos, prórroga y
penales. En penales la ventaja se reduce a casi 50/50 con un ligero ajuste por el equipo con mejor
Elo.

## Estructura del repo

```
wc_predictor.py          # motor: get_lambda, simular_partido, build_elo, recent_form
partido_individual.ipynb # demo: predicción de un partido puntual con datos de API-Football
bracket_completo.ipynb   # demo: simulación de todo el bracket restante, sin API
resultados_reales.csv    # partidos ya jugados, se actualiza conforme avanza el torneo
requirements.txt
```

## Datos

- **Histórico de resultados internacionales:** [martj42/international_results](https://github.com/martj42/international_results)
- **Datos en vivo del torneo:** [API-Football](https://www.api-football.com/)

## Instalación

```bash
git clone https://github.com/<tu-usuario>/world_cup_predictions_2026
cd world_cup_predictions_2026
pip install -r requirements.txt
```

Para usar el notebook de partido individual necesitas una API key de API-Football (hay plan gratuito
con límite de requests por día). El notebook de bracket completo no la necesita.

## Uso rápido

```python
from wc_predictor import simular_partido, build_elo
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/martj42/international_results/master/results.csv',
                  parse_dates=['date'])
elo = build_elo(df)

resultado = simular_partido('Spain', 'Austria', ronda='R32')
print(resultado)
```

## Supuestos y limitaciones

- El Elo se calcula desde 2000 en adelante, con K variable por tipo de torneo (Mundial > Copas
  continentales > eliminatorias > amistosos).
- Con pocos o ningún head-to-head directo, el peso de esa señal es bajo (10%) a propósito — no
  queremos que 1-2 partidos de hace años sesguen la predicción.
- Las alineaciones y bajas de API-Football solo están disponibles ~1 hora antes del kickoff; si se
  corre antes, el modelo sigue funcionando solo con stats del torneo.
- Es un modelo de apoyo, no una garantía — Poisson asume goles independientes entre sí, que es una
  simplificación razonable pero no perfecta del fútbol real.

## Créditos

Este proyecto arrancó a partir de la idea de [mar-antaya/world_cup_predictions](https://github.com/mar-antaya/world_cup_predictions),
de donde también tomé el dataset histórico de martj42. A partir de ahí el enfoque se fue por otro
lado: en vez de un clasificador XGBoost por partido, este modelo usa Poisson con simulación Monte
Carlo, integra API-Football para datos en vivo del torneo, y simula el bracket eliminatorio completo
(Elo, forma reciente, prórroga y penales incluidos).

## Licencia

MIT
