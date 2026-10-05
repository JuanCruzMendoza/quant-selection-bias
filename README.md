# Sesgo de selección en estrategias de trading

Para ver los resultados del trabajo ir a: [poster/poster.pdf](poster/poster.pdf)

## Objetivo del trabajo

Al desarrollar una estrategia es habitual probar muchas variantes y quedarse con la de mejor backtest. Aunque cada variante se valide sin leakage, el máximo de muchas estimaciones ruidosas está sesgado hacia arriba: es el sesgo de selección. El objetivo del trabajo entonces es medirlo:

- **Estrategias:** 1116 configuraciones de Random Forest (subconjunto de features, horizonte, cuantil del target, profundidad y mínimo de muestras por hoja) que arman carteras long-short y long-only sobre 40 acciones grandes del S&P 500, con datos diarios de yfinance.
- **Validación sin leakage:** purged K-fold con embargo sobre 2006–2017, y un hold-out (2018–2025) que se evaluó una sola vez, al final.
- **Medición del sobreajuste:** Deflated Sharpe Ratio (DSR), CSCV (PBO y estimación fuera de muestra del procedimiento de elegir la mejor), un placebo sin señal y baselines sin búsqueda (1/N, risk parity y SPY).

**Resultado principal (long-short):** la configuración elegida tenía un Sharpe Ratio de 0.96 en la validación cruzada y de 0.53 en el hold-out, mientras que el promedio de las configuraciones quedó en 0.17 en los dos períodos. El DSR (0.80, no significativo) y la estimación CSCV (0.40) anticiparon la caída usando solo los datos de desarrollo.

## Estructura

| Archivo | Contenido |
|---|---|
| [docs/plan.md](docs/plan.md) | Diseño del experimento, fijado antes de correrlo |
| [run_experiment.ipynb](run_experiment.ipynb) | Implementación completa y reproducible |
| [docs/resultados.md](docs/resultados.md) | Resultados e insights, long-short y long-only |
| [docs/resultados_LS.md](docs/resultados_LS.md) | Los mismos resultados, solo long-short |
| [poster/](poster/) | Póster en LaTeX y el script que arma sus figuras |
| `results/` | Métricas, tablas y figuras de cada corrida |

## Cómo reproducirlo

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Después se corre `run_experiment.ipynb` de arriba abajo. Al principio del notebook están las instrucciones y las variables `QUICK_RUN`, `RUN_HELDOUT` y `RUN_PLACEBO`.

El póster se compila con LuaLaTeX desde `poster/` (`latexmk -lualatex poster.tex`), y sus figuras se regeneran con `.venv/Scripts/python poster/figuras.py`.
