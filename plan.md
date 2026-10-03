# Spec: TP Final, overfitting por selección

## Tesis

La purged K-fold con embargo resuelve el **leakage**, pero no resuelve el **sesgo de selección**. Si probás muchas configuraciones validadas correctamente y te quedás con la mejor, igual terminás con un falso positivo. El TP lo muestra usando el pipeline del Ej 2 como laboratorio y compara la estrategia elegida contra baselines simples que no requieren búsqueda.

## Datos

- 40 acciones grandes del S&P 500, con datos **diarios** de yfinance (`auto_adjust=False`), 2005–2025:
  - `Adj Close` (ajustado por splits y dividendos) para los retornos. Los retornos mensuales se calculan a partir de los diarios.
  - `Close` (ajustado solo por splits) × `Volume` para el volumen en dólares. Con el precio ajustado por dividendos, el volumen histórico queda subestimado, sobre todo en las acciones que pagan muchos dividendos.
- **Universo, elegido con información disponible al inicio:**
  1. Candidatas: las acciones que integraban el S&P 500 al 1 de enero de 2005, según el archivo `S&P 500 Historical Components & Changes (Updated).csv` del repo [fja05680/sp500](https://github.com/fja05680/sp500). El archivo registra los integrantes del índice en cada fecha desde 1996, y el repo trae un notebook de ejemplo (`sp500_by_date.ipynb`) para sacar la lista de un día dado.
  2. Se filtran las que tienen historia completa en yfinance entre 2005 y 2025. El dataset usa el ticker que tenía cada empresa en esa fecha y no vincula los cambios de símbolo. Las candidatas que yfinance no encuentra se revisan una por una, para no descartar por error empresas que siguen cotizando con otro ticker.
  3. Se eligen las 40 con mayor volumen diario promedio en dólares durante 2005, que es el mismo proxy de tamaño que usan las features.
- El universo queda fijo durante todo el período y el panel está balanceado. 2005 se usa solo para elegir el universo y como historia para las features, así que la primera observación usable es de enero de 2006.
- SPY como referencia de mercado.
- **Tasa libre de riesgo:** T-bill a 3 meses, serie TB3MS de FRED (se baja igual que en el Ej 1, cambiando el `id=`). Viene como tasa anual en %, así que la tasa mensual es TB3MS / 1200.
- **Train:** enero 2006 – diciembre 2017 (144 meses). **Hold-out:** enero 2018 – diciembre 2025 (96 meses). El hold-out se usa una sola vez, al final.

## Pipeline base

### Convención temporal

- La muestra (y la cartera) del mes $m$ se arma al **inicio** de $m$, con datos hasta el cierre de $m-1$.
- Su etiqueta mira los retornos de los meses $m, \dots, m+h-1$.

### Features de la muestra del mes $m$

| Feature | Cálculo |
|---|---|
| Momentum 12-1 | retorno acumulado de $m-12$ a $m-2$ (se saltea el último mes) |
| Momentum 3m | retorno acumulado de $m-3$ a $m-1$ |
| Reversión 1m | retorno de $m-1$ |
| Volatilidad 6m | desvío estándar de los retornos diarios de $m-6$ a $m-1$ (unos 126 días) |
| Tamaño | volumen diario en dólares promedio de $m-12$ a $m-1$ |

- El tamaño es un proxy: yfinance no tiene capitalización histórica, y multiplicar el precio por las acciones en circulación de hoy mete look-ahead.
- El lookback máximo es de 12 meses (momentum 12-1 y tamaño), y es lo que define el largo del embargo.

### Target y modelo

- **Target:** 1 si el retorno de $m, \dots, m+h-1$ está entre los $\lfloor 40/q \rfloor$ mejores del corte transversal de ese mes: 13 acciones con terciles ($q = 3$) y 8 con quintiles ($q = 5$).
- **Modelo:** Random Forest sobre el panel acción-mes, con los mismos parámetros fijos en todas las configuraciones. Cambiarlos después de ver resultados sería un intento escondido.
  - `n_estimators=200`, `max_features="sqrt"`, `bootstrap=True`.
  - `class_weight=None`: solo importa el orden de las probabilidades, y el AUC no depende del umbral.
  - `random_state` fijo. Probar otras semillas también sería un intento escondido.
  - `max_samples = 1/h`: con $h > 1$, las etiquetas de una misma acción en meses consecutivos se solapan y las muestras son redundantes (§4 de la clase). Limitar el bootstrap a la unicidad promedio ($\approx 1/h$) reduce la correlación entre árboles. No se usan pesos por unicidad porque, con un mismo $h$, son prácticamente iguales para todas las muestras.

### Portafolios

- Se rebalancea cada $h$ meses, sin solapamiento (hay una sola cartera viva a la vez), pero el retorno de la cartera se **registra todos los meses**. Así todas las configuraciones producen una serie mensual con el mismo $T$, sin importar $h$. Esto permite comparar los Sharpes entre configuraciones, usar un único $T$ en el DSR y armar la matriz configuraciones × meses de la PBO.
- El ciclo de rebalanceo arranca en enero de 2006 (en enero de 2018 en el hold-out) y no se prueban otros meses de arranque, porque eso sería otro intento escondido. Como 144 y 96 son múltiplos de 6, los ciclos de todos los $h$ cierran exactamente en diciembre de 2017 y de 2025, así que ninguna tenencia del train cruza al hold-out.
- **Patas:** se usa el mismo cuantil que en el target (13 acciones con terciles, 8 con quintiles). Las acciones se eligen por probabilidad predicha y entran con pesos iguales al rebalancear. Los empates de probabilidad en el borde de una pata se resuelven al azar, con semilla fija.
  - **Long-short (LS):** long en las de mayor probabilidad y short en las de menor. El retorno del mes es el de la pata long menos el de la pata short.
  - **Long-only (LO):** solo la pata long.
- **Entre rebalanceos:** las posiciones quedan quietas (buy & hold) y los pesos derivan con los precios.
- **Costos:** 10 bp por lado. En cada rebalanceo se cobra $0.001 \times \sum_i |w_i^{\text{nuevo}} - w_i^{\text{antes}}|$, donde $w^{\text{antes}}$ son los pesos derivados justo antes de rebalancear (en el primer rebalanceo se parte de cero). El turnover que se reporta es el promedio anual de $\sum_i |\Delta w_i|$.
- **Sharpe:** media sobre desvío de los retornos mensuales **netos de costos**.
  - LO y baselines long-only: sobre el exceso respecto de la tasa libre de riesgo.
  - LS y baseline de momentum: retorno directo, porque son autofinanciados.
  - Para el DSR y la PBO se usa el Sharpe mensual sin anualizar. Para reportar se anualiza ($\times\sqrt{12}$).

## Grilla de configuraciones (los N intentos)

| Dimensión | Valores |
|---|---|
| Subconjunto de features | las $2^5 - 1 = 31$ combinaciones |
| Horizonte $h$ | 1, 3, 6 meses |
| Cuantil del target | terciles, quintiles |
| `max_depth` | 2, 4, 8 |
| `min_samples_leaf` | 1, 50 |

- $N = 31 \times 3 \times 2 \times 3 \times 2 = 1116$ configuraciones. `max_samples` no es una dimensión nueva, porque queda fijado por $h$.
- **Grupos (h, cuantil):** 6 grupos de 186 configuraciones cada uno ($31 \times 3 \times 2$). Se usan para comparar los criterios de selección (ver "Selección").
- La selección por Sharpe se hace por separado para LS y para LO. La selección por AUC es la misma para las dos, porque usan las mismas predicciones.

## Validación (dentro de train)

- **Borde train/hold-out:** se excluyen del entrenamiento (en todos los splits de la CV y en el modelo final) y del cálculo del AUC las muestras cuya etiqueta mira retornos de 2018, que son las de los últimos $h-1$ meses de 2017. Si no, la CV entrenaría con retornos del hold-out. Ninguna de esas muestras es un mes de rebalanceo del train, así que el backtest no las necesita.

- **Purged K-fold con $K = 6$:** folds de 24 meses que coinciden con pares de años calendario (2006–07, 2008–09, …, 2016–17). Los bordes de los folds coinciden con los rebalanceos para todo $h$, así que ninguna cartera se arma en un fold y se mantiene en el siguiente.
  - **Purga, antes del fold de test ($h$ meses):** se elimina del train toda muestra cuya ventana de etiqueta se solape con el test. Es leakage directo: la etiqueta de train contendría retornos del test. Con la convención temporal de arriba alcanzaría con $h-1$ meses; el mes extra queda como margen.
  - **Embargo, después del fold de test (12 meses):** se eliminan las muestras de train de los 12 meses siguientes al test, porque sus features miran 12 meses hacia atrás (momentum 12-1 y tamaño) y estarían construidas con datos del test. El embargo cubre también el solapamiento de etiquetas después del test ($h \le 6 < 12$).
  - Las features de las muestras de test sí pueden mirar hacia el train, porque en tiempo real siempre se tiene el pasado.
  - **Costo:** la purga, el embargo y el borde sacan hasta un 20% de los 120 meses de entrenamiento de cada split, según el split y $h$. El costo es el mismo para todas las configuraciones, así que no sesga la comparación entre ellas.
- **Serie OOS:** cada mes se predice con el modelo del split en el que ese mes es test. Con esas predicciones se arma el backtest de los 144 meses.
- Para cada configuración se guardan dos scores:
  - **Sharpe CV:** el Sharpe de la serie OOS, con la definición de "Portafolios".
  - **AUC CV:** el AUC de todas las predicciones out-of-fold juntas (todos los meses, no solo los de rebalanceo), sin las muestras del borde.
- **Advertencia:** el purged K-fold entrena también con folds posteriores al de test. Por ejemplo, para predecir 2008–09 usa datos de 2010–2017. La purga elimina el solapamiento de etiquetas, pero no impide aprender de regímenes futuros, así que la serie OOS de train es algo optimista. No sesga la comparación entre configuraciones, porque todas usan los mismos folds, pero hay que tenerlo en cuenta al leer la caída en el hold-out (ver "Degradación en el hold-out").

## Protocolo del hold-out

- La configuración elegida se **reentrena una sola vez** con todo el train (sin las muestras del borde) y predice 2018–2025 con el modelo fijo, sin walk-forward.
- Para el Spearman y para la comparación contra el promedio se corren también las 1116 configuraciones en el hold-out, con el mismo protocolo. Esos resultados se usan **solo para medir**. No se usan para elegir ni para cambiar la selección, que queda fijada antes de mirar el hold-out.

## Selección: dos criterios a comparar

- **Selección principal (la de la tesis):** la mejor por Sharpe CV entre las 1116 configuraciones, por separado para LS y para LO. Es el criterio natural para el DSR y la PBO, porque optimiza directamente la métrica que se reporta.
- **Comparación de criterios, dentro de cada grupo (h, cuantil):** en cada uno de los 6 grupos (186 configuraciones) se elige la mejor por Sharpe CV y la mejor por AUC CV.
  - El AUC se compara solo dentro de un grupo porque predecir el tercil superior a 1 mes y el quintil superior a 6 meses son problemas distintos, con AUCs que no se pueden comparar.
  - Para que la comparación sea justa, la selección por Sharpe de esta parte también se hace dentro de cada grupo.
  - Da 12 comparaciones pareadas (6 grupos × LS/LO) en vez de una sola.

**Pregunta:** ¿qué criterio se degrada más? Se mide con la caída del Sharpe entre la CV y el hold-out, y con la PBO de cada criterio.

## Medición del overfitting

**Deflated Sharpe Ratio** de la configuración elegida:

$$
SR_0 = \sqrt{V[\widehat{SR}_n]}\left((1-\gamma)\,\Phi^{-1}\!\left(1-\tfrac{1}{N}\right) + \gamma\,\Phi^{-1}\!\left(1-\tfrac{1}{Ne}\right)\right)
$$

$$
DSR = \Phi\!\left(\frac{(\widehat{SR} - SR_0)\sqrt{T-1}}{\sqrt{1 - \hat\gamma_3\,\widehat{SR} + \frac{\hat\gamma_4 - 1}{4}\,\widehat{SR}^2}}\right)
$$

donde $\gamma \approx 0.5772$ es la constante de Euler-Mascheroni y $\hat\gamma_3, \hat\gamma_4$ son la asimetría y la curtosis de los retornos.

- **Qué Sharpe entra:** el Sharpe CV de la elegida (train), mensual y sin anualizar, con $T = 144$.
- **Qué N y qué varianza:** N es la cantidad de configuraciones entre las que se eligió, y $V[\widehat{SR}_n]$ es la varianza de sus Sharpe CV. En la selección principal son las 1116 de LS (o las de LO); en la comparación de criterios, las 186 del grupo.
- **Curtosis:** $\hat\gamma_4$ es la curtosis cruda (vale 3 para una normal). `scipy.stats.kurtosis` devuelve el exceso por defecto, así que hay que usar `fisher=False`.
- **Autocorrelación:** con $h = 3$ o $6$, los retornos mensuales de una cartera que se mantiene quieta pueden estar autocorrelacionados. Se chequea y, si es significativa, se usa un $T$ efectivo, $T_{ef} \approx T / (1 + 2\sum_k \rho_k)$, en la línea de Lo (2002). La autocorrelación también sesga la anualización con $\sqrt{12}$.
- **En el hold-out** hay un solo intento, así que no se calcula el DSR. Se reporta el PSR, que es el mismo estadístico con $SR_0 = 0$.
- **Baselines:** son un solo intento ($N = 1$), así que $SR_0 = 0$ y su DSR coincide con el PSR.
- El DSR está definido para selección por Sharpe. Para la selección por AUC se reporta igual, como referencia, pero aclarando que está mal especificado (ver la última sección).
- Como las configuraciones están muy correlacionadas, el N efectivo es menor que N. Se puede estimar agrupando las series de retornos en clusters y reportar el DSR con los dos valores.

**PBO** (CSCV) sobre la matriz de 144 meses (enero 2006 – diciembre 2017) × configuraciones, armada con los retornos OOS de la purged CV. Mide qué fracción de las veces la campeona in-sample cae debajo de la mediana out-of-sample.
- **Bloques:** $S = 12$ bloques de 12 meses (un año calendario cada uno), lo que da $\binom{12}{6} = 924$ combinaciones IS/OOS. Como el largo de los bloques es múltiplo de 6, sus bordes coinciden con los rebalanceos para todo $h \in \{1, 3, 6\}$. Ningún bloque corta un período de tenencia, así que no hay retornos de la misma cartera a los dos lados de un borde.
- **No se reentrena nada:** la CSCV solo toma subconjuntos de filas de la matriz y calcula Sharpes, sin ajustar modelos. Por eso no necesita purga ni embargo, porque el leakage del modelo ya se controló al generar los retornos.
- **Solo usa el train:** el hold-out no se usa para la PBO. Como robustez opcional, se puede calcular también sobre la matriz del hold-out (96 meses), pero con bloques chicos y ruidosos.
- **PBO por criterio:** para la comparación de criterios se calcula una PBO por grupo y por criterio. En cada combinación, la campeona se elige en los bloques IS por Sharpe o por AUC, y en los dos casos se evalúa por su posición de Sharpe en los bloques OOS.
  - El AUC de un bloque se calcula solo con las muestras cuya ventana de etiqueta cae entera dentro del bloque, para que ninguna etiqueta IS mire retornos de un bloque OOS.
  - Con bloques anuales quedan entre ~55 y ~155 etiquetas positivas por bloque, según $h$ y el cuantil. Alcanza para un AUC estable.
- **Chequeo:** la PBO del placebo debería dar alrededor de 50%.

**Degradación en el hold-out:**
- Sharpe de la elegida en train comparado con el hold-out.
- Correlación de Spearman entre el ranking de CV y el ranking en hold-out, sobre todas las configuraciones, por separado para LS y LO. La hipótesis es que da cerca de 0.
- **La elegida contra el promedio de todas las configuraciones.** La caída del Sharpe entre la CV y el hold-out mezcla tres efectos: el sesgo de selección, el optimismo de la purged CV y el cambio de régimen después de 2018. Los dos últimos afectan a todas las configuraciones por igual, así que la caída extra de la elegida respecto del promedio es la parte atribuible a la selección.

**Placebo:**
- Se arma una sola vez un panel permutado. En cada mes de 2005–2017, el bloque completo de datos del mes de cada acción (retornos diarios y volumen) se reasigna a otra acción mediante una permutación aleatoria, independiente mes a mes.
- Sobre ese panel se recalcula **todo**: features, etiquetas y backtest. Así se obtiene un único "mundo sin señal", consistente para todas las configuraciones (todos los $h$ y cuantiles).
- Se conservan el mercado (el retorno promedio de cada mes) y la dispersión entre acciones, y se destruye cualquier relación entre el pasado de una acción y su futuro.
- Se corre la grilla completa con la purged CV y se registran el Sharpe CV máximo (global, para LS y para LO) y el AUC CV máximo de cada grupo (h, cuantil).
- Se repite 10 veces, cada vez con otra permutación.

**Curva del máximo Sharpe en función de N:**
- Para cada $n \in \{1, 2, 5, 10, 20, 50, 100, 200, 500, 1116\}$ se sortean 1000 subconjuntos de $n$ configuraciones y se promedia su Sharpe CV máximo.
- Se hace con los datos reales y con el placebo, y se superpone $SR_0(n)$ calculado con la varianza de los Sharpes del placebo.
- Si la curva del placebo queda por debajo de la fórmula, es porque las configuraciones están correlacionadas y el N efectivo es menor que n.

## Baselines (un solo intento cada uno, sin búsqueda)

| Baseline                                        | Compara contra |
| ----------------------------------------------- | -------------- |
| 1/N (pesos iguales en las 40 acciones)          | RF long-only   |
| Risk parity inverse-vol (volatilidad de 6m)     | RF long-only   |
| SPY buy & hold                                  | referencia     |
| Momentum 12-1 por terciles (regla fija, sin ML) | RF long-short  |

- Se rebalancean **trimestralmente** desde enero de 2006 (desde enero de 2018 en el hold-out), y eso queda fijado de antemano. El $h$ del RF cambia según la configuración, y elegir el rebalanceo de los baselines después de ver resultados sería otro intento escondido.
- La volatilidad del risk parity es la misma que la de la feature (retornos diarios de los últimos 6 meses), con pesos proporcionales a $1/\sigma_i$.
- Usan los mismos costos, la misma regla buy & hold entre rebalanceos y la misma definición de Sharpe que el RF.
- Su Sharpe en train se calcula sobre los mismos 144 meses que la serie OOS de la CV.

## Entregables

1. Curva del máximo Sharpe CV en función de N: datos reales, placebo y $SR_0$ teórico.
2. Tabla principal para la elegida por Sharpe (LS y LO), los baselines y, como referencia, el Sharpe promedio de las 1116 configuraciones. Columnas: Sharpe en train y en hold-out (anualizados), DSR (train), PSR (hold-out), PBO (solo para los procedimientos de selección), máxima caída, turnover y, para los long-only, information ratio contra 1/N.
3. Tabla de comparación de criterios: por grupo (h, cuantil) y por LS/LO, la elegida por Sharpe contra la elegida por AUC, con Sharpe CV, Sharpe en hold-out, caída y PBO de cada criterio.
4. Scatter del ranking de CV contra el ranking en hold-out, con su Spearman.
5. Retornos acumulados en el hold-out: RF elegido contra baselines.
6. Distribución del Sharpe CV máximo en las 10 repeticiones del placebo, con el valor real marcado.

## Cómputo

- Una corrida de la grilla son $1116 \times 6 \approx 6{,}700$ entrenamientos de RF. El hold-out suma 1116 y el placebo, $10 \times 6{,}700$. En total son unos 75,000.
- Cada entrenamiento sirve para LS, para LO y para los dos criterios de selección, porque todos usan las mismas predicciones.
- Con 200 árboles y `n_jobs=-1` debería llevar del orden de horas. Si no alcanza, se bajan las repeticiones del placebo a 5.

## Limitaciones a mencionar

- **Sesgo de supervivencia (reducido):** el universo se elige por tamaño en 2005, así que no se seleccionan ganadoras mirando el resultado y entran empresas que después se achicaron (GE, Citi, AIG). Pero quedan afuera las que dejaron de cotizar entre 2005 y 2025 (quiebras como Lehman, o empresas adquiridas), porque yfinance no tiene sus datos. Lo mismo pasa con las que cotizaron todo el período pero dejaron de cotizar antes de la descarga (por ejemplo, EA, que pasó a ser privada en agosto de 2026). Eso sesga un poco los resultados hacia arriba. Para la tesis afecta poco, porque el sesgo de selección se mide en términos relativos (la elegida contra el promedio y contra el placebo).
- **Hold-out corto:** son unos 96 retornos mensuales, pero con $h = 3$ o $h = 6$ hay solo unas 32 o 16 decisiones de cartera independientes. El error estándar del Sharpe es de aproximadamente $\sqrt{(1 + SR^2/2)/T}$, así que el hold-out por sí solo discrimina poco.
- **N efectivo:** las configuraciones no son independientes.
- **Placebo con pocas repeticiones:** con 10 repeticiones, la distribución del máximo bajo H₀ queda estimada con pocos puntos.

## Notas sobre la selección por AUC

- **No se puede comparar entre targets distintos:** por eso se elige dentro de cada grupo (h, cuantil).
- **DSR:** la corrección asume que el Sharpe reportado es el máximo de N intentos. Si se elige por AUC, el Sharpe de la elegida está menos inflado, en una magnitud que no se conoce.
- **El AUC no es plata:** el AUC pondera todo el orden de las observaciones, mientras que la estrategia opera solo los extremos y gana según la magnitud de los retornos.
- **La PBO sí se puede calcular:** con bloques anuales, el AUC por bloque es estable (ver "PBO por criterio").
