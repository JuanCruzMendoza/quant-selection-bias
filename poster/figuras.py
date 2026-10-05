"""Figuras del póster (solo long-short), en PDF vectorial.

Lee los resultados que ya guardó run_experiment.ipynb en results/fa3e6998e744_full/ y no recalcula nada: no entrena
modelos ni vuelve a evaluar el hold-out. Usa la Lato de TeX Live, la misma fuente del texto del póster.

Uso, desde la carpeta del repo: .venv/Scripts/python poster/figuras.py
"""
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

POSTER_DIR = Path(__file__).resolve().parent
RESULTS_DIR = POSTER_DIR.parent / 'results' / 'fa3e6998e744_full'
OUT_DIR = POSTER_DIR / 'figuras'
WIDTH = 9.0  # pulgadas: ancho del texto de una columna del póster (A0 vertical, tres columnas)

# Los mismos colores que las figuras del notebook
PERIOD_COLORS = {'val': '#86b6ef', 'ho': '#1c5cab'}
PERIOD_LABELS = {'val': 'Validación (CV, 2006–2017)', 'ho': 'Hold-out (2018–2025)'}
INK = {'primary': '#0b0b0b', 'secondary': '#52514e', 'muted': '#898781', 'grid': '#e1e0d9', 'axis': '#c3c2b7'}
RF_COLOR = '#2a78d6'
LABEL_SIZE = 19  # números rotulados sobre los puntos


def use_lato():
    """Registra en matplotlib la Lato de TeX Live. Si no la encuentra, quedan las fuentes por defecto."""
    try:
        path = subprocess.run(['kpsewhich', 'Lato-Regular.ttf'], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        path = ''
    if not path:
        print('Aviso: no se encontró Lato; se usa la fuente por defecto de matplotlib.')
        return
    for f in Path(path).parent.glob('Lato-*.ttf'):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams['font.family'] = 'Lato'


use_lato()
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'axes.edgecolor': INK['axis'], 'axes.labelcolor': INK['secondary'], 'text.color': INK['primary'],
    'xtick.color': INK['axis'], 'ytick.color': INK['axis'],
    'xtick.labelcolor': INK['secondary'], 'ytick.labelcolor': INK['secondary'],
    'axes.grid': True, 'grid.color': INK['grid'], 'grid.linewidth': 1.0, 'axes.linewidth': 1.2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.size': 6, 'ytick.major.size': 6, 'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'font.size': 21, 'axes.labelsize': 22, 'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 20,
    'lines.linewidth': 3.5, 'legend.frameon': False, 'pdf.fonttype': 42,
})

metrics = pd.read_csv(RESULTS_DIR / 'metrics_real.csv')
selection = pd.read_csv(RESULTS_DIR / 'selection.csv', index_col=0)
pbo = pd.read_csv(RESULTS_DIR / 'pbo.csv', index_col=0)
curves = pd.read_csv(RESULTS_DIR / 'curva_max_sharpe.csv', header=[0, 1], index_col=0)
criteria = pd.read_csv(RESULTS_DIR / 'tabla_criterios.csv')
with open(RESULTS_DIR / 'heldout_analysis.json', encoding='utf-8') as f:
    heldout = json.load(f)

SR_VAL, SR_HO = metrics['sharpe_val_ls'], metrics['sharpe_ho_ls']
N = len(metrics)
CHOSEN = metrics.set_index('id').loc[selection.loc['global_ls', 'id']]  # la mejor por Sharpe Ratio CV
assert np.isclose(CHOSEN['sharpe_val_ls'], SR_VAL.max())
CSCV = pbo.loc['global_ls_sharpe', 'sharpe_oos_campeona']  # Sharpe Ratio OOS promedio de la campeona IS


def period_handles():
    return [plt.Line2D([], [], color=PERIOD_COLORS[p], marker='o', linestyle='none', markersize=13,
                       label=PERIOD_LABELS[p]) for p in ['val', 'ho']]


def dot(ax, x, y, period, offset=13):
    """Punto de un período, con su valor arriba (validación) o abajo (hold-out), a offset puntos del centro."""
    ax.scatter([x], [y], s=320, color=PERIOD_COLORS[period], edgecolor='white', linewidth=2.5, zorder=3)
    above = period == 'val'
    ax.annotate(f'{x:.2f}', (x, y), textcoords='offset points', xytext=(0, offset if above else -offset),
                ha='center', va='bottom' if above else 'top', fontsize=LABEL_SIZE, color=INK['secondary'])


def dumbbell_rows(ax, rows):
    """Una fila por elemento: validación y hold-out unidos por una línea. rows: [(nombre, val, ho)]."""
    ys = np.arange(len(rows))[::-1]
    for y, (_, v_val, v_ho) in zip(ys, rows):
        ax.plot([v_val, v_ho], [y, y], color=INK['axis'], linewidth=4, solid_capstyle='round', zorder=1)
        dot(ax, v_val, y, 'val')
        dot(ax, v_ho, y, 'ho')
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.tick_params(axis='y', length=0, pad=12)
    ax.grid(False, axis='y')
    ax.axvline(0, color=INK['axis'], linewidth=1.5, zorder=0)


def fig1_max_sharpe():
    """Sharpe Ratio CV máximo según la cantidad de configuraciones probadas, contra el máximo esperado por azar."""
    real, chance = curves['real_ls'], curves['teoria_real_ls']['mean']
    n = real.index.to_numpy()
    mean = SR_VAL.mean()
    fig, ax = plt.subplots(figsize=(WIDTH, 7.6), layout='constrained')
    ax.fill_between(n, real['p05'], real['p95'], color=RF_COLOR, alpha=0.15, linewidth=0)
    ax.plot(n, real['mean'], color=RF_COLOR, marker='o', markersize=11, label='Datos reales (banda: p5–p95)')
    ax.plot(n, chance, color=INK['secondary'], linestyle='--', linewidth=3,
            label='Máximo esperado por azar (media + SR₀)')
    ax.axhline(mean, color=INK['muted'], linestyle=':', linewidth=2.5, zorder=0)
    ax.annotate(f'Media de las {N} configuraciones: {mean:.2f}', (n[-1], mean), textcoords='offset points',
                xytext=(0, 9), ha='right', va='bottom', fontsize=LABEL_SIZE, color=INK['secondary'])
    for y, color, dy, va in [(real['mean'].iloc[-1], RF_COLOR, 6, 'bottom'), (chance.iloc[-1], INK['secondary'], -6, 'top')]:
        ax.annotate(f'{y:.2f}', (n[-1], y), textcoords='offset points', xytext=(14, dy), ha='left', va=va,
                    fontsize=LABEL_SIZE, color=color, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks(n, [str(int(v)) for v in n])
    ax.minorticks_off()
    ax.set_xlim(0.75, 2300)
    ax.set_xlabel('Configuraciones probadas (n)')
    ax.set_ylabel('Sharpe Ratio CV máximo (anualizado)')
    fig.legend(loc='outside lower center', ncol=1)
    fig.savefig(OUT_DIR / 'fig1_max_sharpe.pdf')
    plt.close(fig)


def fig2_chosen_vs_all():
    """La RF elegida y la distribución de las 1116 configuraciones, en validación y en hold-out."""
    fig, ax = plt.subplots(figsize=(WIDTH, 5.0), layout='constrained')
    dumbbell_rows(ax, [('RF elegida', CHOSEN['sharpe_val_ls'], CHOSEN['sharpe_ho_ls'])])
    ax.scatter([CSCV], [0], s=300, marker='D', facecolor='white', edgecolor=INK['primary'], linewidth=2.5, zorder=4)
    ax.annotate(f'{CSCV:.2f}', (CSCV, 0), textcoords='offset points', xytext=(0, 13), ha='center', va='bottom',
                fontsize=LABEL_SIZE, color=INK['primary'])
    y_box = -1  # fila de las configuraciones: validación arriba, hold-out abajo
    for period, values, dy in [('val', SR_VAL, 0.2), ('ho', SR_HO, -0.2)]:
        c = PERIOD_COLORS[period]
        ax.boxplot([values], positions=[y_box + dy], vert=False, widths=0.22, whis=(0, 100), showfliers=False,
                   patch_artist=True, manage_ticks=False, zorder=2,
                   boxprops=dict(facecolor=matplotlib.colors.to_rgba(c, 0.25), edgecolor=c, linewidth=2.2),
                   whiskerprops=dict(color=c, linewidth=2.2), capprops=dict(color=c, linewidth=2.2),
                   medianprops=dict(color=c, linewidth=3.5))
        dot(ax, values.mean(), y_box + dy, period, offset=17)  # el rótulo queda afuera de la caja
    ax.set_yticks([0, y_box], ['RF elegida', f'Las {N}\nconfiguraciones'])
    ax.set_ylim(-1.7, 0.6)
    ax.set_xlabel('Sharpe Ratio anualizado')
    cscv = plt.Line2D([], [], marker='D', linestyle='none', markersize=12, markerfacecolor='white',
                      markeredgecolor=INK['primary'], markeredgewidth=2.5, label='Estimación CSCV')
    boxes = matplotlib.patches.Patch(facecolor=matplotlib.colors.to_rgba(INK['muted'], 0.25), edgecolor=INK['muted'],
                                     linewidth=2, label=f'Las {N} (cuartiles y rango)')
    fig.legend(handles=period_handles() + [cscv, boxes], loc='outside lower center', ncol=2)
    fig.savefig(OUT_DIR / 'fig2_elegida_vs_configuraciones.pdf')
    plt.close(fig)


def deciles():
    """Promedio por decil del Sharpe Ratio de validación (1 = el 10% más bajo, 10 = el 10% más alto)."""
    dec = np.ceil(10 * SR_VAL.rank(method='first') / N).astype(int)
    return pd.DataFrame({'val': SR_VAL, 'ho': SR_HO, 'dec': dec}).groupby('dec').mean()


def fig3_deciles():
    """Configuraciones agrupadas en deciles según su Sharpe Ratio de validación; flechas en las puntas."""
    g = deciles()
    fig, ax = plt.subplots(figsize=(WIDTH, 6.6), layout='constrained')
    for period in ['val', 'ho']:
        ax.plot(g.index, g[period], color=PERIOD_COLORS[period], marker='o', markersize=14, markeredgecolor='white',
                markeredgewidth=2.5, zorder=3)
    for d, side in [(1, -1), (10, 1)]:  # en las puntas, de validación a hold-out: vuelven hacia el promedio
        ax.annotate('', xy=(d, g.loc[d, 'ho']), xytext=(d, g.loc[d, 'val']), zorder=2,
                    arrowprops=dict(arrowstyle='-|>', color=INK['secondary'], linewidth=2.5, mutation_scale=26,
                                    shrinkA=12, shrinkB=12))
        for period in ['val', 'ho']:
            ax.annotate(f'{g.loc[d, period]:.2f}', (d, g.loc[d, period]), textcoords='offset points',
                        xytext=(16 * side, 0), ha='left' if side > 0 else 'right', va='center',
                        fontsize=LABEL_SIZE, color=PERIOD_COLORS[period], fontweight='bold')
    ax.text(0.03, 0.97, f"Spearman entre los rankings: {heldout['spearman_ls']['rho']:.2f}", transform=ax.transAxes,
            ha='left', va='top', fontsize=LABEL_SIZE, color=INK['secondary'])
    ax.axhline(0, color=INK['axis'], linewidth=1.5, zorder=0)
    ax.set_xticks(range(1, 11))
    ax.set_xlim(-0.9, 11.3)
    ax.set_xlabel('Decil según el Sharpe Ratio de validación')
    ax.set_ylabel('Sharpe Ratio anualizado\n(promedio del decil)')
    fig.legend(handles=period_handles(), loc='outside lower center', ncol=2)
    fig.savefig(OUT_DIR / 'fig3_deciles.pdf')
    plt.close(fig)


def criteria_means():
    """Promedio sobre los 6 grupos (h, q) de la elegida por Sharpe Ratio y de la elegida por AUC (long-short)."""
    ls = criteria[criteria['side'] == 'LS']
    return {crit: (ls.loc[ls['criterio'] == crit, 'sharpe_val'].mean(), ls.loc[ls['criterio'] == crit, 'sharpe_ho'].mean())
            for crit in ['sharpe', 'auc']}


def fig4_sharpe_vs_auc():
    """Dentro de cada grupo (h, q): la elegida por Sharpe Ratio y la elegida por AUC, promediadas sobre los grupos."""
    means = criteria_means()
    rows = [('Elegida por\nSharpe Ratio', *means['sharpe']), ('Elegida por\nAUC', *means['auc']),
            ('Promedio de las\nconfiguraciones', SR_VAL.mean(), SR_HO.mean())]
    fig, ax = plt.subplots(figsize=(WIDTH, 5.2), layout='constrained')
    dumbbell_rows(ax, rows)
    ax.set_xlim(-0.05, 0.85)
    ax.set_xlabel('Sharpe Ratio anualizado\n(promedio de los 6 grupos)')
    fig.legend(handles=period_handles(), loc='outside lower center', ncol=2)
    fig.savefig(OUT_DIR / 'fig4_sharpe_vs_auc.pdf')
    plt.close(fig)


def summary():
    """Números que cita el póster, para revisarlos contra el texto."""
    auc = metrics.loc[metrics['auc_val'].idxmax()]
    g, means = deciles(), criteria_means()
    sel = selection.loc['global_ls']
    rho = (N - sel['N_eff']) / (N - 1)  # de N_ef = rho + (1 - rho) N
    ls = criteria[criteria['side'] == 'LS'].pivot(index='grupo', columns='criterio', values='sharpe_ho')
    print(f"Elegida por Sharpe Ratio ({CHOSEN.name}): CV {CHOSEN['sharpe_val_ls']:.2f}, "
          f"hold-out {CHOSEN['sharpe_ho_ls']:.2f}, DSR {sel['dsr']:.2f}, "
          f"puesto en hold-out {int(SR_HO.rank(ascending=False)[metrics['id'] == CHOSEN.name].iloc[0])} de {N}")
    print(f"Elegida por AUC ({auc['id']}, AUC {auc['auc_val']:.3f}): CV {auc['sharpe_val_ls']:.2f}, "
          f"hold-out {auc['sharpe_ho_ls']:.2f}")
    print(f'Promedio de las configuraciones: CV {SR_VAL.mean():.2f}, hold-out {SR_HO.mean():.2f}; '
          f'desvío CV {SR_VAL.std():.2f}')
    print(f'Estimación CSCV: {CSCV:.2f}')
    print(f"N efectivo {sel['N_eff']:.0f} (correlación promedio {rho:.2f}), "
          f"SR0 {sel['sr0_m'] * np.sqrt(12):.2f} anual")
    print(f"Curva de azar en n = {N}: {curves['teoria_real_ls']['mean'].iloc[-1]:.2f}")
    print(f"Deciles: 10 {g.loc[10, 'val']:.2f} -> {g.loc[10, 'ho']:.2f}; 1 {g.loc[1, 'val']:.2f} -> {g.loc[1, 'ho']:.2f}; "
          f"Spearman {heldout['spearman_ls']['rho']:.2f}")
    print(f"Por grupo: Sharpe Ratio {means['sharpe'][0]:.2f} -> {means['sharpe'][1]:.2f}, "
          f"AUC {means['auc'][0]:.2f} -> {means['auc'][1]:.2f}; "
          f"la elegida por Sharpe Ratio rinde más en el hold-out en {(ls['sharpe'] > ls['auc']).sum()} de {len(ls)} grupos")


if __name__ == '__main__':
    OUT_DIR.mkdir(exist_ok=True)
    fig1_max_sharpe()
    fig2_chosen_vs_all()
    fig3_deciles()
    fig4_sharpe_vs_auc()
    summary()
