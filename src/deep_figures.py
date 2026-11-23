"""Charts for the deep-analysis pages of the PDF report (exploratory results).

Usage:  python src/deep_figures.py      (after src/deep_analysis.py, src/feature_lab.py, src/v2_sensitivity.py and src/export_dashboard.py)
Output: figures/compare/deep_*.png
"""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.expanduser('~/hundred-momentum'); DB = f'{ROOT}/dashboard'; FIG = f'{ROOT}/figures/compare'
I1, I2, MEN, WOMEN, INK, MUTED, GRID, SURF = '#2f6fd6', '#e8613c', '#007f9e', '#c4850b', '#0b0b0b', '#52514e', '#e7e6e2', '#fcfcfb'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': INK,
                     'axes.facecolor': SURF, 'figure.facecolor': SURF})


def clean(ax):
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.grid(axis='x', color=GRID, lw=.6); ax.set_axisbelow(True); ax.tick_params(length=0)


def forest(ax, labels, series, title, xlabel):
    y = np.arange(len(labels)); off = np.linspace(-.18, .18, len(series)) if len(series) > 1 else [0]
    for (name, col, v, lo, hi), o in zip(series, off):
        for yy, a, b, c in zip(y + o, v, lo, hi):
            if np.isnan(a): continue
            sig = b > 0 or c < 0
            ax.plot([b, c], [yy, yy], color=col, lw=2, alpha=.7, solid_capstyle='round')
            ax.scatter([a], [yy], s=36, color=col if sig else SURF, edgecolor=col, lw=1.6, zorder=3)
    ax._handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=n) for n, c, *_ in series]
    ax.axvline(0, color=INK, lw=1); ax.set_yticks(y, labels); ax.invert_yaxis(); clean(ax)
    ax.set_title(title, loc='left', fontsize=10.5, fontweight='bold', color=INK); ax.set_xlabel(xlabel)


def main():
    EV = pd.read_csv(f'{DB}/deep_events.csv'); KP = pd.read_csv(f'{DB}/deep_bowler_kind_parts.csv'); GP = pd.read_csv(f'{DB}/deep_game_parts.csv')
    LA = pd.read_csv(f'{DB}/lab_ablation.csv')
    # small moments
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    for ax, g in zip(axs, ('Men', 'Women')):
        z = EV[EV.gender == g]; labels = list(dict.fromkeys(z.event))
        ser = []
        for inn, col in (('1st innings', I1), ('Chase', I2)):
            q = z[z.innings == inn].set_index('event').reindex(labels); ser.append((inn, col, q.momentum_after.values, q.lo.values, q.hi.values))
        forest(ax, labels, ser, g, 'Momentum after the moment (SD, beyond expectation)')
    axs[0].legend(handles=axs[0]._handles, loc='lower right', frameon=False, fontsize=8.5)
    fig.suptitle('Small moments that move momentum (filled = interval excludes zero)', x=.01, ha='left', fontsize=12, fontweight='bold', color=INK)
    fig.tight_layout(); fig.savefig(f'{FIG}/deep_events.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    # pace v spin by part
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, g in zip(axs, ('Men', 'Women')):
        for k, col in (('pace', I1), ('spin', I2)):
            z = KP[(KP.gender == g) & (KP.kind == k)].sort_values('part'); ax.plot(z.part, z.runs_saved_per_ball, color=col, lw=2.2, marker='o', label=k.title())
        ax.axhline(0, color=MUTED, lw=.8); ax.set_xticks(range(5), ['1–20', '21–40', '41–60', '61–80', '81–100']); ax.grid(axis='y', color=GRID, lw=.6)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        ax.set_title(g, loc='left', fontsize=10.5, fontweight='bold', color=INK); ax.set_xlabel('Balls of the innings')
    axs[0].set_ylabel('Runs saved per ball v expected'); axs[0].legend(frameon=False)
    fig.suptitle('Pace or spin, part by part of the innings', x=.01, ha='left', fontsize=12, fontweight='bold', color=INK)
    fig.tight_layout(); fig.savefig(f'{FIG}/deep_spin_parts.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    # where matches are won
    fig, axs = plt.subplots(1, 2, figsize=(11, 2.6))
    parts = ['Balls 1–20', 'Balls 21–40', 'Balls 41–60', 'Balls 61–80', 'Balls 81–100']
    for ax, g, col in zip(axs, ('Men', 'Women'), (MEN, WOMEN)):
        M = GP[GP.gender == g].pivot(index='innings', columns='part', values='corr_with_winning')[parts]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('c', [SURF, col]); ax.imshow(M.values, cmap=cmap, vmin=.1, vmax=.6, aspect='auto')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]): ax.text(j, i, f'{M.values[i, j]:.2f}', ha='center', va='center', fontsize=10, fontweight='bold', color='white' if M.values[i, j] > .42 else INK)
        ax.set_xticks(range(5), [p.replace('Balls ', '') for p in parts]); ax.set_yticks(range(2), M.index); ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(g, loc='left', fontsize=10.5, fontweight='bold', color=INK)
    fig.suptitle('Where matches are won: correlation of each part\'s momentum with winning', x=.01, ha='left', fontsize=12, fontweight='bold', color=INK)
    fig.tight_layout(); fig.savefig(f'{FIG}/deep_game_parts.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    # what new information adds
    fig, ax = plt.subplots(figsize=(9, 3.4)); labels = list(dict.fromkeys(LA.added.str.replace('Add ', '')))
    nice = lambda l: l[0].upper() + l[1:]
    ser = []
    for g, col in (('Men', MEN), ('Women', WOMEN)):
        q = LA[LA.gender == g].assign(l=lambda t: t.added.str.replace('Add ', '')).set_index('l').reindex(labels); ser.append((g, col, q['diff'].values * 1000, q.lo.values * 1000, q.hi.values * 1000))
    forest(ax, [nice(l) for l in labels], ser, 'What new information adds to the predictions', 'Change in log-loss, thousandths (left of zero = better)'); ax.legend(handles=ax._handles, loc='lower right', frameon=False)
    fig.tight_layout(); fig.savefig(f'{FIG}/deep_lab.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    print('saved deep figures')


if __name__ == '__main__':
    main()
