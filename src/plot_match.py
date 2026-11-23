"""Single-match momentum chart: model momentum v scoreboard momentum, plus chase win probability.

Usage:  python src/plot_match.py                      (default: London Spirit v Welsh Fire, 12 Aug 2026, men)
        python src/plot_match.py <match_id> <male|female>
Output: figures/<gender>/momentum_match_<match_id>.png
"""
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.expanduser('~/hundred-momentum')
BLUE, ORANGE, INK, MUTED, GRID, SURF, GREY = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e7e6e2', '#fcfcfb', '#d9d8d3'


def load(g, mid):
    d = pd.read_parquet(f'{ROOT}/data/out/{g}/values.parquet'); d['match_id'] = d.match_id.astype(str)
    ref = d[(d.season == d[d.match_id == mid].season.iloc[0]) & (d.ball_no >= 10)]      # same season, from ball 10 on
    # put both measures on one scale: standard deviations within each innings type
    for col, z in [('mom10_std', 'm_z'), ('scoreboard10', 's_z')]:
        s = ref.groupby('innings')[col].agg(['mean', 'std'])
        d[z] = (d[col] - d.innings.map(s['mean'])) / d.innings.map(s['std'])
    return d[d.match_id == mid].copy()


def plot(g, mid):
    x = load(g, mid); M = pd.read_parquet(f'{ROOT}/data/out/{g}/matches.parquet'); M['match_id'] = M.match_id.astype(str); m = M[M.match_id == mid].iloc[0]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.facecolor': SURF, 'figure.facecolor': SURF})
    fig, ax = plt.subplots(3, 1, figsize=(11, 10.5), gridspec_kw=dict(height_ratios=[1, 1, .8], hspace=.5))
    lo = min(-3.4, x[x.ball_no >= 10][['m_z', 's_z']].min().min() - .6); teams = []

    # panels 1-2: momentum, first innings and chase
    for i, inn in enumerate((1, 2)):
        z = x[x.innings == inn]; b = z.ball_no.values; a = ax[i]; teams.append(z.batting_team.iloc[0]); k = (z.ball_no >= 10).values
        dis = (np.sign(z.m_z) != np.sign(z.s_z)).values & k                                  # the two measures point different ways
        for bb, flag in zip(b, dis):
            if flag: a.axvspan(bb - .5, bb + .5, color=GREY, alpha=.6, lw=0)
        a.axhline(0, color=MUTED, lw=.8)
        a.plot(b[k], z.m_z.values[k], color=BLUE, lw=2); a.plot(b[k], z.s_z.values[k], color=ORANGE, lw=2)
        w = z[z.is_wicket == 1]
        for bw in w.ball_no: a.axvline(bw, color=INK, lw=.6, ls=':', alpha=.5)
        a.scatter(w.ball_no, np.full(len(w), lo + .3), marker='^', s=45, color=INK, zorder=5)
        a.set_ylim(lo, 3.4); a.set_xlim(0, 101); a.grid(axis='y', color=GRID, lw=.6); a.set_axisbelow(True)
        for s in ('top', 'right'): a.spines[s].set_visible(False)
        a.set_ylabel('Last 10 balls (SDs)')
        a.text(1, 3.2, '▲ batting side on top', va='top', color=MUTED, fontsize=8.5); a.text(1, lo + .9, '▼ bowling side on top', va='bottom', color=MUTED, fontsize=8.5)
        a.set_title(f"{'1st innings' if inn == 1 else 'Chase'}: {teams[-1]} {int(z.runs_total.sum())}/{int(z.is_wicket.sum())}   ·   "
                    f"the two measures disagree on {dis[k].mean():.0%} of balls (grey)", loc='left', color=INK, fontsize=11, fontweight='bold')

    # panel 3: chase win probability
    z = x[x.innings == 2]; a = ax[2]
    a.plot(z.ball_no, z.wp_pre * 100, color=BLUE, lw=2); a.axhline(50, color=MUTED, lw=.8, ls='--')
    w = z[z.is_wicket == 1]; a.scatter(w.ball_no, w.wp_pre * 100, marker='^', s=45, color=INK, zorder=5)
    a.set_ylim(0, 100); a.set_xlim(0, 101); a.set_ylabel('Win chance (%)'); a.set_xlabel('Ball of the innings')
    a.grid(axis='y', color=GRID, lw=.6)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
    a.set_title(f'Chase: {teams[1]} win probability before each ball', loc='left', color=INK, fontsize=11, fontweight='bold')

    margin = f"{int(m.by_runs)} runs" if pd.notna(m.by_runs) else f"{int(m.by_wickets)} wickets"
    loser = [t for t in teams if t != m.winner][0]
    fig.suptitle(f"{m.winner} beat {loser} by {margin}  ·  The Hundred ({'men' if g == 'male' else 'women'}), {pd.Timestamp(m.date):%d %b %Y}",
                 x=.06, ha='left', fontsize=13, fontweight='bold', color=INK, y=.985)
    H = [Line2D([], [], color=BLUE, lw=2), Line2D([], [], color=ORANGE, lw=2), Line2D([], [], marker='^', ls='', color=INK, ms=7), plt.Rectangle((0, 0), 1, 1, color=GREY)]
    fig.legend(H, ['Model momentum (over or under what the model expected)', 'Scoreboard momentum (runs − 6 per wicket)', 'Wicket', 'Measures disagree'],
               loc='upper left', bbox_to_anchor=(.055, .955), ncol=2, frameon=False, fontsize=9.5, labelcolor=INK)
    out = f'{ROOT}/figures/{g}/momentum_match_{mid}.png'; fig.savefig(out, dpi=160, bbox_inches='tight'); print('saved', out)


if __name__ == '__main__':
    mid = sys.argv[1] if len(sys.argv) > 1 else '1521261'; g = sys.argv[2] if len(sys.argv) > 2 else 'male'
    plot(g, mid)
