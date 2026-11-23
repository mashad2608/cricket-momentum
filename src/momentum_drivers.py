"""What moves momentum, first innings v chase (EXPLORATORY: added after the frozen plan).

Two views, men and women, on the out-of-fold ball values (2022-2026):
  A. WHAT MOVES IT: 10-ball momentum = sum of each ball's standardised gap. Split that sum by the ball's outcome
     (wicket, dot, 1-3 runs, four, six). Share of momentum swings from outcome c = cov(M_c, M) / var(M); shares add up to 1.
     Also the average gap of one ball of each outcome, in each innings.
  B. WHAT THE MODEL WEIGHS: SHAP on the LightGBM half of the final model, turned into runs (as in explain_match.py)
     and grouped into cricket factors. Mean |contribution| per ball, first innings v chase.

Usage:  python src/momentum_drivers.py
Output: data/out/compare/momentum_drivers.json, figures/compare/momentum_drivers_outcomes.png, figures/compare/momentum_drivers_factors.png
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib, shap, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.expanduser('~/hundred-momentum'); sys.path.insert(0, ROOT + '/src')
from pipeline import full
from explain_match import group_of, GROUPS
BLUE, ORANGE, INK, MUTED, GRID, SURF = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e7e6e2', '#fcfcfb'
OUTCOMES = ['Wicket', 'Dot ball', '1–3 runs', 'Four', 'Six']
INN = {1: '1st innings', 2: 'Chase'}

LABEL = {
    'ball_no': 'Ball of the innings', 'balls_left': 'Balls left', 'ball_in_set': 'Ball within the set', 'powerplay': 'Powerplay', 'free_hit': 'Free hit',
    'innings2': 'Chasing (yes/no)', 'score_pre': 'Score so far', 'wkts_pre': 'Wickets down', 'wkts_per_ball_left': 'Wickets in hand per ball left',
    'runs_needed': 'Runs needed', 'req_rate': 'Required rate', 'rate_gap': 'Required minus current rate', 'chase_target': 'Target',
    'bat_balls': 'Batter: balls faced this innings', 'bat_runs': 'Batter: runs this innings', 'bat_sr10': 'Batter: runs per ball, last 10 faced',
    'bat_dots10': 'Batter: dots, last 10 faced', 'set_batter': 'Batter is set', 'new_batter': 'New batter', 'part_balls': 'Partnership: balls',
    'part_runs': 'Partnership: runs', 'since_wicket': 'Balls since last wicket', 'form_vs_hist': 'Batter: form v own history',
    'bat_hist_runs': 'Batter history: runs per ball', 'bat_hist_bnd': 'Batter history: boundary rate', 'bat_death_hist_bnd': 'Batter history: death boundary rate',
    'bat_death_hist_dot': 'Batter history: death dot rate', 'batph_hist_balls': 'Batter history in this phase: balls', 'batph_hist_bnd': 'Batter history in this phase: boundary rate',
    'batph_hist_dot': 'Batter history in this phase: dot rate', 'batph_hist_wkt': 'Batter history in this phase: dismissal rate',
    'bowl_balls': 'Bowler: balls this innings', 'bowl_wkts': 'Bowler: wickets this innings', 'bowl_dots10': 'Bowler: dots, last 10 bowled',
    'bowler_set_no': 'Bowler: which set of theirs', 'consecutive_set': 'Bowler: back-to-back sets', 'bowl_hist_runs': 'Bowler history: runs per ball',
    'bowl_hist_bnd': 'Bowler history: boundary rate', 'bowl_hist_dot': 'Bowler history: dot rate', 'bowlph_hist_bnd': 'Bowler history in this phase: boundary rate',
    'bowlph_hist_dot': 'Bowler history in this phase: dot rate', 'bowlph_hist_wkt': 'Bowler history in this phase: wicket rate',
    'matchup_runs': 'Batter v bowler: expected runs', 'matchup_wkt': 'Batter v bowler: expected wicket rate',
    'last5_runs': 'Team: runs, last 5 balls', 'last5_bnd': 'Team: boundaries, last 5', 'last5_dot': 'Team: dots, last 5', 'last5_wkt': 'Team: wickets, last 5',
    'last10_runs': 'Team: runs, last 10 balls', 'last10_dot': 'Team: dots, last 10', 'last10_wkt': 'Team: wickets, last 10',
    'last20_bnd': 'Team: boundaries, last 20', 'last20_dot': 'Team: dots, last 20', 'last20_wkt': 'Team: wickets, last 20', 'prev_runs': 'Runs off the previous ball',
    'venue_hist_bnd': 'Venue: boundary rate', 'venue_hist_dot': 'Venue: dot rate', 'venue_hist_wkt': 'Venue: wicket rate',
    'season_hist_runs': 'Season: runs per ball', 'season_hist_dot': 'Season: dot rate', 'season_hist_wkt': 'Season: wicket rate',
}


def outcome(d):
    return np.select([d.is_wicket == 1, d.runs_bat >= 6, d.runs_bat == 4, d.runs_total == 0], ['Wicket', 'Six', 'Four', 'Dot ball'], '1–3 runs')


def outcome_shares(d, w=10):
    d = d.sort_values(['match_id', 'innings', 'seq']).copy(); d['oc'] = outcome(d); res = {}
    for inn in (1, 2):
        z = d[d.innings == inn]; grp = z.groupby(['match_id', 'innings'], sort=False)
        M = grp.gap_std.transform(lambda s: s.rolling(w).sum()); ok = M.notna()
        sh, avg = {}, {}
        for c in OUTCOMES:
            Mc = (z.gap_std * (z.oc == c)).groupby([z.match_id, z.innings], sort=False).transform(lambda s: s.rolling(w).sum())
            sh[c] = float(np.cov(Mc[ok], M[ok])[0, 1] / M[ok].var())
            avg[c] = float(z.gap_std[z.oc == c].mean())
        res[INN[inn]] = dict(share=sh, mean_gap=avg, freq={c: float((z.oc == c).mean()) for c in OUTCOMES}, n_balls=int(ok.sum()))
    return res


def factor_importance(g, d, n=6000, seed=0):
    art = joblib.load(f'{ROOT}/data/out/{g}/final_model.pkl'); F = art['feats']; cm = art['cm']; lg = art['model'].get('lgb') or art['model'].get('lgb_shap')
    ex = shap.TreeExplainer(lg); res, feat = {}, {}
    for inn in (1, 2):
        z = d[d.innings == inn]; z = z.sample(min(n, len(z)), random_state=seed); X = z[F].astype(float)
        P = full(lg.predict_proba(X), lg.classes_); E = P @ cm
        sv = ex.shap_values(X); sv = np.stack(sv, -1) if isinstance(sv, list) else sv
        Cf = pd.DataFrame(np.einsum('nfk,nk->nf', sv, P * (cm[None, :] - E[:, None])), columns=F)
        res[INN[inn]] = Cf.T.groupby(group_of).sum().T.abs().mean().to_dict(); feat[INN[inn]] = Cf.abs().mean()
    T = pd.DataFrame(feat); T.index.name = 'feature'
    T.insert(0, 'label', [LABEL.get(f, f) for f in T.index]); T.insert(1, 'factor', [group_of(f) for f in T.index])
    T['rank_1st'] = T['1st innings'].rank(ascending=False, method='min').astype(int); T['rank_chase'] = T['Chase'].rank(ascending=False, method='min').astype(int)
    T['rank_change'] = T.rank_1st - T.rank_chase                                     # positive = more important in the chase
    T = T.sort_values('rank_1st'); T.to_csv(f'{ROOT}/data/out/{g}/feature_ranking_by_innings.csv')
    return res, T


def bars(ax, cats, a, b, title, fmt, labels=('1st innings', 'Chase')):
    y = np.arange(len(cats)); h = .36
    ax.barh(y - h / 2 - .02, a, h, color=BLUE, label=labels[0]); ax.barh(y + h / 2 + .02, b, h, color=ORANGE, label=labels[1])
    for yy, v in zip(y - h / 2 - .02, a): ax.text(v + (max(max(a), max(b)) * .02), yy, fmt(v), va='center', fontsize=8.5, color=INK)
    for yy, v in zip(y + h / 2 + .02, b): ax.text(v + (max(max(a), max(b)) * .02), yy, fmt(v), va='center', fontsize=8.5, color=INK)
    ax.set_yticks(y, cats); ax.set_xlim(0, max(max(a), max(b)) * 1.22)
    ax.grid(axis='x', color=GRID, lw=.6); ax.set_axisbelow(True); ax.tick_params(length=0)
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.set_title(title, loc='left', fontsize=11, fontweight='bold', color=INK)


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
                         'ytick.color': INK, 'axes.facecolor': SURF, 'figure.facecolor': SURF})
    out, RANK = {}, {}
    for g in ('male', 'female'):
        d = pd.read_parquet(f'{ROOT}/data/out/{g}/values.parquet')
        fac, T = factor_importance(g, d); out[g] = dict(outcomes=outcome_shares(d), factors=fac); RANK[g] = T; print(g); print(T.round(4).to_string())
    json.dump(out, open(f'{ROOT}/data/out/compare/momentum_drivers.json', 'w'), indent=1)
    lab = {'male': 'Men', 'female': 'Women'}

    # figure A: share of momentum swings by outcome
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for a, g in zip(ax, ('male', 'female')):
        o = out[g]['outcomes']; bars(a, OUTCOMES, [o['1st innings']['share'][c] * 100 for c in OUTCOMES], [o['Chase']['share'][c] * 100 for c in OUTCOMES], lab[g], lambda v: f'{v:.0f}%')
        a.set_xlabel('Share of 10-ball momentum swings (%)')
    ax[0].invert_yaxis(); h_, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc='upper left', bbox_to_anchor=(.02, .93), ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle('What moves momentum: share of the swings caused by each kind of ball', x=.02, ha='left', fontsize=13, fontweight='bold', color=INK)
    fig.tight_layout(rect=(0, 0, 1, .9)); fig.savefig(f'{ROOT}/figures/compare/momentum_drivers_outcomes.png', dpi=160, bbox_inches='tight'); plt.close(fig)

    # figure B: what the model weighs, by cricket factor
    order = sorted([gname for gname, _ in GROUPS], key=lambda c: -(out['male']['factors']['1st innings'].get(c, 0) + out['male']['factors']['Chase'].get(c, 0)))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for a, g in zip(ax, ('male', 'female')):
        f = out[g]['factors']; bars(a, order, [f['1st innings'].get(c, 0) for c in order], [f['Chase'].get(c, 0) for c in order], lab[g], lambda v: f'{v:.2f}')
        a.set_xlabel('Average effect on expected runs per ball (SHAP, runs)')
    ax[0].invert_yaxis(); h_, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc='upper left', bbox_to_anchor=(.02, .93), ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle('What the model weighs when setting expectations, 1st innings v chase', x=.02, ha='left', fontsize=13, fontweight='bold', color=INK)
    fig.tight_layout(rect=(0, 0, 1, .9)); fig.savefig(f'{ROOT}/figures/compare/momentum_drivers_factors.png', dpi=160, bbox_inches='tight'); plt.close(fig)

    # figure C: every feature ranked, 1st innings v chase (one chart per gender)
    for g in ('male', 'female'):
        T = RANK[g].assign(avg=lambda t: (t['1st innings'] + t['Chase']) / 2).sort_values('avg', ascending=False); n = len(T); y = np.arange(n)
        fig, ax = plt.subplots(figsize=(7.2, .2 * n + 1.2))
        for yy, a_, b_ in zip(y, T['1st innings'], T['Chase']): ax.plot([a_, b_], [yy, yy], color=GRID, lw=2, zorder=1)
        ax.scatter(T['1st innings'], y, s=42, color=BLUE, zorder=3, label='1st innings', edgecolor=SURF, lw=1.2)
        ax.scatter(T['Chase'], y, s=42, color=ORANGE, zorder=3, label='Chase', edgecolor=SURF, lw=1.2)
        ax.set_yticks(y, [f"{r.label}   #{r.rank_1st} / #{r.rank_chase}" for r in T.itertuples()], fontsize=8.5); ax.invert_yaxis()
        ax.set_xlim(0, T[['1st innings', 'Chase']].max().max() * 1.08); ax.set_ylim(n - .4, -.6)
        ax.grid(axis='x', color=GRID, lw=.6); ax.set_axisbelow(True); ax.tick_params(length=0)
        for sp in ('top', 'right', 'left'): ax.spines[sp].set_visible(False)
        ax.set_xlabel('Average effect on expected runs per ball (SHAP, runs)', fontsize=8.5)
        ax.set_title(f"{lab[g]}: rank in 1st innings / chase shown after each name", loc='left', fontsize=9.5, fontweight='bold', color=INK)
        ax.legend(loc='lower right', frameon=False, fontsize=9.5)
        fig.tight_layout(); fig.savefig(f'{ROOT}/figures/compare/feature_ranking_{g}.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    print('saved figures')


if __name__ == '__main__':
    main()
