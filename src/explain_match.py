"""Explain the highs and lows of momentum in ONE match.

Momentum = what happened − what the model expected. So each high or low has two halves:
  1. WHAT HAPPENED  – runs, boundaries, dots and wickets against the model's expectation (from its outcome probabilities)
  2. WHY THE MODEL EXPECTED THAT – SHAP on the LightGBM half of the final model, turned into runs and grouped into cricket categories
Plus the captain's decisions in the match (PSM decisions D1 and D3) with the across-all-matches PSM verdict.
PSM itself cannot run on one match (it needs hundreds of comparable situations), so here it is context, not a new estimate.

Usage:  python src/explain_match.py [match_id] [male|female]
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib, shap, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.expanduser('~/hundred-momentum'); sys.path.insert(0, ROOT + '/src')
from pipeline import full
K = ['W', '0', '1', '2', '3', '4', '5p']
BLUE, ORANGE, INK, MUTED, GRID, SURF = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e7e6e2', '#fcfcfb'

# cricket-language groups for the model's features
GROUPS = [
    ('Stage of innings', lambda f: f in {'ball_no', 'balls_left', 'powerplay', 'ball_in_set', 'score_pre', 'innings2', 'crr', 'free_hit'}),
    ('Wickets in hand', lambda f: f in {'wkts_pre', 'wkts_per_ball_left'}),
    ('Chase pressure', lambda f: f in {'runs_needed', 'req_rate', 'rate_gap', 'chase_target', 'pressure_index'}),
    ('Batter set / partnership', lambda f: f in {'bat_balls', 'bat_runs', 'bat_sr', 'new_batter', 'set_batter', 'part_balls', 'part_runs', 'since_wicket'}),
    ('Batter quality (history)', lambda f: f.startswith(('bat_hist', 'batph_hist', 'bat_death_hist')) or f in {'form_vs_hist', 'matchup_runs'}),
    ('Recent scoring flow', lambda f: f.startswith('last') or f in {'prev_runs', 'bat_sr10', 'bat_dots10', 'since_boundary', 'strike_rotation'}),
    ('Bowler (spell and quality)', lambda f: f.startswith(('bowl', 'bowlph')) or f in {'bowler_set_no', 'consecutive_set', 'balls_since_bowler_last', 'matchup_wkt'}),
    ('Venue and season', lambda f: f.startswith(('venue', 'season'))),
]


def group_of(f):
    return next((g for g, rule in GROUPS if rule(f)), 'Other')


def explain(mid='1521261', g='male', block=10, n_each=2):
    o = f'{ROOT}/data/out/{g}'; fdir = f'{ROOT}/figures/{g}'
    d = pd.read_parquet(f'{o}/values.parquet'); d['match_id'] = d.match_id.astype(str); x = d[d.match_id == mid].sort_values(['innings', 'seq']).reset_index(drop=True)
    M = pd.read_parquet(f'{o}/matches.parquet'); M['match_id'] = M.match_id.astype(str); m = M[M.match_id == mid].iloc[0]
    art = joblib.load(f'{o}/final_model.pkl'); F = art['feats']; cm = art['cm']; lg = art['model'].get('lgb') or art['model'].get('lgb_shap')
    hot_cut = json.load(open(f'{o}/results.json'))['hot_cut']
    psm = pd.read_csv(f'{o}/rq3_psm.csv', index_col=0)

    # ---------- SHAP in runs: linearise expected runs around each ball's probabilities
    X = x[F].astype(float); P = full(lg.predict_proba(X), lg.classes_); E = P @ cm
    sv = shap.TreeExplainer(lg).shap_values(X); sv = np.stack(sv, -1) if isinstance(sv, list) else sv     # (balls, features, classes) in log-odds
    w = P * (cm[None, :] - E[:, None])                                                                    # d(expected runs) / d(log-odds of class k)
    contrib = np.einsum('nfk,nk->nf', sv, w)                                                              # runs, per ball and feature
    C = pd.DataFrame(contrib, columns=F).T.groupby(group_of).sum().T                                      # per ball and cricket group
    check = np.corrcoef(C.sum(1), E - E.mean())[0, 1]

    # ---------- blocks of `block` balls; pick the highest and lowest momentum in each innings
    x['blk'] = (x.ball_no - 1) // block
    x['bnd'] = (x.runs_bat >= 4).astype(int); x['dot'] = ((x.runs_total == 0) & (x.is_wicket == 0)).astype(int)
    x['e_bnd'] = x.p_4 + x.p_5p; x['e_dot'] = x.p_0; x['e_w'] = x.p_W
    B = x.groupby(['innings', 'blk']).agg(team=('batting_team', 'first'), b0=('ball_no', 'min'), b1=('ball_no', 'max'), mom=('gap_std', 'sum'), runs=('runs_total', 'sum'), e_runs=('exp_runs', 'sum'),
                                          bnd=('bnd', 'sum'), e_bnd=('e_bnd', 'sum'), dot=('dot', 'sum'), e_dot=('e_dot', 'sum'), w=('is_wicket', 'sum'), e_w=('e_w', 'sum'),
                                          wp0=('wp_pre', 'first'), wp1=('wp_pre', 'last')).reset_index()
    B = B[(B.b1 - B.b0) >= block // 2]
    pick = pd.concat([pd.concat([z.nlargest(n_each, 'mom'), z.nsmallest(n_each, 'mom')]) for _, z in B.groupby('innings')]).sort_values(['innings', 'b0']).reset_index(drop=True)
    pick['label'] = [f"{'1st inns' if r.innings == 1 else 'Chase'} balls {r.b0}–{r.b1}" for r in pick.itertuples()]

    rows = []
    for r in pick.itertuples():
        sel = (x.innings == r.innings) & (x.blk == r.blk); z = x[sel]
        bats = z.groupby('batter').size().sort_values(ascending=False); bowls = z.groupby('bowler').size().sort_values(ascending=False)
        shp = C[sel.values].sum().sort_values()
        rows.append(dict(r._asdict(), batters=', '.join(bats.index[:2]), bowlers=', '.join(bowls.index[:2]), shap=shp))

    # ---------- decisions in the match (the PSM decisions), with the all-matches verdict
    S = x.groupby(['innings', 'set_idx']).agg(team=('batting_team', 'first'), bowler=('bowler', 'first'), runs=('runs_total', 'sum'), legal=('legal', 'sum'),
                                              mom_end=('mom10_std', 'last'), b1=('ball_no', 'max')).reset_index()
    S['next_bowler'] = S.groupby('innings').bowler.shift(-1); S['prev_bowler'] = S.groupby('innings').bowler.shift(1)
    S['next_gap'] = x.groupby(['innings', 'set_idx']).gap_std.sum().groupby('innings').shift(-1).values
    dec = []
    for s in S.itertuples():
        if pd.isna(s.next_bowler) or s.legal < 5: continue
        if s.mom_end >= hot_cut and pd.notna(s.prev_bowler):
            fresh = s.next_bowler not in (s.bowler, s.prev_bowler)
            dec.append(dict(innings=s.innings, after_ball=s.b1, decision='D3 hot spell for the batting side',
                            what=f"{s.team} momentum {s.mom_end:+.1f} SD (≥ {hot_cut:.1f}); captain {'fresh bowler: ' + s.next_bowler if fresh else 'went back to ' + s.next_bowler}",
                            next_set=s.next_gap))
        if s.runs >= 10:
            kept = s.next_bowler == s.bowler
            dec.append(dict(innings=s.innings, after_ball=s.b1, decision='D1 expensive set (10+ runs)',
                            what=f"{s.bowler} went for {s.runs}; captain {'kept the bowler on' if kept else 'changed to ' + s.next_bowler}", next_set=s.next_gap))
    DEC = pd.DataFrame(dec)

    # ---------- figure 1: momentum with the highlighted blocks
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.facecolor': SURF, 'figure.facecolor': SURF})
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw=dict(hspace=.45))
    for i, inn in enumerate((1, 2)):
        z = x[x.innings == inn]; a = ax[i]; k = z.ball_no >= 10
        a.axhline(0, color=MUTED, lw=.8); a.plot(z.ball_no[k], z.mom10_std[k], color=INK, lw=1.8)
        for n, r in enumerate(pick.itertuples(), 1):
            if r.innings != inn: continue
            col = BLUE if r.mom > 0 else ORANGE
            a.axvspan(r.b0 - .5, r.b1 + .5, color=col, alpha=.15, lw=0)
            a.text((r.b0 + r.b1) / 2, a.get_ylim()[1], f'#{n}', ha='center', va='top', color=col, fontweight='bold')
        a.scatter(z.ball_no[z.is_wicket == 1], np.full((z.is_wicket == 1).sum(), a.get_ylim()[0]), marker='^', color=INK, s=35, zorder=5)
        a.set_xlim(0, 101); a.grid(axis='y', color=GRID, lw=.6); a.set_axisbelow(True); [a.spines[s].set_visible(False) for s in ('top', 'right')]
        a.set_ylabel('Model momentum, last 10 balls (SDs)')
        a.set_title(f"{'1st innings' if inn == 1 else 'Chase'}: {z.batting_team.iloc[0]} batting", loc='left', color=INK, fontsize=11, fontweight='bold', pad=14)
    ax[1].set_xlabel('Ball of the innings')
    fig.suptitle('Momentum highs (blue) and lows (orange), numbered as in the SHAP chart  ·  ▲ wicket', x=.06, ha='left', fontsize=12.5, fontweight='bold', color=INK, y=.99)
    fig.savefig(f'{fdir}/match_{mid}_phases.png', dpi=160, bbox_inches='tight'); plt.close(fig)

    # ---------- figure 2: one panel per highlighted block, SHAP by cricket group (runs over the block)
    n = len(rows)
    for cols, suffix, W in ((4, '', 19), (2, '_portrait', 12)):
        rws = int(np.ceil(n / cols))
        fig, ax = plt.subplots(rws, cols, figsize=(W, 4.6 * rws), squeeze=False); lim = max(np.abs(r['shap']).max() for r in rows) * 1.15
        for j, (a, r) in enumerate(zip(ax.flat, rows), 1):
            s = r['shap']; a.barh(s.index, s.values, color=[BLUE if v > 0 else ORANGE for v in s.values], height=.6)
            a.axvline(0, color=MUTED, lw=.8); a.set_xlim(-lim, lim); a.tick_params(axis='y', labelsize=8.5); a.grid(axis='x', color=GRID, lw=.6); a.set_axisbelow(True)
            [a.spines[q].set_visible(False) for q in ('top', 'right')]
            a.set_title(f"#{j}  {r['label']}  ·  momentum {r['mom']:+.1f}\n"
                        f"runs {r['runs']} (exp {r['e_runs']:.1f})  ·  4s/6s {r['bnd']} (exp {r['e_bnd']:.1f})\n"
                        f"dots {r['dot']} (exp {r['e_dot']:.1f})  ·  wickets {r['w']} (exp {r['e_w']:.1f})", loc='left', fontsize=8.8, color=BLUE if r['mom'] > 0 else ORANGE)
        for a in list(ax.flat)[n:]: a.axis('off')
        fig.suptitle('Why the model expected what it did in each high (blue) and low (orange)' + ('\n' if cols == 2 else ': ') + 'SHAP, in runs, by cricket factor', x=.01, ha='left', fontsize=13, fontweight='bold', color=INK)
        fig.supxlabel('← model expected FEWER runs because of this factor' + ('\n' if cols == 2 else '   ·   ') + 'model expected MORE runs because of this factor →   (runs over the 10-ball block)', fontsize=10, color=MUTED)
        fig.tight_layout(rect=(0, .02, 1, .96 if cols == 4 else .955), w_pad=3); fig.savefig(f'{fdir}/match_{mid}_shap{suffix}.png', dpi=140); plt.close(fig)

    # ---------- written report
    L = [f"# Why momentum moved: {m.team1} v {m.team2}, {pd.Timestamp(m.date):%d %b %Y}", '',
         f"Winner: {m.winner}. Momentum is what happened minus what the model expected, so every high or low has two parts: what happened, and why the model expected what it did (SHAP).",
         f"SHAP check: the runs-converted SHAP values track the LightGBM expectation with r = {check:.3f}.", '']
    for j, r in enumerate(rows, 1):
        top_up = [f"{k} ({v:+.2f})" for k, v in r['shap'].sort_values(ascending=False).items() if v > .05][:3]
        top_dn = [f"{k} ({v:+.2f})" for k, v in r['shap'].items() if v < -.05][:3]
        L += [f"## #{j} {r['label']}: {'HIGH' if r['mom'] > 0 else 'LOW'} for {r['team']} (momentum {r['mom']:+.2f} SD)", '',
              f"- **What happened:** {r['runs']} runs (model expected {r['e_runs']:.1f}); boundaries {r['bnd']} (expected {r['e_bnd']:.1f}); dots {r['dot']} (expected {r['e_dot']:.1f}); wickets {r['w']} (expected {r['e_w']:.2f}).",
              f"- **Who:** batting {r['batters']}; bowling {r['bowlers']}."]
        if r['innings'] == 2 and pd.notna(r['wp0']): L.append(f"- **Win chance** for {r['team']}: {r['wp0']:.0%} at the start of the block, {r['wp1']:.0%} at the end.")
        L += [f"- **What raised the model's expectation:** {', '.join(top_up) or 'nothing notable'}.",
              f"- **What lowered it:** {', '.join(top_dn) or 'nothing notable'}.", '']
    L += ["## Captain's decisions in this match, with the PSM evidence from all matches", '']
    if len(DEC):
        L += ['| Innings | After ball | Decision point | What the captain did | Next set momentum (SD) |', '|---|---|---|---|---|']
        for r in DEC.itertuples():
            L.append(f"| {'1st' if r.innings == 1 else 'Chase'} | {r.after_ball} | {r.decision} | {r.what} | {r.next_set:+.2f} |")
        h = DEC[DEC.decision.str.startswith('D3')]; fr = h.what.str.contains('fresh bowler')
        L += ['', f"**In this match:** {len(h)} hot spells; the captain brought on a fresh bowler after {int(fr.sum())} of them. "
              f"The next set's momentum averaged {h.next_set[fr].mean():+.2f} SD after a fresh bowler" + (f" and {h.next_set[~fr].mean():+.2f} SD otherwise." if (~fr).any() else ".")
              + ' One match is far too few to judge whether that worked.']
    else:
        L.append('- No D1 or D3 decision points in this match.')
    v3 = psm.loc['D3 fresh bowler after hot spell']; v1 = psm.loc['D1 keep v change bowler']
    L += ['', f"- **D3 across all {g} matches (PSM):** {int(v3.n_treated)} hot spells. The effect of a fresh bowler on the next set was {v3.att:+.2f} SD (95% interval {v3.lo:+.2f} to {v3.hi:+.2f}, p = {v3.p:.2f}). **No detectable effect.**",
          f"- **D1 across all {g} matches:** captains kept an expensive bowler on only {int(v1.n_treated)} times out of {int(v1.n_units)}, too few to estimate an effect.",
          '- One match can show *what was decided*. Whether it *worked* can only be judged across many matches, which is what PSM does.']
    rep = '\n'.join(L); open(f'{o}/match_{mid}_explained.md', 'w').write(rep); print(rep)
    print(f'\nfigures: {fdir}/match_{mid}_phases.png, {fdir}/match_{mid}_shap.png')


if __name__ == '__main__':
    explain(sys.argv[1] if len(sys.argv) > 1 else '1521261', sys.argv[2] if len(sys.argv) > 2 else 'male')
